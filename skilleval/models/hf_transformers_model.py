"""Optional HuggingFace Transformers backend for local chat inference."""

from __future__ import annotations

import logging
from typing import Any

from skilleval.core.config import Config
from skilleval.core.registry import model_registry
from skilleval.models.base import BaseModel, ModelResponse
from skilleval.utils.compat import check_transformers_trl_compat

logger = logging.getLogger(__name__)


@model_registry.register("hf_transformers")
class HFTransformersModel(BaseModel):
    """Local chat backend based on ``AutoModelForCausalLM``.

    This follows the same prompt/build/generate/decode pattern as the Qwen
    WebWalker script, so Qwen instruct checkpoints can be used interchangeably
    with API-backed models in SkillEval protocols.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        check_transformers_trl_compat(strict=True)
        self._model_path = config.get("model.model_path", self.model_id)
        self._dtype = config.get("model.dtype", "auto")
        self._device_map = config.get("model.device_map", "auto")
        self._gpu_id = int(config.get("model.gpu_id", 0))
        self._trust_remote_code = bool(config.get("model.trust_remote_code", True))
        self._single_gpu = bool(config.get("model.single_gpu", True))
        self._top_p = float(config.get("model.top_p", 1.0))
        self._tokenizer = None
        self._model = None

    @property
    def name(self) -> str:
        return f"hf_transformers/{self.model_id}"

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for hf_transformers backend. "
                "Install with: pip install transformers torch accelerate"
            ) from exc

        logger.info("Loading HF model: %s", self._model_path)
        torch_dtype = self._resolve_dtype(torch)
        device_map = self._resolve_device_map(torch)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=self._trust_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=self._trust_remote_code,
            )
        except ValueError as exc:
            msg = str(exc)
            if "model type `qwen3`" in msg or "KeyError: 'qwen3'" in msg:
                raise RuntimeError(
                    "Failed to load a Qwen3 model because this Transformers "
                    "build does not recognize `model_type=qwen3`.\n"
                    f"- transformers=={transformers.__version__}\n\n"
                    "Fix options:\n"
                    "- Upgrade: `pip install -U --pre transformers accelerate`\n"
                    "- Or use a Qwen2.5 model id such as "
                    "`Qwen/Qwen2.5-7B-Instruct`."
                ) from exc
            raise

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    def _resolve_dtype(self, torch: Any) -> Any:
        dtype = str(self._dtype).lower()
        if dtype == "auto":
            return None
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        if dtype == "float32":
            return torch.float32
        raise ValueError("model.dtype must be one of: auto, float16, bfloat16, float32")

    def _resolve_device_map(self, torch: Any) -> Any:
        setting = str(self._device_map).lower()
        if setting == "cpu":
            return None
        if torch.cuda.is_available() and self._single_gpu:
            torch.cuda.set_device(self._gpu_id)
            return {"": f"cuda:{self._gpu_id}"}
        return self._device_map

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return self._messages_to_prompt(messages)

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        self._load()
        import torch

        prompt = self._format_messages(messages)
        inputs = self._tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._model.device)

        temperature = float(kwargs.get("temperature", self.temperature))
        do_sample = temperature > 0.0
        generate_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = float(kwargs.get("top_p", self._top_p))

        t0 = self._timer()
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generate_kwargs)
        latency = (self._timer() - t0) * 1000

        new_token_ids = output_ids[0][inputs.input_ids.shape[1]:]
        text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        return self._track(
            ModelResponse(
                text=text,
                input_tokens=int(inputs.input_ids.shape[1]),
                output_tokens=int(new_token_ids.shape[0]),
                cost=0.0,
                latency_ms=latency,
                raw=output_ids,
            )
        )

    def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        logger.warning(
            "hf_transformers backend does not support native tool calling; "
            "falling back to prompt-based tool-use emulation."
        )
        tool_lines = []
        for t in tools:
            fn = t.get("function", t)
            tool_lines.append(f"- {fn.get('name', 'unknown')}: {fn.get('description', '')}")
        messages = list(messages)
        if messages:
            messages[0] = {
                "role": messages[0]["role"],
                "content": (
                    messages[0]["content"]
                    + "\n\nAvailable tools:\n"
                    + "\n".join(tool_lines)
                    + "\nRespond with JSON-like tool calls when needed."
                ),
            }
        return self.generate(messages, **kwargs)

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def create_hf_transformers_model(
    model_id: str,
    model_path: str | None = None,
    dtype: str = "auto",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device_map: str = "auto",
    gpu_id: int = 0,
    trust_remote_code: bool = True,
    single_gpu: bool = True,
) -> HFTransformersModel:
    """Convenience factory for dataset-local HF model definitions."""
    config = Config({
        "model": {
            "provider": "hf_transformers",
            "model_id": model_id,
            "model_path": model_path or model_id,
            "dtype": dtype,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "device_map": device_map,
            "gpu_id": gpu_id,
            "trust_remote_code": trust_remote_code,
            "single_gpu": single_gpu,
        }
    })
    return HFTransformersModel(config)
