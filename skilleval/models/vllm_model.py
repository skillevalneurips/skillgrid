"""vLLM offline inference backend for fast local model evaluation."""

from __future__ import annotations

import logging
from typing import Any

from skilleval.core.config import Config
from skilleval.core.registry import model_registry
from skilleval.models.base import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


@model_registry.register("vllm")
class VLLMModel(BaseModel):
    """vLLM-based model for fast offline inference.
    
    Config options:
        model.model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-4B")
        model.dtype: Data type ("bfloat16", "float16", "auto")
        model.max_tokens: Max output tokens (default: 2048)
        model.temperature: Sampling temperature (default: 0.0)
        model.gpu_memory_utilization: GPU memory fraction (default: 0.9)
        model.tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
        model.trust_remote_code: Trust remote code (default: True)
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._llm = None
        self._tokenizer = None
        
        # vLLM-specific config
        self._dtype = config.get("model.dtype", "bfloat16")
        self._gpu_memory_utilization = float(config.get("model.gpu_memory_utilization", 0.9))
        self._tensor_parallel_size = int(config.get("model.tensor_parallel_size", 1))
        self._trust_remote_code = config.get("model.trust_remote_code", True)
        self._max_model_len = config.get("model.max_model_len", None)  # None = use model default

    @property
    def name(self) -> str:
        return f"vllm/{self.model_id}"

    def _load(self) -> None:
        """Lazy-load vLLM model on first use."""
        if self._llm is not None:
            return
            
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")
        
        logger.info("Loading vLLM model: %s (dtype=%s, gpu_util=%.2f)", 
                    self.model_id, self._dtype, self._gpu_memory_utilization)
        
        llm_kwargs = dict(
            model=self.model_id,
            dtype=self._dtype,
            gpu_memory_utilization=self._gpu_memory_utilization,
            tensor_parallel_size=self._tensor_parallel_size,
            trust_remote_code=self._trust_remote_code,
        )
        if self._max_model_len is not None:
            llm_kwargs["max_model_len"] = int(self._max_model_len)
        
        self._llm = LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        logger.info("vLLM model loaded successfully")

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate response using vLLM."""
        self._load()
        
        from vllm import SamplingParams
        
        # Apply chat template (disable thinking mode for Qwen3 models)
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Disable Qwen3 thinking mode
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # Build sampling params
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            stop=kwargs.get("stop", None),
        )
        
        t0 = self._timer()
        outputs = self._llm.generate([prompt], sampling_params)
        latency = (self._timer() - t0) * 1000
        
        output = outputs[0]
        response_text = output.outputs[0].text
        
        # Token counts
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        
        return self._track(ModelResponse(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=0.0,  # Local model, no cost
            latency_ms=latency,
        ))

    def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate with tool descriptions injected into prompt.
        
        vLLM doesn't have native tool calling, so we inject tool
        descriptions into the system prompt.
        """
        tool_lines = []
        for t in tools:
            fn = t.get("function", t)
            tool_lines.append(f"- {fn.get('name', 'unknown')}: {fn.get('description', '')}")

        tool_block = "\n\nAvailable tools:\n" + "\n".join(tool_lines)
        tool_block += "\n\nTo use a tool, respond with JSON: {\"tool\": \"name\", \"input\": \"...\"}"

        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            messages[0] = {**messages[0], "content": messages[0]["content"] + tool_block}
        else:
            messages.insert(0, {"role": "system", "content": tool_block.strip()})

        return self.generate(messages, **kwargs)


def create_vllm_model(
    model_id: str,
    dtype: str = "bfloat16",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    max_model_len: int | None = None,
) -> VLLMModel:
    """Convenience function to create a vLLM model.
    
    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-4B")
        dtype: Data type ("bfloat16", "float16", "auto")
        max_tokens: Maximum output tokens
        temperature: Sampling temperature (0 = greedy)
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        trust_remote_code: Whether to trust remote code in model repo
        max_model_len: Maximum context length (None = use model default)
    
    Returns:
        Configured VLLMModel instance
    """
    config = Config({
        "model": {
            "provider": "vllm",
            "model_id": model_id,
            "dtype": dtype,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
        }
    })
    return VLLMModel(config)
