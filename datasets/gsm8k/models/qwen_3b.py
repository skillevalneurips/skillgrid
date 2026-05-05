"""Qwen 2.5-3B-Instruct — runs locally on GPU via transformers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from skilleval.core.config import Config
from skilleval.models.base import BaseModel, ModelResponse


MODEL_NAME = "qwen-3b"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


class Qwen3BModel(BaseModel):
    """Local Qwen 2.5-3B-Instruct model."""

    def __init__(self) -> None:
        super().__init__(Config({"model": {"model_id": MODEL_ID, "max_tokens": 512}}))
        self._tokenizer = None
        self._model = None

    @property
    def name(self) -> str:
        return f"local/{MODEL_ID}"

    def _load(self):
        """Lazy-load model and tokenizer on first use."""
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            self._model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    def generate(self, messages, **kwargs):
        self._load()
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        t0 = self._timer()

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                do_sample=False,
            )

        latency = (self._timer() - t0) * 1000
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._track(ModelResponse(
            text=response_text,
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(new_tokens),
            cost=0.0,
            latency_ms=latency,
        ))

    def generate_with_tools(self, messages, tools, **kwargs):
        # Qwen 3B doesn't support native tool calling.
        # Inject tool descriptions into the system prompt.
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


def create_model() -> Qwen3BModel:
    return Qwen3BModel()
