"""Google Gemini API model backend."""

from __future__ import annotations

import logging
import os
from typing import Any

from skilleval.core.config import Config
from skilleval.core.registry import model_registry
from skilleval.models.base import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


@model_registry.register("google")
class GoogleModel(BaseModel):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        api_key_env = config.get("model.api_key_env", "GOOGLE_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")
        self._cost_in = config.get("model.cost_per_1k_input", 0.001)
        self._cost_out = config.get("model.cost_per_1k_output", 0.002)
        self._client = None

    @property
    def name(self) -> str:
        return f"google/{self.model_id}"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._client = genai.GenerativeModel(self.model_id)
            except ImportError:
                raise ImportError("pip install google-generativeai")
        return self._client

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        client = self._get_client()
        prompt = self._messages_to_prompt(messages)
        t0 = self._timer()
        response = client.generate_content(prompt)
        latency = (self._timer() - t0) * 1000

        text = response.text if response.text else ""
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) if hasattr(response, "usage_metadata") else 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) if hasattr(response, "usage_metadata") else 0
        cost = (input_tokens / 1000 * self._cost_in
                + output_tokens / 1000 * self._cost_out)
        resp = ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency,
            raw=response,
        )
        return self._track(resp)

    def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        logger.warning(
            "Google tool calling uses a different schema; "
            "falling back to prompt-based tool use."
        )
        tool_desc = "\n".join(
            f"- {t.get('function', t).get('name', 'unknown')}: "
            f"{t.get('function', t).get('description', '')}"
            for t in tools
        )
        messages = list(messages)
        messages[0] = {
            "role": messages[0]["role"],
            "content": messages[0]["content"]
            + f"\n\nAvailable tools:\n{tool_desc}\n\n"
            "Respond with JSON tool calls when needed.",
        }
        return self.generate(messages, **kwargs)

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m["role"].upper()
            parts.append(f"{role}: {m['content']}")
        return "\n\n".join(parts)
