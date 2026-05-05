"""Local / self-hosted model backend via OpenAI-compatible API (vLLM, Ollama, etc.)."""

from __future__ import annotations

import logging
import os
from typing import Any

from skilleval.core.config import Config
from skilleval.core.registry import model_registry
from skilleval.models.base import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


@model_registry.register("local")
class LocalModel(BaseModel):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._base_url = config.get("model.base_url", "http://localhost:8000/v1")
        api_key_env = config.get("model.api_key_env")
        self._api_key = os.environ.get(api_key_env, "dummy") if api_key_env else "dummy"
        self._client = None

    @property
    def name(self) -> str:
        return f"local/{self.model_id}"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self._base_url,
                    api_key=self._api_key,
                )
            except ImportError:
                raise ImportError("pip install openai")
        return self._client

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        client = self._get_client()
        t0 = self._timer()
        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        latency = (self._timer() - t0) * 1000
        choice = response.choices[0]
        usage = response.usage or type("Usage", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        resp = ModelResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost=0.0,
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
        client = self._get_client()
        t0 = self._timer()
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=tools,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )
        except Exception:
            logger.warning("Local model may not support tool calling; falling back to text-only.")
            return self.generate(messages, **kwargs)
        latency = (self._timer() - t0) * 1000
        choice = response.choices[0]
        usage = response.usage or type("Usage", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })
        resp = ModelResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost=0.0,
            latency_ms=latency,
            raw=response,
        )
        return self._track(resp)
