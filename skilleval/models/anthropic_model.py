"""Anthropic API model backend (Claude)."""

from __future__ import annotations

import logging
import os
from typing import Any

from skilleval.core.config import Config
from skilleval.core.registry import model_registry
from skilleval.models.base import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


@model_registry.register("anthropic")
class AnthropicModel(BaseModel):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        api_key_env = config.get("model.api_key_env", "ANTHROPIC_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")
        self._cost_in = config.get("model.cost_per_1k_input", 0.003)
        self._cost_out = config.get("model.cost_per_1k_output", 0.015)
        self._client = None

    @property
    def name(self) -> str:
        return f"anthropic/{self.model_id}"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError("pip install anthropic")
        return self._client

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        client = self._get_client()
        system_msg = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_msgs.append(m)

        t0 = self._timer()
        response = client.messages.create(
            model=self.model_id,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system_msg,
            messages=user_msgs,
            temperature=kwargs.get("temperature", self.temperature),
        )
        latency = (self._timer() - t0) * 1000
        text = response.content[0].text if response.content else ""
        cost = (response.usage.input_tokens / 1000 * self._cost_in
                + response.usage.output_tokens / 1000 * self._cost_out)
        resp = ModelResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
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
        client = self._get_client()
        system_msg = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_msgs.append(m)

        anthropic_tools = self._convert_tools(tools)
        t0 = self._timer()
        response = client.messages.create(
            model=self.model_id,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system_msg,
            messages=user_msgs,
            tools=anthropic_tools,
            temperature=kwargs.get("temperature", self.temperature),
        )
        latency = (self._timer() - t0) * 1000
        text = ""
        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
        cost = (response.usage.input_tokens / 1000 * self._cost_in
                + response.usage.output_tokens / 1000 * self._cost_out)
        resp = ModelResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost=cost,
            latency_ms=latency,
            raw=response,
        )
        return self._track(resp)

    @staticmethod
    def _convert_tools(openai_tools: list[dict]) -> list[dict]:
        """Convert OpenAI-format tool defs to Anthropic format."""
        result = []
        for t in openai_tools:
            func = t.get("function", t)
            result.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            })
        return result
