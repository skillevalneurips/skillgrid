"""OpenAI API model backend (GPT-4o, GPT-4o-mini, etc.)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from skilleval.core.config import Config
from skilleval.core.registry import model_registry
from skilleval.models.base import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


def _call_with_retry(fn, *, retries: int = 3, backoff: float = 2.0):
    """Call *fn* with exponential backoff on transient failures.

    Retries 400/408/409/429/5xx and connection-type errors. Non-transient
    errors (auth, 404) propagate immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            transient = (
                isinstance(status, int) and (status == 400 or status == 408
                                             or status == 409 or status == 429
                                             or 500 <= status < 600)
            ) or exc.__class__.__name__ in {
                "APIConnectionError", "APITimeoutError", "RateLimitError",
                "InternalServerError", "BadRequestError",
            }
            if not transient or attempt == retries:
                raise
            sleep = backoff ** attempt
            logger.warning(
                "OpenAI call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, retries + 1, exc, sleep,
            )
            time.sleep(sleep)
    raise last_exc  # unreachable


@model_registry.register("openai")
class OpenAIModel(BaseModel):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        api_key_env = config.get("model.api_key_env", "OPENAI_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")
        self._cost_in = config.get("model.cost_per_1k_input", 0.0025)
        self._cost_out = config.get("model.cost_per_1k_output", 0.01)
        self._use_max_completion_tokens = bool(config.get("model.use_max_completion_tokens", False))
        self._reasoning_effort = config.get("model.reasoning_effort")
        self._omit_temperature = bool(config.get("model.omit_temperature", False))
        self._client = None

    @property
    def name(self) -> str:
        return f"openai/{self.model_id}"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
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
        request = self._chat_request_kwargs(messages=messages, kwargs=kwargs)
        response = _call_with_retry(lambda: client.chat.completions.create(**request))
        latency = (self._timer() - t0) * 1000
        choice = response.choices[0]
        usage = response.usage
        cost = (usage.prompt_tokens / 1000 * self._cost_in
                + usage.completion_tokens / 1000 * self._cost_out)
        resp = ModelResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
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
        # OpenAI rejects tools=[]; fall back to plain generate when empty.
        if not tools:
            return self.generate(messages, **kwargs)
        client = self._get_client()
        t0 = self._timer()
        request = self._chat_request_kwargs(messages=messages, kwargs=kwargs, tools=tools)
        response = _call_with_retry(lambda: client.chat.completions.create(**request))
        latency = (self._timer() - t0) * 1000
        choice = response.choices[0]
        usage = response.usage
        cost = (usage.prompt_tokens / 1000 * self._cost_in
                + usage.completion_tokens / 1000 * self._cost_out)
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
            cost=cost,
            latency_ms=latency,
            raw=response,
        )
        return self._track(resp)

    def _chat_request_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        kwargs: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }
        if tools:
            request["tools"] = tools

        max_completion_tokens = kwargs.get("max_completion_tokens")
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        use_mct = (
            self._use_max_completion_tokens
            or max_completion_tokens is not None
            or self.model_id.startswith("gpt-5")
            or self.model_id.startswith("o1")
        )
        if use_mct:
            request["max_completion_tokens"] = max_completion_tokens or max_tokens
        else:
            request["max_tokens"] = max_tokens

        reasoning_effort = kwargs.get("reasoning_effort", self._reasoning_effort)
        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort

        temperature = kwargs.get("temperature", self.temperature)
        if not self._omit_temperature and temperature is not None:
            request["temperature"] = temperature

        return request
