"""
Lightweight LLM engines for OpenAI and Grok (xAI).

Design goals:
- No new third-party deps (stdlib-only HTTP).
- OpenAI-compatible Chat Completions API.
- Simple, reusable interface: engine.chat(messages, ...) -> str

Environment variables:
- LLM_PROVIDER: "openai" | "grok" | "xai" (used by callers, optional)
OpenAI:
- OPENAI_API_KEY (required)
- OPENAI_BASE_URL (default: https://api.openai.com/v1)
- OPENAI_MODEL (optional default model)
Grok / xAI:
- XAI_API_KEY or GROK_API_KEY (required)
- XAI_BASE_URL (default: https://api.x.ai/v1)
- GROK_MODEL or XAI_MODEL (optional default model)
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

_logger = logging.getLogger(__name__)


class LLMEngineError(RuntimeError):
    """Raised when an LLM provider call fails."""


def _env_any(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


# HTTP status codes that should trigger a retry.
_RETRYABLE_STATUS_CODES = {429, 502, 503, 529}

# Default retry configuration.
_MAX_RETRIES = 7
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 120.0


def _post_json(
    *,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")

    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES + 1):
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={
                "User-Agent": "skill-r1/1.0",
                **dict(headers),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                try:
                    return json.loads(body)
                except json.JSONDecodeError as e:
                    raise LLMEngineError(f"Invalid JSON from provider: {e}\nBody:\n{body[:2000]}") from e

        except urllib.error.HTTPError as e:
            status = getattr(e, "code", None)
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            last_exc = LLMEngineError(
                f"Provider HTTP {status or 'unknown'}: {getattr(e, 'reason', '')}\n{body[:4000]}"
            )
            last_exc.__cause__ = e

            if status in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                # Respect Retry-After header if the server provides one.
                retry_after = None
                if hasattr(e, "headers"):
                    retry_after_hdr = e.headers.get("Retry-After")
                    if retry_after_hdr is not None:
                        try:
                            retry_after = float(retry_after_hdr)
                        except (ValueError, TypeError):
                            retry_after = None

                backoff = min(_BASE_BACKOFF_S * (2 ** attempt), _MAX_BACKOFF_S)
                wait = max(retry_after or 0, backoff)
                _logger.warning(
                    "Rate-limited (HTTP %s) on attempt %d/%d. Retrying in %.1fs...",
                    status, attempt + 1, _MAX_RETRIES + 1, wait,
                )
                time.sleep(wait)
                continue

            raise last_exc from e

        except (urllib.error.URLError, socket.timeout) as e:
            last_exc = LLMEngineError(f"Provider request failed: {e}")
            last_exc.__cause__ = e

            # Retry on connection-level transient errors too.
            if attempt < _MAX_RETRIES:
                backoff = min(_BASE_BACKOFF_S * (2 ** attempt), _MAX_BACKOFF_S)
                _logger.warning(
                    "Connection error on attempt %d/%d: %s. Retrying in %.1fs...",
                    attempt + 1, _MAX_RETRIES + 1, e, backoff,
                )
                time.sleep(backoff)
                continue

            raise last_exc from e

    # Should not be reached, but just in case:
    raise last_exc or LLMEngineError("Max retries exceeded.")


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    # allow passing https://api.openai.com for convenience
    return base_url + "/v1"


@dataclass
class ChatCompletionResult:
    content: str
    raw: Dict[str, Any]


class BaseChatEngine:
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleChatEngine(BaseChatEngine):
    """
    Engine for OpenAI-compatible Chat Completions APIs.
    Works for both OpenAI and xAI (Grok) endpoints.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float = 120.0,
        extra_headers: Optional[Mapping[str, str]] = None,
        is_reasoning_model: bool = False,
    ):
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = _normalize_base_url(base_url)
        self.timeout_s = float(timeout_s)
        self.extra_headers = dict(extra_headers or {})
        self.is_reasoning_model = is_reasoning_model

        if not self.api_key:
            raise LLMEngineError("Missing API key.")
        if not self.model:
            raise LLMEngineError("Missing model name.")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.is_reasoning_model:
            payload["max_completion_tokens"] = int(max_new_tokens)
        else:
            payload["temperature"] = float(temperature)
            payload["top_p"] = float(top_p)
            payload["max_completion_tokens"] = int(max_new_tokens)

        data = _post_json(url=url, headers=self._headers(), payload=payload, timeout_s=self.timeout_s)

        # OpenAI-compatible response typically:
        # { choices: [ { message: { role, content }, ... } ], ... }
        try:
            choices = data.get("choices") or []
            if not choices:
                raise KeyError("choices")
            choice0 = choices[0] or {}
            message = choice0.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip() != "":
                return content.strip()

            # fallback: some providers / modes may return "text"
            text = choice0.get("text")
            if isinstance(text, str) and text.strip() != "":
                return text.strip()
        except Exception as e:
            raise LLMEngineError(f"Unexpected provider response shape: {e}\nRaw:\n{json.dumps(data)[:4000]}") from e

        raise LLMEngineError(f"Empty response from provider.\nRaw:\n{json.dumps(data)[:4000]}")


def create_engine(provider: str, *, model: Optional[str] = None) -> BaseChatEngine:
    """
    Create an engine by provider name.

    Args:
        provider: "openai" | "grok" | "xai"
        model: optional override; if not provided, uses provider env defaults.
    """
    p = (provider or "").strip().lower()

    if p == "openai":
        api_key = _env_any("OPENAI_API_KEY")
        if not api_key:
            raise LLMEngineError("OPENAI_API_KEY is required for provider=openai.")
        base_url = _env_any("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model_name = model or _env_any("OPENAI_MODEL") or "gpt-4o-mini"
        return OpenAICompatibleChatEngine(api_key=api_key, model=model_name, base_url=base_url)

    if p in ("grok", "xai"):
        api_key = _env_any("XAI_API_KEY", "GROK_API_KEY")
        if not api_key:
            raise LLMEngineError("XAI_API_KEY (or GROK_API_KEY) is required for provider=grok/xai.")
        base_url = _env_any("XAI_BASE_URL") or "https://api.x.ai/v1"
        model_name = model or _env_any("GROK_MODEL", "XAI_MODEL") or "grok-2-latest"
        timeout = float(_env_any("XAI_TIMEOUT") or "300")
        is_reasoning = "reasoning" in model_name.lower() or "grok-4" in model_name.lower()
        return OpenAICompatibleChatEngine(
            api_key=api_key, model=model_name, base_url=base_url, timeout_s=timeout,
            is_reasoning_model=is_reasoning,
        )

    raise LLMEngineError(f"Unknown provider: {provider!r}. Expected 'openai' or 'grok'/'xai'.")

