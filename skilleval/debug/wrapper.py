"""DebugModel: transparent proxy that tees every LLM call to disk.

Wraps any ``BaseModel`` instance. Proxies ``generate`` and
``generate_with_tools`` to the underlying model and writes one JSON line
per call to ``<log_dir>/agent.jsonl``. No truncation.

All other attribute access (``model_id``, ``total_cost``, …) falls
through via ``__getattr__`` so callers cannot tell the wrapper apart
from the wrapped model.

Remove-and-done story: this module is purely additive. Comment out the
wrap call in ``run.py`` and the runtime is identical to pre-debug.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from skilleval.models.base import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


class DebugModel(BaseModel):
    """Proxy model that records every ``generate`` call to disk."""

    def __init__(self, wrapped: BaseModel, log_dir: str | Path) -> None:
        # Do NOT call super().__init__() — we're a pass-through, not a
        # real model. All attrs fall through to ``_wrapped`` via __getattr__.
        self._wrapped = wrapped
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._log_dir / "agent.jsonl"
        self._call_idx = 0

    # ------------------------------------------------------------------
    # BaseModel required overrides — delegate + log
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return getattr(self._wrapped, "name", self._wrapped.__class__.__name__)

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        response = self._wrapped.generate(messages, **kwargs)
        self._dump("generate", messages, response, kwargs, tools=None)
        return response

    def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        response = self._wrapped.generate_with_tools(messages, tools, **kwargs)
        self._dump("generate_with_tools", messages, response, kwargs, tools=tools)
        return response

    # ------------------------------------------------------------------
    # Transparent delegation for everything else
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only invoked for attrs NOT found on DebugModel itself.
        return getattr(self._wrapped, name)

    # ------------------------------------------------------------------
    # Internal: write one jsonl line per call
    # ------------------------------------------------------------------

    def _dump(
        self,
        method: str,
        messages: list[dict[str, Any]],
        response: ModelResponse,
        kwargs: dict[str, Any],
        tools: list[dict[str, Any]] | None,
    ) -> None:
        record: dict[str, Any] = {
            "call_idx": self._call_idx,
            "ts": time.time(),
            "method": method,
            "model_id": getattr(self._wrapped, "model_id", "unknown"),
            "messages": messages,
            "response_text": response.text,
            "response_tool_calls": response.tool_calls,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "kwargs": _jsonable(kwargs),
        }
        if tools is not None:
            record["tools"] = tools
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str))
                f.write("\n")
        except Exception as exc:
            logger.warning("DebugModel failed to write jsonl: %s", exc)
        self._call_idx += 1


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion for kwargs that may contain non-JSON types."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return {k: str(v) for k, v in (obj or {}).items()}
