"""Abstract base class for LLM backends."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from skilleval.core.config import Config


@dataclass
class ModelResponse:
    """Standardized response from any LLM backend."""
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    raw: Any = None  # provider-specific raw response


class BaseModel(ABC):
    """Interface every model provider must implement.

    To add a new provider:
    1. Subclass ``BaseModel``.
    2. Implement ``generate`` and ``generate_with_tools``.
    3. Register with ``@model_registry.register("name")``.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model_id: str = config.get("model.model_id", "unknown")
        self.max_tokens: int = config.get("model.max_tokens", 4096)
        self.temperature: float = config.get("model.temperature", 0.0)
        self._total_cost: float = 0.0
        self._total_tokens: int = 0

    # -- required overrides --------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a text completion."""

    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response that may include tool calls."""

    # -- shared bookkeeping --------------------------------------------------

    def _track(self, resp: ModelResponse) -> ModelResponse:
        self._total_cost += resp.cost
        self._total_tokens += resp.input_tokens + resp.output_tokens
        return resp

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def reset_counters(self) -> None:
        self._total_cost = 0.0
        self._total_tokens = 0

    @staticmethod
    def _timer() -> float:
        return time.perf_counter()
