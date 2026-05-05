"""Trace collector: captures interaction traces during agent episodes."""

from __future__ import annotations

import time
from typing import Any

from skilleval.core.types import EpisodeTrace, TraceEntry


class TraceCollector:
    """Wraps agent execution to capture detailed interaction traces.

    Hooks into tool calls, observations, and state transitions to
    build EpisodeTrace objects for downstream analysis.
    """

    def __init__(self) -> None:
        self._current_entries: list[TraceEntry] = []
        self._step_counter: int = 0
        self._start_time: float = 0.0

    def start_episode(self, task_id: str, model_id: str) -> None:
        self._current_entries = []
        self._step_counter = 0
        self._start_time = time.perf_counter()
        self._task_id = task_id
        self._model_id = model_id

    def record_step(
        self,
        action: str,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        observation: str = "",
        success: bool = True,
        error: str | None = None,
        skill_used: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> TraceEntry:
        """Record a single interaction step."""
        entry = TraceEntry(
            step=self._step_counter,
            action=action,
            tool_name=tool_name,
            tool_args=tool_args or {},
            observation=observation,
            state=state or {},
            success=success,
            error=error,
            latency_ms=(time.perf_counter() - self._start_time) * 1000,
            skill_used=skill_used,
        )
        self._current_entries.append(entry)
        self._step_counter += 1
        return entry

    def finish_episode(
        self,
        success: bool,
        total_cost: float = 0.0,
        total_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeTrace:
        """Finalize and return the episode trace."""
        return EpisodeTrace(
            task_id=self._task_id,
            model_id=self._model_id,
            entries=list(self._current_entries),
            total_cost=total_cost,
            total_tokens=total_tokens,
            success=success,
            metadata=metadata or {},
        )
