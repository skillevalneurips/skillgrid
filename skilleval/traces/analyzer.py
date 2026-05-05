"""Trace analyzer: extract patterns, failures, and statistics from traces."""

from __future__ import annotations

from collections import Counter
from typing import Any

from skilleval.core.types import EpisodeTrace


class TraceAnalyzer:
    """Analyze episode traces for skill creation and evaluation insights."""

    def __init__(self, traces: list[EpisodeTrace]) -> None:
        self.traces = traces

    def tool_usage_stats(self) -> dict[str, int]:
        """Count how often each tool is called across all traces."""
        counter: Counter[str] = Counter()
        for trace in self.traces:
            for entry in trace.entries:
                if entry.tool_name:
                    counter[entry.tool_name] += 1
        return dict(counter.most_common())

    def skill_usage_stats(self) -> dict[str, int]:
        """Count how often each skill is used across all traces."""
        counter: Counter[str] = Counter()
        for trace in self.traces:
            for entry in trace.entries:
                if entry.skill_used:
                    counter[entry.skill_used] += 1
        return dict(counter.most_common())

    def error_analysis(self) -> dict[str, Any]:
        """Analyze error patterns across traces."""
        error_types: Counter[str] = Counter()
        error_tools: Counter[str] = Counter()
        recovery_sequences: list[list[str]] = []

        for trace in self.traces:
            in_error = False
            current_recovery: list[str] = []
            for entry in trace.entries:
                if not entry.success:
                    in_error = True
                    error_key = f"{entry.tool_name}:{entry.error or 'unknown'}"
                    error_types[error_key] += 1
                    if entry.tool_name:
                        error_tools[entry.tool_name] += 1
                    current_recovery = [entry.action]
                elif in_error:
                    current_recovery.append(entry.action)
                    if entry.success:
                        recovery_sequences.append(current_recovery)
                        in_error = False
                        current_recovery = []

        return {
            "total_errors": sum(error_types.values()),
            "error_types": dict(error_types.most_common(20)),
            "error_tools": dict(error_tools.most_common(10)),
            "recovery_sequences": recovery_sequences[:10],
            "avg_errors_per_episode": (
                sum(t.num_errors for t in self.traces) / max(len(self.traces), 1)
            ),
        }

    def common_tool_sequences(self, window_size: int = 3, top_k: int = 10) -> list[tuple[tuple[str, ...], int]]:
        """Find the most common consecutive tool-call sequences."""
        counter: Counter[tuple[str, ...]] = Counter()
        for trace in self.traces:
            tools = [e.tool_name for e in trace.entries if e.tool_name]
            for i in range(len(tools) - window_size + 1):
                seq = tuple(tools[i : i + window_size])
                counter[seq] += 1
        return counter.most_common(top_k)

    def episode_statistics(self) -> dict[str, Any]:
        """Compute summary statistics across all episodes."""
        if not self.traces:
            return {}
        steps = [t.num_steps for t in self.traces]
        costs = [t.total_cost for t in self.traces]
        tokens = [t.total_tokens for t in self.traces]
        successes = [t.success for t in self.traces]
        return {
            "total_episodes": len(self.traces),
            "success_rate": sum(successes) / len(successes),
            "steps_mean": sum(steps) / len(steps),
            "steps_max": max(steps),
            "steps_min": min(steps),
            "cost_total": sum(costs),
            "cost_mean": sum(costs) / len(costs),
            "tokens_total": sum(tokens),
            "tokens_mean": sum(tokens) / len(tokens),
        }
