"""Evaluation metrics for skill composition benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skilleval.core.types import (
    EpisodeTrace,
    EvalResult,
    RuntimePolicy,
    SkillOrigin,
    SkillRetrieval,
    SkillVisibility,
    UpdateStrategy,
)


class MetricsComputer:
    """Compute aggregate and per-pattern metrics from episode traces."""

    def compute(
        self,
        traces: list[EpisodeTrace],
        predictions: dict[str, Any] | None = None,
        dataset_evaluator: Any | None = None,
        tasks: list[Any] | None = None,
        experiment_id: str = "",
        model_id: str = "",
        dataset_id: str = "",
        skill_origin: SkillOrigin = SkillOrigin.SPEC_DERIVED,
        runtime_policy: RuntimePolicy = RuntimePolicy.ORACLE_BUNDLE,
        visibility: SkillVisibility | None = None,
        retrieval: SkillRetrieval | None = None,
        evolution: UpdateStrategy | None = None,
    ) -> EvalResult:
        """Aggregate traces into an EvalResult."""
        if not traces:
            return EvalResult(
                experiment_id=experiment_id,
                model_id=model_id,
                dataset_id=dataset_id,
                skill_origin=skill_origin,
                runtime_policy=runtime_policy,
                visibility=visibility,
                retrieval=retrieval,
                evolution=evolution,
            )

        successes = [t for t in traces if t.success]
        success_rate = len(successes) / len(traces)

        avg_steps = sum(t.num_steps for t in traces) / len(traces)
        avg_cost = sum(t.total_cost for t in traces) / len(traces)
        avg_tokens = int(sum(t.total_tokens for t in traces) / len(traces))

        recovery_rate = self._compute_recovery_rate(traces)
        selection_accuracy = self._compute_selection_accuracy(traces)

        result = EvalResult(
            experiment_id=experiment_id,
            model_id=model_id,
            dataset_id=dataset_id,
            skill_origin=skill_origin,
            runtime_policy=runtime_policy,
            visibility=visibility,
            retrieval=retrieval,
            evolution=evolution,
            success_rate=success_rate,
            avg_steps=avg_steps,
            avg_cost=avg_cost,
            avg_tokens=avg_tokens,
            recovery_rate=recovery_rate,
            selection_accuracy=selection_accuracy,
        )
        result.metadata.update(self._aggregate_gold_scores(traces))
        return result

    @staticmethod
    def _compute_recovery_rate(traces: list[EpisodeTrace]) -> float:
        """Fraction of episodes with errors that ultimately succeeded."""
        episodes_with_errors = [t for t in traces if t.num_errors > 0]
        if not episodes_with_errors:
            return 0.0
        recovered = sum(1 for t in episodes_with_errors if t.success)
        return recovered / len(episodes_with_errors)

    @staticmethod
    def _compute_selection_accuracy(traces: list[EpisodeTrace]) -> float:
        """Placeholder for skill selection accuracy measurement.

        Override with dataset-specific gold skill mappings.
        """
        return 0.0

    @staticmethod
    def _aggregate_gold_scores(traces: list[EpisodeTrace]) -> dict[str, Any]:
        """Average numeric dataset-specific scores stored on traces.

        Dataset adapters return arbitrary metric keys from
        ``evaluate_prediction()``. The evaluator stores that dict under
        ``trace.metadata["gold_score"]``; this helper lifts numeric metrics
        into result metadata so dataset-specific scores survive aggregation.
        """
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for trace in traces:
            score = trace.metadata.get("gold_score", {})
            if not isinstance(score, dict):
                continue
            for key, value in score.items():
                if key == "success" or not isinstance(value, (int, float)):
                    continue
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1

        if not totals:
            return {}

        aggregated = {
            f"mean_{key}": totals[key] / counts[key]
            for key in sorted(totals)
            if counts[key]
        }
        aggregated["gold_score_count"] = max(counts.values()) if counts else 0
        return {"dataset_metrics": aggregated}
