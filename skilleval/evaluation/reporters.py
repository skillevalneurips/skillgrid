"""Result reporters: console, JSON, and CSV output."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from skilleval.core.types import EvalResult

logger = logging.getLogger(__name__)


class JSONReporter:
    """Write evaluation results to JSON files."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: list[EvalResult], filename: str = "results.json") -> Path:
        path = self.output_dir / filename
        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Results saved to %s", path)
        return path

    def save_traces(self, traces: list[Any], filename: str = "traces.json") -> Path:
        path = self.output_dir / filename
        data = [t.to_dict() if hasattr(t, "to_dict") else t for t in traces]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


class CSVReporter:
    """Write evaluation results to CSV."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: list[EvalResult], filename: str = "results.csv") -> Path:
        path = self.output_dir / filename
        if not results:
            return path

        fieldnames = [
            "experiment_id", "model_id", "dataset_id",
            "skill_origin", "runtime_policy",
            "visibility", "retrieval", "evolution",
            "success_rate", "avg_steps", "avg_cost", "avg_tokens",
            "recovery_rate", "selection_accuracy",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = r.to_dict()
                row.pop("per_pattern", None)
                row.pop("metadata", None)
                writer.writerow(row)
        logger.info("CSV results saved to %s", path)
        return path


class ConsoleReporter:
    """Pretty-print results to console."""

    @staticmethod
    def report(results: list[EvalResult]) -> None:
        if not results:
            print("No results to report.")
            return

        header = f"{'Experiment':<45} {'Success':>8} {'Steps':>7} {'Cost':>8} {'Recovery':>9}"
        print("\n" + "=" * 80)
        print("SKILLEVAL-BENCH RESULTS")
        print("=" * 80)
        print(header)
        print("-" * 80)

        for r in results:
            if r.visibility is not None:
                name = (
                    f"{r.dataset_id}/{r.model_id}"
                    f"/{r.visibility.value}/{r.retrieval.value if r.retrieval else '?'}"
                    f"/{r.evolution.value if r.evolution else '?'}"
                )
            else:
                name = f"{r.dataset_id}/{r.model_id}/{r.skill_origin.value}/{r.runtime_policy.value}"
            print(
                f"{name:<45} {r.success_rate:>7.1%} {r.avg_steps:>7.1f} "
                f"${r.avg_cost:>7.4f} {r.recovery_rate:>8.1%}"
            )

        print("-" * 80)
        print("=" * 80 + "\n")
