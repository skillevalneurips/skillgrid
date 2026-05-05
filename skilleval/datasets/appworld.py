"""AppWorld dataset adapter: multi-app API-based task execution."""

from __future__ import annotations

import logging
from typing import Any

from skilleval.core.registry import dataset_registry
from skilleval.core.types import TaskDomain, TaskInstance
from skilleval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


@dataset_registry.register("appworld")
class AppWorldDataset(BaseDataset):

    @property
    def name(self) -> str:
        return "appworld"

    @property
    def domain(self) -> TaskDomain:
        return TaskDomain.EXECUTION_ENV

    def load(self) -> None:
        split = self.config.get("dataset.split", "test_normal")
        max_samples = self.config.get("dataset.max_samples")

        try:
            self._load_appworld_tasks(split, max_samples)
        except Exception:
            logger.warning("AppWorld environment not available; using stub data.")
            for idx, item in enumerate(self._stub_data()):
                if max_samples and idx >= max_samples:
                    break
                self._tasks.append(
                    TaskInstance(
                        task_id=f"appworld_{idx}",
                        domain=self.domain,
                        instruction=item["instruction"],
                        composition_pattern=self._assign_pattern(item),
                        tools_required=["api_call", "code_execute"],
                        gold_answer=item.get("expected_state", {}),
                        metadata={
                            "apps": item.get("apps", []),
                            "num_apis": item.get("num_apis", 0),
                            "split": split,
                        },
                    )
                )
        logger.info("Loaded %d AppWorld tasks", len(self._tasks))

    def _load_appworld_tasks(self, split: str, max_samples: int | None) -> None:
        """Placeholder: integrate with the AppWorld engine."""
        raise NotImplementedError(
            "Implement with actual AppWorld data path. "
            "See https://github.com/stonybrookNLP/appworld for setup."
        )

    def evaluate_prediction(
        self, task: TaskInstance, prediction: Any
    ) -> dict[str, float]:
        if isinstance(prediction, dict) and "success" in prediction:
            return {
                "success_rate": float(prediction["success"]),
                "side_effect_rate": float(prediction.get("side_effects", 0) > 0),
            }
        return {"success_rate": 0.0}

    @staticmethod
    def _assign_pattern(item: dict) -> str:
        num_apps = len(item.get("apps", []))
        if num_apps >= 3:
            return "FP"
        if num_apps >= 2:
            return "PO"
        return "SL"

    @staticmethod
    def _stub_data() -> list[dict]:
        return [
            {
                "instruction": "Schedule a meeting with John for tomorrow at 2pm and send him an email confirmation.",
                "apps": ["calendar", "email"],
                "num_apis": 4,
                "expected_state": {"calendar_event_created": True, "email_sent": True},
            },
            {
                "instruction": "Find all overdue tasks and create a summary note.",
                "apps": ["todo", "notes"],
                "num_apis": 3,
                "expected_state": {"note_created": True},
            },
        ]
