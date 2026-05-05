"""ALFWorld dataset adapter: text-based embodied household tasks."""

from __future__ import annotations

import logging
from typing import Any

from skilleval.core.registry import dataset_registry
from skilleval.core.types import TaskDomain, TaskInstance
from skilleval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

TASK_TYPE_PATTERN = {
    "pick_and_place": "SL",
    "pick_clean_then_place": "SL",
    "pick_heat_then_place": "PO",
    "pick_cool_then_place": "PO",
    "look_at_obj_in_light": "SL",
    "pick_two_obj_and_place": "FP",
}


@dataset_registry.register("alfworld")
class ALFWorldDataset(BaseDataset):

    @property
    def name(self) -> str:
        return "alfworld"

    @property
    def domain(self) -> TaskDomain:
        return TaskDomain.EXECUTION_ENV

    def load(self) -> None:
        max_samples = self.config.get("dataset.max_samples")
        task_types = self.config.get("dataset.task_types")

        try:
            self._load_alfworld_tasks(max_samples, task_types)
        except Exception:
            logger.warning("ALFWorld environment not available; using stub data.")
            for idx, item in enumerate(self._stub_data()):
                if max_samples and idx >= max_samples:
                    break
                self._tasks.append(
                    TaskInstance(
                        task_id=f"alfworld_{idx}",
                        domain=self.domain,
                        instruction=item["instruction"],
                        composition_pattern=TASK_TYPE_PATTERN.get(
                            item["task_type"], "SL"
                        ),
                        tools_required=["goto", "take", "put", "open", "close", "toggle", "heat", "cool", "clean", "examine", "inventory", "look"],
                        gold_answer=item.get("gold_actions", []),
                        metadata={"task_type": item["task_type"], "split": "unseen"},
                    )
                )
        logger.info("Loaded %d ALFWorld tasks", len(self._tasks))

    def _load_alfworld_tasks(
        self, max_samples: int | None, task_types: list[str] | None
    ) -> None:
        """Placeholder: load from ALFWorld's JSON task files."""
        raise NotImplementedError(
            "Implement with actual ALFWorld data path. "
            "See https://github.com/alfworld/alfworld for setup."
        )

    def evaluate_prediction(
        self, task: TaskInstance, prediction: Any
    ) -> dict[str, float]:
        if isinstance(prediction, dict) and "success" in prediction:
            return {"success_rate": float(prediction["success"])}
        return {"success_rate": 0.0}

    @staticmethod
    def _stub_data() -> list[dict]:
        return [
            {
                "instruction": "Put a clean mug in the cabinet.",
                "task_type": "pick_clean_then_place",
                "gold_actions": [
                    "goto countertop",
                    "take mug",
                    "goto sinkbasin",
                    "clean mug",
                    "goto cabinet",
                    "put mug",
                ],
            },
            {
                "instruction": "Put a hot potato in the fridge.",
                "task_type": "pick_heat_then_place",
                "gold_actions": [
                    "goto countertop",
                    "take potato",
                    "goto microwave",
                    "heat potato",
                    "goto fridge",
                    "put potato",
                ],
            },
        ]
