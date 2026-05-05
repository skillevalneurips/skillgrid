"""MATH dataset adapter: competition-level mathematics problems."""

from __future__ import annotations

import logging
import re
from typing import Any

from skilleval.core.config import Config
from skilleval.core.registry import dataset_registry
from skilleval.core.types import TaskDomain, TaskInstance
from skilleval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


@dataset_registry.register("math")
class MATHDataset(BaseDataset):

    @property
    def name(self) -> str:
        return "math"

    @property
    def domain(self) -> TaskDomain:
        return TaskDomain.MATH_REASONING

    def load(self) -> None:
        split = self.config.get("dataset.split", "test")
        max_samples = self.config.get("dataset.max_samples")
        categories = self.config.get("dataset.categories")

        try:
            from datasets import load_dataset
            ds = load_dataset("hendrycks/competition_math", split=split)
        except Exception:
            logger.warning("HuggingFace datasets not available; using stub data.")
            ds = self._stub_data()

        for idx, item in enumerate(ds):
            if max_samples and idx >= max_samples:
                break
            cat = item.get("type", "unknown")
            if categories and cat not in categories:
                continue
            answer = self._extract_boxed(item.get("solution", ""))
            self._tasks.append(
                TaskInstance(
                    task_id=f"math_{idx}",
                    domain=self.domain,
                    instruction=item["problem"],
                    composition_pattern=self._assign_pattern(item),
                    tools_required=["calculator", "symbolic_engine"],
                    gold_answer=answer,
                    metadata={
                        "category": cat,
                        "level": item.get("level", ""),
                        "raw_solution": item.get("solution", ""),
                    },
                )
            )
        logger.info("Loaded %d MATH tasks", len(self._tasks))

    def evaluate_prediction(
        self, task: TaskInstance, prediction: Any
    ) -> dict[str, float]:
        gold = self._normalize(str(task.gold_answer))
        pred = self._normalize(str(prediction))
        return {"exact_match": float(gold == pred)}

    @staticmethod
    def _extract_boxed(solution: str) -> str:
        match = re.search(r"\\boxed\{(.+?)\}", solution)
        return match.group(1).strip() if match else solution.strip()

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("\\left", "").replace("\\right", "")
        return text

    @staticmethod
    def _assign_pattern(item: dict) -> str:
        level = str(item.get("level", ""))
        if "5" in level:
            return "FP"
        if "3" in level or "4" in level:
            return "PO"
        return "SL"

    @staticmethod
    def _stub_data() -> list[dict]:
        return [
            {
                "problem": "Find the value of $x$ if $3x + 7 = 22$.",
                "solution": "$3x = 15$, so $x = \\boxed{5}$.",
                "type": "algebra",
                "level": "Level 1",
            },
        ]
