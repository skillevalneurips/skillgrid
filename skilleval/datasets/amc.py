"""AMC dataset adapter: AMC 2023 competition math."""

from __future__ import annotations

import logging
import re
from typing import Any

from skilleval.core.registry import dataset_registry
from skilleval.core.types import TaskDomain, TaskInstance
from skilleval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


@dataset_registry.register("amc")
class AMCDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "amc"

    @property
    def domain(self) -> TaskDomain:
        return TaskDomain.MATH_REASONING

    def load(self) -> None:
        """Load AMC from HuggingFace."""
        test_ds_name = self.config.get("dataset.test_hf", "math-ai/amc23")
        max_test = self.config.get("dataset.max_test_samples")

        self._train_tasks = []
        self._test_tasks = self._load_hf(
            test_ds_name, max_samples=max_test, prefix="test",
        )
        self._tasks = list(self._test_tasks)
        logger.info("Loaded AMC: %d test", len(self._test_tasks))

    def _load_hf(
        self,
        dataset_name: str,
        max_samples: int | None,
        prefix: str,
    ) -> list[TaskInstance]:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="test")
        except Exception as exc:
            logger.warning(
                "HuggingFace datasets unavailable for %s: %s; using stub data.",
                dataset_name, exc,
            )
            ds = self._stub_data()

        out: list[TaskInstance] = []
        for idx, item in enumerate(ds):
            if max_samples and idx >= int(max_samples):
                break
            answer = self._extract_answer(item)
            out.append(
                TaskInstance(
                    task_id=f"amc_{prefix}_{idx}",
                    domain=self.domain,
                    instruction=item["question"],
                    composition_pattern=None,
                    tools_required=[],
                    gold_answer=answer,
                    metadata={
                        "source": dataset_name,
                        "original_id": item.get("id", str(idx)),
                    },
                )
            )
        return out

    def evaluate_prediction(
        self, task: TaskInstance, prediction: Any,
    ) -> dict[str, float]:
        gold = str(task.gold_answer).strip()
        pred = str(prediction).strip()

        if pred == gold:
            return {"success": 1.0, "exact_match": 1.0}

        tag = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", pred, re.IGNORECASE)
        if tag:
            candidate = tag.group(1)
            if candidate == gold:
                return {"success": 1.0, "exact_match": 1.0}

        nums = re.findall(r"-?\d+", pred)
        if nums and nums[-1] == gold:
            return {"success": 1.0, "exact_match": 1.0}

        return {"success": 0.0, "exact_match": 0.0}

    def get_answer_format_prompt(self) -> str | None:
        return (
            "Output format:\n"
            "The answer is an integer.\n"
            "When you have the final answer, end your reply with a single "
            "line in this exact format:\n"
            "ANSWER: <integer>\n"
            "Do not include leading zeros, units, or any other text on that line."
        )

    @staticmethod
    def _extract_answer(item: dict) -> str:
        raw = str(item.get("answer", "")).strip()
        if raw:
            return raw
        return ""

    @staticmethod
    def _stub_data() -> list[dict[str, str]]:
        return [
            {
                "question": "What is 2 + 2?",
                "answer": "4",
                "id": "0",
            }
        ]
