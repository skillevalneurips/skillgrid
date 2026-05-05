"""AIME dataset adapter: competition-level math (AIME 2024 + 2025)."""

from __future__ import annotations

import logging
import re
from typing import Any

from skilleval.core.config import Config
from skilleval.core.registry import dataset_registry
from skilleval.core.types import TaskDomain, TaskInstance
from skilleval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


@dataset_registry.register("aime")
class AIMEDataset(BaseDataset):

    @property
    def name(self) -> str:
        return "aime"

    @property
    def domain(self) -> TaskDomain:
        return TaskDomain.MATH_REASONING

    def load(self) -> None:
        """Load AIME from HuggingFace: aime24 as train, aime25 as test.

        AIME answers are integers in [0, 999].
        """
        train_ds_name = self.config.get("dataset.train_hf", "math-ai/aime24")
        test_ds_name = self.config.get("dataset.test_hf", "math-ai/aime25")
        max_train = self.config.get("dataset.max_train_samples")
        max_test = self.config.get("dataset.max_test_samples")

        self._train_tasks = self._load_hf(
            train_ds_name, max_samples=max_train, prefix="train",
        )
        self._test_tasks = self._load_hf(
            test_ds_name, max_samples=max_test, prefix="test",
        )
        self._tasks = list(self._test_tasks)
        logger.info(
            "Loaded AIME: %d train, %d test",
            len(self._train_tasks), len(self._test_tasks),
        )

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
                    task_id=f"aime_{prefix}_{idx}",
                    domain=self.domain,
                    instruction=item["problem"],
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

        # Direct match
        if pred == gold:
            return {"success": 1.0, "exact_match": 1.0}

        # ANSWER: tag (authoritative when present)
        tag = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", pred, re.IGNORECASE)
        if tag:
            candidate = tag.group(1)
            # Strip trailing .0 for integer comparison
            if candidate.endswith(".0"):
                candidate = candidate[:-2]
            if candidate == gold:
                return {"success": 1.0, "exact_match": 1.0}

        # Fallback: last integer in the response
        numbers = re.findall(r"\b\d+\b", pred)
        if numbers and numbers[-1] == gold:
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
        """Extract the integer answer from a HF AIME row.

        Tries ``answer`` field first, then ``solution`` (which stores the
        answer inside ``\\boxed{…}`` on math-ai/aime24 and aime25).
        """
        raw = str(item.get("answer", "")).strip()
        if raw:
            return raw
        sol = str(item.get("solution", "")).strip()
        if sol:
            m = re.search(r"\\boxed\{([^{}]+)\}", sol)
            if m:
                return m.group(1).strip()
            nums = re.findall(r"-?\d+", sol)
            if nums:
                return nums[-1]
            return sol
        return ""

    @staticmethod
    def _stub_data() -> list[dict[str, str]]:
        return [
            {
                "problem": (
                    "Find the sum of all integer bases b > 9 for which "
                    "17_b is a divisor of 97_b."
                ),
                "answer": "70",
                "id": "0",
            },
        ]
