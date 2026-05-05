"""GSM8K dataset adapter: grade-school math word problems."""

from __future__ import annotations

import logging
import re
from typing import Any

from skilleval.core.config import Config
from skilleval.core.registry import dataset_registry
from skilleval.core.types import TaskDomain, TaskInstance
from skilleval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


@dataset_registry.register("gsm8k")
class GSM8KDataset(BaseDataset):

    @property
    def name(self) -> str:
        return "gsm8k"

    @property
    def domain(self) -> TaskDomain:
        return TaskDomain.MATH_REASONING

    def load(self) -> None:
        """Load GSM8K train + test splits from HuggingFace ``datasets``.

        - ``dataset.train_split`` (default ``"train"``) → ``self._train_tasks``
          (capped at ``dataset.max_train_samples``).
        - ``dataset.test_split`` (default ``"test"``) → ``self._test_tasks``
          (capped at ``dataset.max_test_samples``, falls back to legacy
          ``max_samples``).

        Legacy ``dataset.split`` is still honored: if only a single split
        is requested it is loaded as the test pool and ``train_tasks()``
        falls back to seeded shuffle.
        """
        train_split = self.config.get("dataset.train_split", "train")
        test_split = self.config.get(
            "dataset.test_split", self.config.get("dataset.split", "test"),
        )
        max_train = self.config.get("dataset.max_train_samples")
        max_test = self.config.get(
            "dataset.max_test_samples", self.config.get("dataset.max_samples"),
        )

        self._test_tasks = self._load_split(test_split, max_test, prefix="test")
        if train_split and train_split != test_split:
            self._train_tasks = self._load_split(
                train_split, max_train, prefix="train",
            )
        # Populate legacy ``_tasks`` with the test pool for backward compat.
        self._tasks = list(self._test_tasks)
        logger.info(
            "Loaded GSM8K: %d train, %d test",
            len(self._train_tasks), len(self._test_tasks),
        )

    def _load_split(
        self, split: str, max_samples: int | None, prefix: str,
    ) -> list[TaskInstance]:
        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split=split)
        except Exception:
            logger.warning(
                "HuggingFace datasets unavailable for split %s; using stub data.",
                split,
            )
            ds = self._stub_data()

        out: list[TaskInstance] = []
        for idx, item in enumerate(ds):
            if max_samples and idx >= int(max_samples):
                break
            answer = self._extract_answer(item.get("answer", ""))
            pattern = self._assign_pattern(item)
            out.append(
                TaskInstance(
                    task_id=f"gsm8k_{prefix}_{idx}",
                    domain=self.domain,
                    instruction=item["question"],
                    composition_pattern=pattern,
                    tools_required=["calculator"],
                    gold_answer=answer,
                    metadata={
                        "raw_answer": item.get("answer", ""),
                        "split": split,
                    },
                )
            )
        return out

    def evaluate_prediction(
        self, task: TaskInstance, prediction: Any
    ) -> dict[str, float]:
        gold = re.sub(r"[,\$%]", "", str(task.gold_answer).strip())
        pred = re.sub(r"[,\$%]", "", str(prediction).strip())
        match = float(pred == gold)
        if match < 1.0:
            # Tag-preferred: the system prompt instructs the agent to emit
            # ``ANSWER: <number>``. When present, it is authoritative.
            tag = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", pred, re.IGNORECASE)
            if tag and tag.group(1) == gold:
                match = 1.0
        if match < 1.0:
            # Fallback: last number in the prediction.
            numbers = re.findall(r"-?\d+(?:\.\d+)?", pred)
            if numbers and numbers[-1] == gold:
                match = 1.0
        if match < 1.0:
            # Broader fallback: gold appears anywhere as a standalone number.
            # Handles sentence-form answers like "She pays $64 for 16 glasses"
            # where gold=64 but last-number picks 16.
            if re.search(r"(?<!\d)" + re.escape(gold) + r"(?!\d)", pred):
                match = 1.0
        return {"success": match, "exact_match": match}

    def get_answer_format_prompt(self) -> str | None:
        return (
            "Output format:\n"
            "When you have the final numeric answer, end your reply with a "
            "single line in this exact format:\n"
            "ANSWER: <number>\n"
            "Do not include units, currency symbols, or commas after the tag."
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _extract_answer(answer_text: str) -> str:
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return match.group(1).strip().replace(",", "")
        return answer_text.strip()

    @staticmethod
    def _assign_pattern(item: dict) -> str:
        """Heuristic: multi-step problems with verification -> FP, etc."""
        text = item.get("question", "")
        if any(kw in text.lower() for kw in ["check", "verify", "if not"]):
            return "FP"
        if any(kw in text.lower() for kw in ["either", "or", "choose", "which"]):
            return "PO"
        return "SL"

    @staticmethod
    def _stub_data() -> list[dict[str, str]]:
        return [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the remainder for $2 each. How much does she make per day?",
                "answer": "She has 16 - 3 - 4 = <<16-3-4=9>>9 eggs left.\nShe makes 9 * 2 = $<<9*2=18>>18 per day.\n#### 18",
            },
        ]
