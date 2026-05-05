"""GAIA dataset adapter: General AI Assistants benchmark.

Loads the GAIA 2023 dataset from HuggingFace (validation split as test,
since the test set has no public labels). Supports file attachments and
robust fuzzy scoring ported from GAIABenchmark.question_scorer.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Any

from skilleval.core.registry import dataset_registry
from skilleval.core.types import TaskDomain, TaskInstance
from skilleval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


@dataset_registry.register("gaia")
class GAIADataset(BaseDataset):

    @property
    def name(self) -> str:
        return "gaia"

    @property
    def domain(self) -> TaskDomain:
        return TaskDomain.GENERAL_REASONING

    def load(self) -> None:
        """Load GAIA from HuggingFace.

        Uses validation split as test (public labels available) and
        optionally a portion of it as train for TD probes / BU updates.
        """
        levels = self.config.get("dataset.levels", [1, 2, 3])
        max_test = self.config.get(
            "dataset.max_test_samples",
            self.config.get("dataset.max_samples"),
        )
        max_train = self.config.get("dataset.max_train_samples")
        split_seed = int(self.config.get("dataset.split_seed", 42))
        train_fraction = float(self.config.get("dataset.train_fraction", 0.2))

        try:
            from datasets import load_dataset
            ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
        except Exception:
            logger.warning("HuggingFace datasets not available; using stub data.")
            ds = self._stub_data()

        all_tasks: list[TaskInstance] = []
        for idx, item in enumerate(ds):
            level = item.get("Level", item.get("level", 1))
            if isinstance(level, str):
                try:
                    level = int(level)
                except ValueError:
                    level = 1
            if level not in levels:
                continue

            task_id = item.get("task_id", f"gaia_{idx}")
            question = item.get("Question", item.get("question", ""))
            gold = item.get("Final answer", item.get("answer", ""))
            file_name = item.get("file_name", "") or ""

            metadata: dict[str, Any] = {"level": level}
            if file_name:
                metadata["file_name"] = file_name
                question = f"{question}\n\n[Attached file: {file_name}]"

            all_tasks.append(
                TaskInstance(
                    task_id=str(task_id),
                    domain=self.domain,
                    instruction=question,
                    composition_pattern=self._level_to_pattern(level),
                    tools_required=[
                        "web_search", "web_browse", "file_reader",
                        "code_executor", "calculator",
                    ],
                    gold_answer=gold,
                    metadata=metadata,
                )
            )

        # Split into train/test using seeded shuffle
        import random
        rng = random.Random(split_seed)
        shuffled = list(all_tasks)
        rng.shuffle(shuffled)

        n_train = int(len(shuffled) * train_fraction)
        if max_train is not None:
            n_train = min(n_train, int(max_train))

        self._train_tasks = shuffled[:n_train]
        self._test_tasks = shuffled[n_train:]

        if max_test is not None:
            self._test_tasks = self._test_tasks[:int(max_test)]

        self._tasks = list(self._test_tasks)
        logger.info(
            "Loaded GAIA: %d train, %d test (from %d total)",
            len(self._train_tasks), len(self._test_tasks), len(all_tasks),
        )

    def evaluate_prediction(
        self, task: TaskInstance, prediction: Any,
    ) -> dict[str, float]:
        """Robust scoring ported from GAIABenchmark.question_scorer.

        Handles numeric comparison, comma-separated lists, and
        case-insensitive string matching with whitespace normalization.
        """
        gold = str(task.gold_answer).strip()
        pred = str(prediction).strip()

        if not gold or not pred:
            return {"success": 0.0, "exact_match": 0.0}

        if self._is_float(gold):
            norm_pred = self._normalize_number_str(pred)
            norm_gold = float(gold)
            match = float(norm_pred == norm_gold)
            return {"success": match, "exact_match": match}

        if any(c in gold for c in [",", ";"]):
            gt_elems = self._split_string(gold)
            ma_elems = self._split_string(pred)
            if len(gt_elems) != len(ma_elems):
                return {"success": 0.0, "exact_match": 0.0}
            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if self._is_float(gt_elem):
                    comparisons.append(
                        self._normalize_number_str(ma_elem) == float(gt_elem)
                    )
                else:
                    comparisons.append(
                        self._normalize_str(ma_elem, remove_punct=False)
                        == self._normalize_str(gt_elem, remove_punct=False)
                    )
            match = float(all(comparisons))
            return {"success": match, "exact_match": match}

        match = float(self._normalize_str(pred) == self._normalize_str(gold))
        return {"success": match, "exact_match": match}

    def get_answer_format_prompt(self) -> str | None:
        return (
            "Output format:\n"
            "When you have the final answer, submit it using the <answer> tag:\n"
            "<answer>your answer here</answer>\n\n"
            "The answer should be concise — just the value, name, or number "
            "requested. Do not include explanations inside the <answer> tag."
        )

    # -- scoring helpers (ported from GAIABenchmark) -------------------------

    @staticmethod
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _normalize_number_str(number_str: str) -> float:
        cleaned = number_str
        for char in ["$", "%", ","]:
            cleaned = cleaned.replace(char, "")
        cleaned = cleaned.strip()
        try:
            return float(cleaned)
        except ValueError:
            m = re.search(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                cleaned,
            )
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    pass
            return float("inf")

    @staticmethod
    def _split_string(s: str, char_list: list[str] | None = None) -> list[str]:
        if char_list is None:
            char_list = [",", ";"]
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    @staticmethod
    def _normalize_str(input_str: str, remove_punct: bool = True) -> str:
        no_spaces = re.sub(r"\s", "", input_str)
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        return no_spaces.lower()

    @staticmethod
    def _level_to_pattern(level: int) -> str:
        mapping = {1: "SL", 2: "PO", 3: "FP"}
        return mapping.get(level, "SL")

    @staticmethod
    def _stub_data() -> list[dict]:
        return [
            {
                "Question": "What is the population of France as of 2023?",
                "Final answer": "68 million",
                "Level": 1,
                "task_id": "stub_0",
            },
            {
                "Question": "What is 2 + 2?",
                "Final answer": "4",
                "Level": 1,
                "task_id": "stub_1",
            },
        ]
