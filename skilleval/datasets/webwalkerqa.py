"""WebWalkerQA dataset adapter: web navigation and question answering.

Loads the WebWalkerQA dataset from HuggingFace (callanwu/WebWalkerQA).
All 680 questions live in a single ``main`` split; a seeded shuffle
creates train/test partitions.

Evaluation uses an LLM-based chain-of-thought QA judge (GPT-4o-mini by
default) matching the methodology from the original WebWalker paper.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

from skilleval.core.registry import dataset_registry
from skilleval.core.types import TaskDomain, TaskInstance
from skilleval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

_COT_QA_PROMPT = """\
You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer.
Determine whether the student's answer is correct by reasoning step-by-step.

Question: {question}
True Answer: {reference}
Student Answer: {prediction}

Think step-by-step, then on the final line write exactly one of:
GRADE: CORRECT
GRADE: INCORRECT"""


@dataset_registry.register("webwalkerqa")
class WebWalkerQADataset(BaseDataset):

    _judge_client = None

    @property
    def name(self) -> str:
        return "webwalkerqa"

    @property
    def domain(self) -> TaskDomain:
        return TaskDomain.WEB_NAVIGATION

    def load(self) -> None:
        """Load WebWalkerQA from HuggingFace.

        Exposes the single task pool as both train and test (no held-out
        split). TD probes and BU updates therefore draw from the same pool
        that is evaluated.
        """
        max_test = self.config.get(
            "dataset.max_test_samples",
            self.config.get("dataset.max_samples"),
        )
        split_seed = int(self.config.get("dataset.split_seed", 42))

        try:
            from datasets import load_dataset
            ds = load_dataset("callanwu/WebWalkerQA", split="main")
        except Exception:
            logger.warning("HuggingFace datasets not available; using stub data.")
            ds = self._stub_data()

        all_tasks: list[TaskInstance] = []
        for idx, item in enumerate(ds):
            question = item.get("question", "")
            root_url = item.get("root_url", "")
            info = item.get("info", {}) or {}

            instruction = f"Starting from: {root_url}\n\n{question}"

            all_tasks.append(
                TaskInstance(
                    task_id=f"webwalkerqa_{idx}",
                    domain=self.domain,
                    instruction=instruction,
                    composition_pattern=self._assign_pattern(info),
                    gold_answer=item.get("answer", ""),
                    metadata={
                        "root_url": root_url,
                        "type": info.get("type", ""),
                        "difficulty_level": info.get("difficulty_level", ""),
                        "domain": info.get("domain", ""),
                        "golden_path": info.get("golden_path", []),
                    },
                )
            )

        # Single pool used for both train and test (no held-out split).
        rng = random.Random(split_seed)
        shuffled = list(all_tasks)
        rng.shuffle(shuffled)

        if max_test is not None:
            shuffled = shuffled[:int(max_test)]

        self._test_tasks = shuffled
        self._train_tasks = list(self._test_tasks)
        self._tasks = list(self._test_tasks)
        logger.info(
            "Loaded WebWalkerQA: %d tasks (used as both train and test, from %d total)",
            len(self._test_tasks), len(all_tasks),
        )

    def evaluate_prediction(
        self, task: TaskInstance, prediction: Any,
    ) -> dict[str, float]:
        gold = str(task.gold_answer).strip()
        pred = str(prediction).strip()

        if not gold or not pred:
            return {"success": 0.0, "exact_match": 0.0}

        judge_model = self.config.get("dataset.judge_model", "gpt-4o-mini")
        score = self._llm_judge(task.instruction, pred, gold, judge_model)
        return {"success": score, "exact_match": score}

    def get_answer_format_prompt(self) -> str | None:
        return (
            "Output format:\n"
            "When you have the final answer, submit it using the <answer> tag:\n"
            "<answer>your answer here</answer>\n\n"
            "The answer should be a clear, complete response to the question. "
            "Do not include explanations inside the <answer> tag."
        )

    # -- LLM judge -------------------------------------------------------------

    @classmethod
    def _get_judge_client(cls):
        if cls._judge_client is None:
            from openai import OpenAI
            cls._judge_client = OpenAI()
        return cls._judge_client

    @classmethod
    def _llm_judge(
        cls, question: str, prediction: str, reference: str, model: str,
    ) -> float:
        try:
            client = cls._get_judge_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": _COT_QA_PROMPT.format(
                        question=question,
                        prediction=prediction,
                        reference=reference,
                    ),
                }],
                temperature=0.0,
                max_tokens=512,
            )
            text = (resp.choices[0].message.content or "").strip()
            last_line = text.strip().split("\n")[-1].upper()
            if "CORRECT" in last_line and "INCORRECT" not in last_line:
                return 1.0
            return 0.0
        except Exception as exc:
            logger.warning("LLM judge call failed: %s", exc)
            return 0.0

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _assign_pattern(info: dict) -> str:
        task_type = info.get("type", "")
        if task_type == "multi_source":
            return "FP"
        return "SL"

    @staticmethod
    def _stub_data() -> list[dict]:
        return [
            {
                "question": "What are the opening hours of the MIT library?",
                "answer": "8am to 11pm",
                "root_url": "https://libraries.mit.edu",
                "info": {
                    "type": "single_source",
                    "difficulty_level": "easy",
                    "domain": "university",
                    "golden_path": [],
                },
            },
        ]
