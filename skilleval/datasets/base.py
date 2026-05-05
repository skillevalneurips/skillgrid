"""Abstract base class for all benchmark datasets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator

from skilleval.core.types import (
    TaskDomain,
    TaskInstance,
)
from skilleval.core.config import Config

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Interface that every dataset adapter must implement.

    To add a new dataset:
    1. Subclass ``BaseDataset``.
    2. Implement all abstract methods.
    3. Register with ``@dataset_registry.register("your_name")``.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._tasks: list[TaskInstance] = []
        self._train_tasks: list[TaskInstance] = []
        self._test_tasks: list[TaskInstance] = []
        self._loaded = False

    # -- required overrides --------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``'gsm8k'``."""

    @property
    @abstractmethod
    def domain(self) -> TaskDomain:
        """Which task domain this dataset belongs to."""

    @abstractmethod
    def load(self) -> None:
        """Download / read raw data and populate ``self._tasks``."""

    @abstractmethod
    def evaluate_prediction(
        self, task: TaskInstance, prediction: Any
    ) -> dict[str, float]:
        """Compare a single prediction against gold and return metric dict.

        Contract:
            Must return a dict containing a ``"success"`` key with a float
            value in [0.0, 1.0]. The evaluator treats ``success >= 0.5`` as
            a solved task. Any additional metric keys (``exact_match``,
            ``f1``, ``llm_judge``, ``pass``, ``task_reward``, ``recall@10``)
            are optional and stored as metadata.
        """

    # -- optional overrides --------------------------------------------------

    def get_tools(self) -> list[dict[str, str]]:
        """Return tool definitions available for this dataset."""
        return self.config.get("dataset.tools", [])

    def get_tool_executors(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        """Return real tool implementations for this dataset.

        Maps tool name -> callable that takes a parsed-args dict and returns
        the observation. Tools not present in this dict fall back to a stub
        string in the agent protocol.

        Default: empty dict (all tools stubbed). Override to wire real
        executors for stateful/environment tools (browser, sandbox, SQL,
        calculator, etc.).
        """
        return {}

    def get_answer_format_prompt(self) -> str | None:
        """Return a scoring-contract instruction appended to the agent's
        system prompt, or ``None`` if the dataset scores on tool state
        rather than a string match.

        Co-located with ``evaluate_prediction`` in subclasses so the
        producer (what the agent is told to emit) and consumer (the regex
        the scorer expects) cannot drift.
        """
        return None

    # -- shared logic --------------------------------------------------------

    def tasks(self) -> list[TaskInstance]:
        """All loaded tasks. Alias for test_tasks() when splits are populated."""
        if not self._loaded:
            self.load()
            self._loaded = True
        if self._test_tasks:
            return self._test_tasks
        return self._tasks

    def train_tasks(self) -> list[TaskInstance]:
        """Tasks reserved for library construction (TD probe, BU updates).

        Subclasses that load both splits should populate ``self._train_tasks``
        directly. Datasets without a native split fall back to a seeded
        shuffle of ``self._tasks`` — train/test are disjoint slices.
        """
        if not self._loaded:
            self.load()
            self._loaded = True
        if self._train_tasks:
            return self._train_tasks
        self._derive_splits_from_single_pool()
        return self._train_tasks

    def test_tasks(self) -> list[TaskInstance]:
        """Tasks used for every method's final evaluation (held out)."""
        if not self._loaded:
            self.load()
            self._loaded = True
        if self._test_tasks:
            return self._test_tasks
        self._derive_splits_from_single_pool()
        return self._test_tasks

    def _derive_splits_from_single_pool(self) -> None:
        """Fallback for datasets with no native train split.

        Shuffles ``self._tasks`` with ``split_seed`` and slices off the last
        ``max_test_samples`` tasks as test; the rest is train (capped at
        ``max_train_samples`` if set).
        """
        if self._train_tasks or self._test_tasks:
            return
        import random
        pool = list(self._tasks)
        if not pool:
            return
        seed = int(self.config.get("dataset.split_seed", 42))
        max_train = self.config.get("dataset.max_train_samples")
        max_test = self.config.get("dataset.max_test_samples") or \
            self.config.get("dataset.max_samples")
        rng = random.Random(seed)
        rng.shuffle(pool)
        n_test = int(max_test) if max_test else min(len(pool) // 5, len(pool))
        self._test_tasks = pool[:n_test]
        remainder = pool[n_test:]
        if max_train:
            remainder = remainder[: int(max_train)]
        self._train_tasks = remainder

    def __iter__(self) -> Iterator[TaskInstance]:
        return iter(self.tasks())

    def __len__(self) -> int:
        return len(self.tasks())

    def sample(self, n: int, seed: int = 42) -> list[TaskInstance]:
        """Return a random subsample of *n* tasks."""
        import random
        rng = random.Random(seed)
        pool = self.tasks()
        return rng.sample(pool, min(n, len(pool)))

    def summary(self) -> dict[str, Any]:
        tasks = self.tasks()
        pattern_counts = {}
        for t in tasks:
            key = t.composition_pattern
            if key is not None:
                pattern_counts[key] = pattern_counts.get(key, 0) + 1
        return {
            "name": self.name,
            "domain": self.domain.value,
            "total_tasks": len(tasks),
            "pattern_distribution": pattern_counts,
            "tools": [t["name"] for t in self.get_tools()],
        }
