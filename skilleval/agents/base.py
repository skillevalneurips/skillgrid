"""Abstract base class for benchmark agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from skilleval.core.types import EpisodeTrace, TaskInstance
from skilleval.models.base import BaseModel
from skilleval.skills.library import SkillLibrary


class BaseAgent(ABC):
    """Interface for agents that solve benchmark tasks.

    An agent combines a model, a skill library, and a protocol/policy
    to produce an episode trace for each task.
    """

    def __init__(self, model: BaseModel, library: SkillLibrary) -> None:
        self.model = model
        self.library = library

    @abstractmethod
    def solve(
        self,
        task: TaskInstance,
        max_steps: int = 50,
        **kwargs: Any,
    ) -> EpisodeTrace:
        """Attempt to solve one task and return the full episode trace."""

    def solve_batch(
        self,
        tasks: list[TaskInstance],
        max_steps: int = 50,
        **kwargs: Any,
    ) -> list[EpisodeTrace]:
        """Solve a batch of tasks sequentially (override for parallel)."""
        return [self.solve(task, max_steps=max_steps, **kwargs) for task in tasks]
