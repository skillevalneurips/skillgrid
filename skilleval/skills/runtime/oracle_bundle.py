"""Runtime policy: Oracle Bundle (OB).

A pre-selected bundle of skills is provided to the agent.
Selection is not part of the evaluation -- this isolates
composition quality from retrieval quality.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from skilleval.core.registry import runtime_policy_registry
from skilleval.core.types import RuntimePolicy, TaskInstance
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)


@runtime_policy_registry.register("OB")
class OracleBundlePolicy:
    """Provide a curated (or random) skill bundle per task."""

    def __init__(
        self,
        bundle_size: int = 5,
        include_distractors: int = 0,
        seed: int = 42,
    ) -> None:
        self.bundle_size = bundle_size
        self.include_distractors = include_distractors
        self.rng = random.Random(seed)

    @property
    def policy_type(self) -> RuntimePolicy:
        return RuntimePolicy.ORACLE_BUNDLE

    def select_skills(
        self,
        task: TaskInstance,
        library: SkillLibrary,
    ) -> list[Any]:
        """Select the best-matching skills for this task.

        Strategy:
          1. Find skills whose tool_calls overlap with task tools.
          2. Pad with random distractors if configured.
        """
        all_skills = library.list_skills()
        required = set(task.tools_required)

        # Prefer skills whose declared tool_calls overlap with the task's
        # required tools. For tool-sparse datasets (math, pure reasoning)
        # SD skills may have empty tool_calls; in that case fall back to
        # taking the top-N skills so LB is not accidentally identical to NL.
        relevant = [
            s for s in all_skills
            if set(s.spec.tool_calls) & required
        ]
        if not relevant:
            relevant = list(all_skills)
        relevant = relevant[: self.bundle_size]

        if self.include_distractors > 0:
            distractors = [s for s in all_skills if s not in relevant]
            self.rng.shuffle(distractors)
            relevant.extend(distractors[: self.include_distractors])
            self.rng.shuffle(relevant)

        return relevant
