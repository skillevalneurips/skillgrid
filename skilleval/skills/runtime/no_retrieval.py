"""Runtime policy: No Retrieval (NR).

Pass-through policy. The agent uses whatever skills Axis 1 (Visibility)
gave it — no re-selection, no retrieval during execution.
"""

from __future__ import annotations

from typing import Any

from skilleval.core.registry import runtime_policy_registry
from skilleval.core.types import RuntimePolicy, TaskInstance
from skilleval.skills.library import SkillLibrary


@runtime_policy_registry.register("NR")
class NoRetrievalPolicy:
    """Pass-through: no skill re-selection at runtime.

    Whatever Axis 1 (Visibility) decided is what the agent gets.
    """

    @property
    def policy_type(self) -> RuntimePolicy:
        return RuntimePolicy.ORACLE_BUNDLE  # closest existing enum value

    def select_skills(
        self,
        task: TaskInstance,
        library: SkillLibrary,
    ) -> list[Any]:
        """Return all skills in the library — no filtering."""
        return library.list_skills()
