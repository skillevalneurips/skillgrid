"""Abstract base class for executable skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from skilleval.core.types import SkillOrigin, SkillSpec


class BaseSkill(ABC):
    """A reusable, composable unit of agent behaviour.

    Levels:
      1 - Atomic: wraps a single tool call with pre/post validation.
      2 - Macro: chains multiple atomic skills in a fixed workflow.
      3 - Programmatic: contains control flow (branching, loops, repair).
    """

    def __init__(self, spec: SkillSpec) -> None:
        self.spec = spec

    @property
    def skill_id(self) -> str:
        return self.spec.skill_id

    @property
    def level(self) -> int:
        return self.spec.level

    @property
    def origin(self) -> SkillOrigin:
        return self.spec.origin

    # -- required overrides --------------------------------------------------

    @abstractmethod
    def execute(self, context: dict[str, Any], **kwargs: Any) -> SkillResult:
        """Run the skill given the current agent context.

        Args:
            context: environment state, conversation history, etc.

        Returns:
            SkillResult with output, observations, and success flag.
        """

    @abstractmethod
    def to_prompt(self) -> str:
        """Serialize this skill into a prompt-injectable representation.

        Used by in-context and Anthropic-style protocols.
        """

    # -- optional overrides --------------------------------------------------

    def check_preconditions(self, context: dict[str, Any]) -> bool:
        """Return True if preconditions for this skill are met."""
        return True

    def check_postconditions(self, context: dict[str, Any], result: "SkillResult") -> bool:
        """Return True if postconditions are satisfied after execution."""
        return result.success

    def get_fallback(self) -> str | None:
        """Return fallback skill ID, if any."""
        return self.spec.fallback

    def similarity(self, other: BaseSkill) -> float:
        """Semantic similarity to another skill (for dedup / retrieval)."""
        shared_tools = set(self.spec.tool_calls) & set(other.spec.tool_calls)
        all_tools = set(self.spec.tool_calls) | set(other.spec.tool_calls)
        return len(shared_tools) / max(len(all_tools), 1)

    def __repr__(self) -> str:
        return f"<Skill {self.skill_id} L{self.level} ({self.origin.value})>"


@dataclass
class SkillResult:
    """Outcome of executing a single skill."""
    success: bool = False
    output: Any = None
    observations: list[str] = field(default_factory=list)
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    retries: int = 0
