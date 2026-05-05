"""Per-dataset skill-creation recipe.

A recipe customizes how TopDown (SD) and BottomUp (TD) creators behave for
a specific dataset. Defaults in this file reproduce today's behavior — so
datasets without a recipe keep working unchanged.

Datasets ship their own recipe at ``datasets/<name>/skills/recipe.py``,
exposing a ``build_recipe() -> SkillRecipe`` function. The evaluator calls
the loader (see ``loader.py``) to resolve the recipe once per run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from skilleval.core.types import EpisodeTrace, SkillSpec


@dataclass
class SkillRecipe:
    """Per-dataset overrides for skill generation.

    All fields optional; an empty ``SkillRecipe()`` is the "use defaults"
    case. Fields are consumed by ``TopDownCreator``, ``BottomUpCreator``,
    and ``llm_skill_writer`` at construction time.
    """

    sd_prompt: str | None = None
    """Override for the spec-derived prompt. ``None`` → built-in SD_PROMPT."""

    td_prompt: str | None = None
    """Override for the trace-derived prompt. ``None`` → built-in TD_PROMPT."""

    seed_skills: list[SkillSpec] = field(default_factory=list)
    """Hand-authored skills merged into the library after LLM generation."""

    sample_task_count: int = 5
    """How many task instructions to show the SD writer model."""

    max_skills: int | None = None
    """Overrides ``experiment.max_skills`` if set."""

    trace_summarizer: Callable[[list[EpisodeTrace]], str] | None = None
    """Custom function to condense probe traces for the TD prompt."""

    extra_context: dict[str, Any] = field(default_factory=dict)
    """Arbitrary kwargs forwarded to prompt formatting (e.g. answer-format hint)."""
