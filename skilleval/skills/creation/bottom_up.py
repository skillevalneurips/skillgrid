"""Trace-Derived (TD) skill creation via LLM.

The LLM is given a summary of probe traces and returns exactly
``max_skills`` procedural skills that generalize the observed patterns.
No sub-trace clustering, no frequency thresholds, no separate recovery
pipeline — the LLM handles all of that implicitly.
"""

from __future__ import annotations

import logging

from skilleval.core.registry import skill_creator_registry
from skilleval.core.types import EpisodeTrace
from skilleval.models.base import BaseModel
from skilleval.skills.creation.llm_skill_writer import generate_library_from_traces
from skilleval.skills.creation.recipe import SkillRecipe
from skilleval.skills.creation.top_down import PromptSkill
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)


@skill_creator_registry.register("bottom_up")
class BottomUpCreator:
    """LLM-driven TD skill creation."""

    def __init__(
        self,
        writer_model: BaseModel,
        max_skills: int = 3,
        recipe: SkillRecipe | None = None,
    ) -> None:
        self.writer_model = writer_model
        self.recipe = recipe or SkillRecipe()
        self.max_skills = self.recipe.max_skills if self.recipe.max_skills is not None else max_skills

    def create(
        self,
        traces: list[EpisodeTrace],
        library: SkillLibrary | None = None,
    ) -> SkillLibrary:
        lib = library or SkillLibrary()
        if self.max_skills > 0:
            specs = generate_library_from_traces(
                writer_model=self.writer_model,
                traces=traces,
                max_skills=self.max_skills,
                prompt_template=self.recipe.td_prompt,
                trace_summarizer=self.recipe.trace_summarizer,
                extra_context=self.recipe.extra_context,
            )
            for spec in specs:
                lib.add(PromptSkill(spec))
        else:
            specs = []
        for seed_spec in self.recipe.seed_skills:
            lib.add(PromptSkill(seed_spec))
        logger.info(
            "Bottom-up creation complete: %d skills (%d LLM + %d seed)",
            lib.size, len(specs), len(self.recipe.seed_skills),
        )
        return lib
