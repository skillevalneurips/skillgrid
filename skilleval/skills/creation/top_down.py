"""Spec-Derived (SD) skill creation via LLM.

The LLM is given the dataset's domain, tool specs, and a handful of
sample task instructions, and returns exactly ``max_skills`` procedural
skills. No heuristic candidate generation, no scoring, no thresholds.
"""

from __future__ import annotations

import logging
from typing import Any

from skilleval.core.registry import skill_creator_registry
from skilleval.core.types import SkillOrigin, SkillSpec
from skilleval.datasets.base import BaseDataset
from skilleval.models.base import BaseModel
from skilleval.skills.creation.llm_skill_writer import generate_library_from_spec
from skilleval.skills.creation.recipe import SkillRecipe
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)


@skill_creator_registry.register("top_down")
class TopDownCreator:
    """LLM-driven SD skill creation."""

    def __init__(
        self,
        writer_model: BaseModel,
        max_skills: int = 3,
        recipe: SkillRecipe | None = None,
    ) -> None:
        self.writer_model = writer_model
        self.recipe = recipe or SkillRecipe()
        # Recipe may override max_skills.
        self.max_skills = self.recipe.max_skills if self.recipe.max_skills is not None else max_skills

    def create(
        self,
        dataset: BaseDataset,
        library: SkillLibrary | None = None,
    ) -> SkillLibrary:
        # Prefer train_tasks for samples so eval stays leakage-free, falling
        # back to tasks() when a dataset only has one pool.
        sample_pool = dataset.train_tasks() if hasattr(dataset, "train_tasks") else dataset.tasks()
        if not sample_pool:
            sample_pool = dataset.tasks()
        sample_tasks = [t.instruction for t in sample_pool[: self.recipe.sample_task_count]]

        lib = library or SkillLibrary()
        if self.max_skills > 0:
            specs = generate_library_from_spec(
                writer_model=self.writer_model,
                domain=dataset.domain.value,
                tools=dataset.get_tools(),
                sample_tasks=sample_tasks,
                max_skills=self.max_skills,
                prompt_template=self.recipe.sd_prompt,
                extra_context=self.recipe.extra_context,
            )
            for spec in specs:
                lib.add(PromptSkill(spec))
        else:
            specs = []
        for seed_spec in self.recipe.seed_skills:
            lib.add(PromptSkill(seed_spec))
        logger.info(
            "Top-down creation complete: %d skills (%d LLM + %d seed)",
            lib.size, len(specs), len(self.recipe.seed_skills),
        )
        return lib


class PromptSkill:
    """Minimal concrete skill backed by a prompt template.

    Used as the default skill type for SD/TD creation. Subclass
    ``BaseSkill`` for richer implementations with runtime side effects.
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

    def execute(self, context: dict[str, Any], **kwargs: Any) -> Any:
        from skilleval.skills.base import SkillResult
        return SkillResult(
            success=True,
            output=self.spec.template,
            observations=[f"Prompt-based execution of {self.spec.name}"],
        )

    def to_prompt(self) -> str:
        lines = [
            f"### Skill: {self.spec.name}",
            f"**Description:** {self.spec.description}",
            f"**Tools:** {', '.join(self.spec.tool_calls)}",
        ]
        if self.spec.preconditions:
            lines.append(f"**Preconditions:** {'; '.join(self.spec.preconditions)}")
        if self.spec.postconditions:
            lines.append(f"**Postconditions:** {'; '.join(self.spec.postconditions)}")
        lines.append(f"**Template:** {self.spec.template}")
        return "\n".join(lines)

    def similarity(self, other: Any) -> float:
        if not hasattr(other, "spec"):
            return 0.0
        shared = set(self.spec.tool_calls) & set(other.spec.tool_calls)
        total = set(self.spec.tool_calls) | set(other.spec.tool_calls)
        return len(shared) / max(len(total), 1)

    def __repr__(self) -> str:
        return f"<PromptSkill {self.skill_id} L{self.level}>"
