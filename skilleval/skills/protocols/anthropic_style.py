"""Skill protocol 3: Modern Anthropic-Style Skills.

Skills are structured as self-contained SKILL.md files with:
  - When to use (trigger conditions)
  - Step-by-step instructions
  - Validation checks
  - Metadata (tags, dependencies)

Reference: https://github.com/anthropics/skills
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from skilleval.core.registry import skill_protocol_registry
from skilleval.models.base import BaseModel
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)


@dataclass
class AnthropicSkillCard:
    """Structured representation of an Anthropic-style skill."""
    name: str
    description: str
    when_to_use: str
    steps: list[str]
    validation: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.name}",
            "",
            f"**Description:** {self.description}",
            "",
            "## When to Use",
            self.when_to_use,
            "",
            "## Steps",
        ]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"{i}. {step}")
        if self.validation:
            lines.append("")
            lines.append("## Validation")
            for v in self.validation:
                lines.append(f"- {v}")
        if self.tags:
            lines.append("")
            lines.append(f"**Tags:** {', '.join(self.tags)}")
        if self.dependencies:
            lines.append(f"**Dependencies:** {', '.join(self.dependencies)}")
        return "\n".join(lines)


@skill_protocol_registry.register("anthropic_style")
class AnthropicStyleProtocol:
    """Skills follow the Anthropic SKILL.md convention.

    The agent dynamically selects and follows structured skill cards.
    """

    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def skill_to_card(self, skill: Any) -> AnthropicSkillCard:
        """Convert a BaseSkill / PromptSkill into an AnthropicSkillCard."""
        spec = skill.spec
        return AnthropicSkillCard(
            name=spec.name,
            description=spec.description,
            when_to_use=f"Use when the task requires: {', '.join(spec.tool_calls)}",
            steps=self._decompose_template(spec.template),
            validation=spec.postconditions,
            tags=[spec.origin.value, f"level-{spec.level}"],
            dependencies=spec.tool_calls,
        )

    def build_prompt(
        self,
        task_instruction: str,
        skill_cards: list[AnthropicSkillCard],
        env_tools: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Build prompt with Anthropic-style skill cards."""
        skills_md = "\n\n---\n\n".join(card.to_markdown() for card in skill_cards)
        tool_section = "\n".join(
            f"- {t['name']}: {t.get('description', '')}" for t in env_tools
        )

        system = (
            "You are a capable agent with access to structured skills and tools.\n\n"
            "## Skill Library\n\n"
            f"{skills_md}\n\n"
            f"## Available Tools\n{tool_section}\n\n"
            "## Instructions\n"
            "1. Read the task carefully.\n"
            "2. Check which skill(s) match the task (see 'When to Use').\n"
            "3. Follow the steps in the matching skill.\n"
            "4. Validate your work using the validation checks.\n"
            "5. If no skill matches, reason from first principles using the tools."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task_instruction},
        ]

    def run_episode(
        self,
        task_instruction: str,
        library: SkillLibrary,
        env_tools: list[dict[str, str]],
        max_turns: int = 5,
    ) -> list[dict[str, Any]]:
        """Run one task episode with Anthropic-style protocol."""
        cards = [self.skill_to_card(s) for s in library.list_skills()]
        messages = self.build_prompt(task_instruction, cards, env_tools)
        steps: list[dict[str, Any]] = []

        for turn in range(max_turns):
            response = self.model.generate(messages)
            steps.append({
                "step": turn,
                "action": "generate",
                "text": response.text,
                "tokens": response.input_tokens + response.output_tokens,
            })

            if self._is_complete(response.text):
                break

            messages.append({"role": "assistant", "content": response.text})
            messages.append({
                "role": "user",
                "content": "Continue following the skill steps or provide the final answer.",
            })

        return steps

    @staticmethod
    def _decompose_template(template: str) -> list[str]:
        if not template:
            return ["Analyze the task.", "Execute using available tools.", "Verify the result."]
        parts = [p.strip() for p in template.split(".") if p.strip()]
        return parts if parts else [template]

    @staticmethod
    def _is_complete(text: str) -> bool:
        lower = text.lower()
        return any(m in lower for m in ["final answer", "the answer is", "answer:", "solution:"])
