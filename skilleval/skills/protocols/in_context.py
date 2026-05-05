"""Skill protocol 2: Conventional In-Context Skills.

Skills are injected as text instructions in the system/user prompt.
The model follows the skill descriptions as step-by-step procedures.
"""

from __future__ import annotations

import logging
import json
import re
from typing import Any

from skilleval.core.registry import skill_protocol_registry
from skilleval.models.base import BaseModel
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)


@skill_protocol_registry.register("in_context")
class InContextProtocol:
    """Skills are serialized into the prompt context.

    The model is instructed to follow skills as reusable procedures.
    This can result in long contexts for large skill libraries.
    """

    def __init__(self, model: BaseModel, max_skills_in_context: int = 10) -> None:
        self.model = model
        self.max_skills_in_context = max_skills_in_context

    def build_prompt(
        self,
        task_instruction: str,
        skills: list[Any],
        env_tools: list[dict[str, str]],
        answer_format: str | None = None,
    ) -> list[dict[str, str]]:
        """Construct the full prompt with in-context skills."""
        skill_section = self._format_skills(skills)
        tool_section = self._format_tools(env_tools)

        final_instruction = (
            "Return only the final answer in the requested format."
            if _is_json_only_answer_format(answer_format)
            else "Reason step by step and show your work."
        )
        system = (
            "You are a capable agent with access to reusable skills and tools.\n\n"
            f"## Available Skills\n{skill_section}\n\n"
            f"## Available Tools\n{tool_section}\n\n"
            "Follow the skills as step-by-step procedures when applicable. "
            "Use tools directly when no skill matches. "
            f"{final_instruction}"
        )
        if answer_format:
            # Recency bias: the scoring contract is the last thing in the
            # system prompt so it sits closest to the user task.
            system = f"{system}\n\n{answer_format}"
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
        answer_format: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run one task episode with in-context skill protocol.

        Multi-turn: the model may produce intermediate reasoning steps
        and the orchestrator can inject feedback.
        """
        skills = library.list_skills()[: self.max_skills_in_context]
        messages = self.build_prompt(
            task_instruction, skills, env_tools, answer_format=answer_format,
        )
        steps: list[dict[str, Any]] = []

        for turn in range(max_turns):
            response = self.model.generate(messages)
            steps.append({
                "step": turn,
                "action": "generate",
                "text": response.text,
                "tokens": response.input_tokens + response.output_tokens,
            })

            if self._is_final(response.text):
                break

            messages.append({"role": "assistant", "content": response.text})
            messages.append({
                "role": "user",
                "content": "Continue with the next step, or provide the final answer.",
            })

        return steps

    def _format_skills(self, skills: list[Any]) -> str:
        parts = []
        for i, skill in enumerate(skills, 1):
            if hasattr(skill, "to_prompt"):
                parts.append(f"{i}. {skill.to_prompt()}")
            else:
                parts.append(f"{i}. {skill.spec.name}: {skill.spec.description}")
        return "\n\n".join(parts) if parts else "(No skills available)"

    @staticmethod
    def _format_tools(tools: list[dict[str, str]]) -> str:
        parts = []
        for t in tools:
            parts.append(f"- **{t['name']}**: {t.get('description', '')}")
        return "\n".join(parts) if parts else "(No tools available)"

    @staticmethod
    def _is_final(text: str) -> bool:
        if _contains_recommendation_json(text):
            return True
        lower = text.lower()
        return any(marker in lower for marker in [
            "final answer", "the answer is", "answer:", "solution:",
            '"recommendations"', "recommendations:",
        ])


def _is_json_only_answer_format(answer_format: str | None) -> bool:
    if not answer_format:
        return False
    lower = answer_format.lower()
    return "json only" in lower or '"recommendations"' in lower


def _contains_recommendation_json(text: str) -> bool:
    raw = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return False
    try:
        parsed = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and isinstance(parsed.get("recommendations"), list)
