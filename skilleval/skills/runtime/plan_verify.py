"""Runtime policy: Plan-Retrieve (PR).

PR first retrieves a small skill bundle with the same semantic retrieval
procedure used by RR. A planner then chooses one retrieved skill using only
names/descriptions, and an executor sees that skill's template for one answer
attempt.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from skilleval.core.registry import runtime_policy_registry
from skilleval.core.types import RuntimePolicy, TaskInstance
from skilleval.models.base import BaseModel
from skilleval.skills.library import SkillLibrary
from skilleval.skills.runtime.retrieve_route import RetrieveRoutePolicy

logger = logging.getLogger(__name__)


def _build_skills_context(library: SkillLibrary) -> str:
    """Format skills for the planner without exposing templates."""
    blocks = []
    for skill in library.list_skills():
        blocks.append(
            f"- ID: {skill.skill_id}\n"
            f"  Name: {skill.spec.name}\n"
            f"  Description: {skill.spec.description}"
        )
    return "\n".join(blocks) if blocks else "(no skills available)"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_math_answer(text: str) -> str | None:
    if not text:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            ans = line[7:].strip()
            ans = re.sub(r"^\*+|\*+$", "", ans).strip()
            ans = re.sub(r"^\\boxed\{([^}]*)\}$", r"\1", ans)
            if ans:
                return ans

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    answer_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(-?\d+(?:\.\d+)?)",
        r"(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?answer\s+is\s*(-?\d+(?:\.\d+)?)",
        r"=\s*(-?\d+(?:\.\d+)?)\s*$",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    numbers = re.findall(r"\b-?\d+(?:\.\d+)?\b", text)
    if numbers:
        return numbers[-1].strip()
    return None


def _wants_math_answer(answer_format: str | None) -> bool:
    if not answer_format:
        return False
    lower = answer_format.lower()
    return "answer:" in lower or ("integer" in lower and "answer" in lower)


def _extract_final_answer(text: str, answer_format: str | None) -> str:
    """Return the dataset-specific final answer when possible."""
    raw = (text or "").strip()
    if not raw:
        return ""

    if answer_format and "recommendations" in answer_format.lower():
        parsed = _extract_json_object(raw)
        if parsed is not None:
            return json.dumps(parsed, ensure_ascii=False)

    if _wants_math_answer(answer_format):
        return _extract_math_answer(raw) or raw

    return raw


def _looks_valid_for_format(text: str, answer_format: str | None) -> bool:
    if not text.strip():
        return False
    if answer_format and "recommendations" in answer_format.lower():
        parsed = _extract_json_object(text)
        recs = parsed.get("recommendations") if parsed else None
        return isinstance(recs, list)
    if _wants_math_answer(answer_format):
        return _extract_math_answer(text) is not None
    return True


@runtime_policy_registry.register("PV")
class PlanVerifyPolicy:
    """Plan-Retrieve executor.

    The registry key remains ``PV`` for backward compatibility with existing
    run code, but the taxonomy runner maps paper PR to this policy.
    """

    def __init__(
        self,
        model: BaseModel,
        max_replans: int = 1,
        answer_format: str | None = None,
        retrieval_bundle_size: int = 1,
        description_only: bool = False,
    ) -> None:
        self.model = model
        self.max_replans = max_replans
        self.answer_format = answer_format
        self.retrieval_bundle_size = max(1, retrieval_bundle_size)
        self.description_only = description_only

    @property
    def policy_type(self) -> RuntimePolicy:
        return RuntimePolicy.PLAN_VERIFY

    def pick_skill(
        self,
        task: TaskInstance,
        candidate_library: SkillLibrary,
    ) -> dict[str, Any]:
        skills_context = _build_skills_context(candidate_library)
        prompt = (
            "You are a planning agent. Select the most useful skill for the "
            "task from the retrieved candidate skills. You may see only skill "
            "names and descriptions, not skill templates.\n\n"
            f"Task:\n{task.instruction}\n\n"
            f"Retrieved candidate skills:\n{skills_context}\n\n"
            "Return only JSON with this shape:\n"
            '{"skill_id":"<exact ID>","instruction":"<brief executor instruction>"}'
        )
        response = self.model.generate([{"role": "user", "content": prompt}])
        return self._parse_pick(response.text, candidate_library)

    def execute(
        self,
        task: TaskInstance,
        skill: Any,
        instruction: str,
        retry: bool = False,
    ) -> str:
        retry_note = ""
        if retry:
            retry_note = (
                "Your previous output did not match the required answer format. "
                "Return only the final answer in the required format.\n\n"
            )

        format_hint = self.answer_format or "Return the final answer only."
        prompt = (
            f"{retry_note}"
            "Use the selected skill guidance to solve the task.\n\n"
            f"Task:\n{task.instruction}\n\n"
            f"Planner instruction:\n{instruction}\n\n"
            f"Selected skill: {skill.spec.name}\n"
            f"Skill description: {skill.spec.description}\n"
            f"Skill guidance:\n{skill.spec.template}\n\n"
            f"Required answer format:\n{format_hint}"
        )
        try:
            response = self.model.generate([{"role": "user", "content": prompt}])
            return (response.text or "").strip()
        except Exception as exc:
            logger.warning("PR executor failed: %s", exc)
            return ""

    def run_episode(
        self,
        task: TaskInstance,
        library: SkillLibrary,
        max_steps: int = 20,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        candidate_library = self._retrieve_candidates(task, library)
        candidate_ids = [s.skill_id for s in candidate_library.list_skills()]
        if not candidate_ids:
            history.append({"step": 0, "action": "final_answer", "text": ""})
            return history

        pick = self.pick_skill(task, candidate_library)
        skill_id = pick.get("skill_id", "")
        instruction = pick.get("instruction", "solve the task")

        skill = self._resolve_skill(skill_id, candidate_library)
        if skill is None:
            history.append({"step": 0, "action": "final_answer", "text": ""})
            return history

        history.append({
            "step": 0,
            "action": "plan",
            "skill_id": skill.skill_id,
            "tool_name": skill.skill_id,
            "instruction": instruction,
            "selected_skill_ids": candidate_ids,
            "observation": f"Planner selected skill: {skill.skill_id}",
        })

        result = self.execute(task, skill, instruction)
        if not _looks_valid_for_format(result, self.answer_format):
            logger.info("PR: executor output failed format check; retrying once")
            result = self.execute(task, skill, instruction, retry=True)

        library.record_usage(skill.skill_id)
        final_answer = _extract_final_answer(result, self.answer_format)

        history.append({
            "step": 1,
            "action": "tool_call",
            "skill_id": skill.skill_id,
            "tool_name": skill.skill_id,
            "result": result,
            "observation": result,
            "status": "passed",
        })
        history.append({
            "step": 2,
            "action": "final_answer",
            "skill_id": skill.skill_id,
            "text": final_answer,
        })
        return history

    def _retrieve_candidates(
        self,
        task: TaskInstance,
        library: SkillLibrary,
    ) -> SkillLibrary:
        retriever = RetrieveRoutePolicy(
            self.model,
            retrieval_bundle_size=self.retrieval_bundle_size,
            description_only=self.description_only,
        )
        candidates = retriever.select_skills(task, library)
        candidate_library = SkillLibrary()
        for skill in candidates:
            candidate_library.add(skill)
        return candidate_library

    @staticmethod
    def _resolve_skill(skill_id: str, library: SkillLibrary) -> Any | None:
        skill = library.get(skill_id)
        if skill is not None:
            return skill
        for candidate in library.list_skills():
            if candidate.spec.name == skill_id or candidate.spec.name.lower() == skill_id.lower():
                return candidate
            if skill_id and (skill_id in candidate.skill_id or candidate.skill_id.endswith(skill_id)):
                return candidate
        skills = library.list_skills()
        if skills:
            logger.info("PR: skill_id '%s' not found; falling back to '%s'", skill_id, skills[0].skill_id)
            return skills[0]
        return None

    @staticmethod
    def _parse_pick(text: str, library: SkillLibrary) -> dict[str, Any]:
        parsed = _extract_json_object(text)
        if parsed:
            return {
                "skill_id": str(parsed.get("skill_id", "")),
                "instruction": str(parsed.get("instruction", "solve the task")),
            }
        skills = library.list_skills()
        fallback_id = skills[0].skill_id if skills else ""
        return {"skill_id": fallback_id, "instruction": (text or "")[:200]}
