"""LLM-based skill updater for Batch Update (BU) evolution.

After each round, the LLM rewrites every skill in the library using
the round's traces as feedback. Library size stays fixed.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from skilleval.core.types import EpisodeTrace, SkillSpec
from skilleval.models.base import BaseModel
from skilleval.skills.creation.llm_skill_writer import generate_library_from_traces
from skilleval.skills.creation.top_down import PromptSkill
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)

SKILL_UPDATE_PROMPT = """\
You are a skill optimizer for an AI agent framework.

A "skill" is a reusable procedure (text instructions or tool-call \
sequence) that guides the agent in solving a type of problem. \
You will be given the full skill specification and a balanced sample \
of agent episode traces — both successes and failures.

Your job: rewrite the skill to make it more effective based on the traces.

## Current Skill (full specification)
```json
{skill_json}
```

## Agent Traces (mix of successes and failures)
Each trace shows: the task, whether it succeeded, and what the agent \
did step by step. The agent had access to this skill (and possibly \
others) during every episode.

{traces_text}

## Instructions
Analyze the traces:
- Where did agents succeed? What pattern worked?
- Where did agents fail? What went wrong?
- Is the current skill template helpful or misleading?
- Did this skill contribute to the outcome, or was it irrelevant?

If the skill is already effective, keep its core approach and refine \
the wording. If the skill is causing confusion or errors, rewrite \
the instructions significantly.

Return ONLY valid JSON with these fields:
{{
  "name": "skill name",
  "description": "what this skill does and when to use it",
  "tool_calls": ["tool1", "tool2"],
  "template": "step-by-step instructions for executing this skill",
  "preconditions": ["when to use this skill"],
  "postconditions": ["what should be true after"]
}}
"""


def _format_trace(trace: EpisodeTrace) -> str:
    """Format a single trace into readable text for the LLM."""
    status = "SUCCESS" if trace.success else "FAILED"
    lines = [f"### Task: {trace.task_id} [{status}]"]

    for entry in trace.entries:
        step_line = f"  Step {entry.step}: "
        if entry.tool_name:
            # Show tool call with shortened args
            args_str = json.dumps(entry.tool_args, default=str)
            if len(args_str) > 120:
                args_str = args_str[:120] + "..."
            step_line += f"{entry.tool_name}({args_str})"
        else:
            action_str = entry.action[:100] if entry.action else "no action"
            step_line += action_str

        if entry.error:
            step_line += f" → ERROR: {entry.error[:100]}"
        elif entry.observation:
            obs = entry.observation[:80]
            step_line += f" → {obs}"

        if entry.skill_used:
            step_line += f"  [skill: {entry.skill_used}]"

        lines.append(step_line)

    return "\n".join(lines)


def _has_repair_signal(trace: EpisodeTrace) -> bool:
    if not trace.success:
        return True
    score = trace.metadata.get("gold_score", {})
    if not isinstance(score, dict):
        return False
    if float(score.get("schema_valid", 1.0)) < 1.0:
        return True
    if float(score.get("context_repeat_rate", 0.0)) > 0.0:
        return True
    if "exact_match" in score and float(score.get("exact_match", 1.0)) < 1.0:
        return True
    return False


def _create_recovery_skills(
    traces: list[EpisodeTrace],
    model: BaseModel,
    max_new_skills: int,
    prompt_template: str | None = None,
    trace_summarizer: Any = None,
    extra_context: dict[str, Any] | None = None,
) -> list[PromptSkill]:
    if max_new_skills <= 0 or not traces:
        return []
    specs = generate_library_from_traces(
        writer_model=model,
        traces=traces,
        max_skills=max_new_skills,
        prompt_template=prompt_template,
        trace_summarizer=trace_summarizer,
        extra_context=extra_context,
    )
    return [PromptSkill(spec) for spec in specs]


def _select_balanced_traces(
    traces: list[EpisodeTrace],
    max_traces: int = 10,
    fail_ratio: float = 0.7,
    seed: int = 42,
) -> list[EpisodeTrace]:
    """Select a balanced mix of failure and success traces.

    Target ratio is ``fail_ratio`` failures to ``1 - fail_ratio`` successes.
    When one side can't fill its quota the other side takes the slack.
    """
    failures = [t for t in traces if not t.success]
    successes = [t for t in traces if t.success]

    fail_target = int(max_traces * fail_ratio)
    success_target = max_traces - fail_target

    rng = random.Random(seed)

    fail_pick = min(fail_target, len(failures))
    success_pick = min(success_target, len(successes))

    # Overflow: if one side is short, give its empty slots to the other.
    fail_slack = fail_target - fail_pick
    success_slack = success_target - success_pick

    fail_pick = min(fail_pick + success_slack, len(failures))
    success_pick = min(success_pick + fail_slack, len(successes))

    sampled_fails = rng.sample(failures, fail_pick) if fail_pick else []
    sampled_successes = rng.sample(successes, success_pick) if success_pick else []

    selected = sampled_fails + sampled_successes
    rng.shuffle(selected)

    logger.info(
        "Balanced trace selection: %d failures + %d successes = %d total "
        "(from %d fail / %d success available)",
        len(sampled_fails), len(sampled_successes), len(selected),
        len(failures), len(successes),
    )
    return selected


def update_skill(
    skill: PromptSkill,
    traces: list[EpisodeTrace],
    model: BaseModel,
) -> PromptSkill:
    """Use the LLM to rewrite a single skill based on traces.

    ``traces`` should already be a balanced selection (via
    ``_select_balanced_traces``).  Every skill receives the same set
    and the LLM decides whether/how the skill should change.
    """
    if not traces:
        logger.info(
            "No traces for skill %s, keeping as-is.", skill.skill_id,
        )
        return skill

    traces_text = "\n\n".join(_format_trace(t) for t in traces)
    skill_json = json.dumps(skill.spec.to_dict(), indent=2, default=str)

    prompt = SKILL_UPDATE_PROMPT.format(
        skill_json=skill_json,
        traces_text=traces_text,
    )

    response = model.generate([{"role": "user", "content": prompt}])

    try:
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            # Remove opening fence (```json or ```)
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        updated = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "LLM returned invalid JSON for skill %s, keeping original.",
            skill.skill_id,
        )
        return skill

    updated_spec = SkillSpec(
        skill_id=skill.spec.skill_id,
        name=updated.get("name", skill.spec.name),
        description=updated.get("description", skill.spec.description),
        origin=skill.spec.origin,
        level=skill.spec.level,
        tool_calls=updated.get("tool_calls", skill.spec.tool_calls),
        template=updated.get("template", skill.spec.template),
        preconditions=updated.get("preconditions", skill.spec.preconditions),
        postconditions=updated.get("postconditions", skill.spec.postconditions),
    )

    logger.info("Updated skill %s → %s", skill.skill_id, updated_spec.name)
    return PromptSkill(updated_spec)


def update_library(
    library: SkillLibrary,
    traces: list[EpisodeTrace],
    model: BaseModel,
    max_traces: int = 10,
    fail_ratio: float = 0.7,
    max_new_skills: int = 0,
    allow_bootstrap: bool = False,
    recovery_prompt_template: str | None = None,
    trace_summarizer: Any = None,
    extra_context: dict[str, Any] | None = None,
) -> SkillLibrary:
    """Rewrite skills and optionally add trace-derived recovery skills.

    Selects a balanced mix of failure/success traces once, then feeds
    the same set to every existing skill. If the library is empty, BU can
    bootstrap trace-derived recovery skills so ``NL+BU`` is not degenerate.

    If every trace has no repair signal there is no signal for improvement,
    so the library is returned unchanged.
    """
    if not traces:
        logger.info("No traces available for library update.")
        return library

    if not any(_has_repair_signal(t) for t in traces):
        logger.info(
            "All %d traces have no repair signal -- library unchanged this round",
            len(traces),
        )
        return library

    selected = _select_balanced_traces(
        traces, max_traces=max_traces, fail_ratio=fail_ratio,
    )

    skills = library.list_skills()
    updated_library = SkillLibrary()

    if not skills:
        if not allow_bootstrap:
            logger.info("Library empty and bootstrap disabled; unchanged.")
            return library
        for recovery_skill in _create_recovery_skills(
            selected,
            model,
            max_new_skills=max_new_skills,
            prompt_template=recovery_prompt_template,
            trace_summarizer=trace_summarizer,
            extra_context=extra_context,
        ):
            updated_library.add(recovery_skill)
        logger.info(
            "Library bootstrapped from traces: %d recovery skills",
            updated_library.size,
        )
        return updated_library

    for skill in skills:
        updated_skill = update_skill(skill, selected, model)
        updated_library.add(updated_skill)

    for recovery_skill in _create_recovery_skills(
        selected,
        model,
        max_new_skills=max_new_skills,
        prompt_template=recovery_prompt_template,
        trace_summarizer=trace_summarizer,
        extra_context=extra_context,
    ):
        updated_library.add(recovery_skill)

    logger.info(
        "Library updated: %d skills rewritten, %d total skills after recovery additions",
        len(skills), updated_library.size,
    )
    return updated_library
