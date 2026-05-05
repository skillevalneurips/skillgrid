"""Shared LLM-based skill-writing utilities.

Exposes two entry points used by both TopDown (SD) and BottomUp (TD)
creators:

- ``generate_library_from_spec``: SD — the LLM reads dataset domain,
  tool specs, and sample tasks, then outputs exactly ``max_skills``
  skill specs.
- ``generate_library_from_traces``: TD — the LLM reads a summary of
  probe traces and outputs skills that generalize the patterns.

Both use the same LLM prompt structure for the output (JSON array of
skill objects with name, description, tool_calls, template). Failures
(invalid JSON, empty response, LLM error) fall back to an empty list —
the caller treats that as "no skills built" and the evaluator handles
empty libraries gracefully.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from skilleval.core.types import EpisodeTrace, SkillOrigin, SkillSpec
from skilleval.models.base import BaseModel

logger = logging.getLogger(__name__)


# Legacy JSON-based prompts (kept for backward compatibility)
SD_PROMPT_JSON = """\
You are designing a skill library for an AI agent that will solve tasks \
from a specific dataset.

## Dataset context

- Domain: {domain}
- Available tools:
{tools_block}
- Sample task instructions:
{sample_block}

## Your task

Write exactly {max_skills} reusable skills that would help an agent solve \
this kind of task. Prefer distinct, non-overlapping procedures — do not \
produce redundant variants of the same skill.

Return ONLY a JSON array of exactly {max_skills} skill objects. Each object \
must have these fields:

- "name": short semantic identifier, snake_case
- "description": 1-2 sentences stating WHEN to use this skill
- "tool_calls": list of tool names this skill uses (a subset of the available \
tools; may be empty if the task is pure reasoning)
- "template": multi-section markdown (<= 500 words) with exactly these \
sections in this order:
  ## When to use
  ## Procedure   (numbered steps)
  ## Tool invocation examples   (omit entire section if tool_calls is empty)
  ## Output format
  ## Worked example
  ## Common pitfalls

Return JSON only. No markdown fences, no prose around it.
"""


TD_PROMPT_JSON = """\
You are extracting reusable skills from agent interaction traces.

## Probe traces

{traces_block}

## Your task

Write exactly {max_skills} reusable skills that generalize the patterns \
you observe. Focus on: common sub-procedures, recurring tool sequences, \
and recovery strategies for failures. Do not produce overlapping skills.

Return ONLY a JSON array of exactly {max_skills} skill objects. Each object \
must have these fields:

- "name": short semantic identifier, snake_case
- "description": 1-2 sentences stating WHEN to use this skill
- "tool_calls": list of tool names this skill uses
- "template": multi-section markdown (<= 500 words) with sections:
  ## When to use
  ## Procedure
  ## Tool invocation examples
  ## Output format
  ## Worked example
  ## Common pitfalls

Return JSON only.
"""


# New SKILL.md markdown-based prompts (default)
SD_PROMPT = """\
You are designing a skill library for an AI agent that will solve tasks \
from a specific dataset.

## Dataset context

- Domain: {domain}
- Available tools:
{tools_block}
- Sample task instructions:
{sample_block}

## Your task

Write exactly {max_skills} reusable skills. Each skill must be a complete \
SKILL.md file. Prefer distinct, non-overlapping procedures — do not produce \
redundant variants of the same skill.

## Required format for EACH skill

```
---
name: skill_name_snake_case
description: 1-2 sentences stating WHEN to use this skill. Be specific about trigger conditions.
---

# Skill Name In Title Case

## Overview

Brief description of what this skill does and its purpose.

## Workflow

1. First step with specific action
2. Second step with formula or method  
3. Continue with clear, actionable steps

## Example

**Input:** [Concrete example input]

**Solution:**
1. [Step applied]
2. [Continue]

**Output:** [Result]

## Common pitfalls

- Pitfall 1 and how to avoid
- Pitfall 2 and how to avoid
```

Separate each skill with a line containing only: ===SKILL_SEPARATOR===

Return the skills directly. No JSON, no extra prose.
"""


TD_PROMPT = """\
You are extracting reusable skills from agent interaction traces.

## Probe traces

{traces_block}

## Your task

Write exactly {max_skills} skills that generalize patterns from the traces. \
Each skill must be a complete SKILL.md file. Focus on: common sub-procedures, \
recurring reasoning patterns, and recovery strategies for failures. \
Do not produce overlapping skills.

## Required format for EACH skill

```
---
name: skill_name_snake_case
description: 1-2 sentences stating WHEN to use. Be specific about triggers.
---

# Skill Name In Title Case

## Overview

Brief description of what this skill does based on the traces.

## Workflow

1. Numbered steps with formulas/methods
2. Be specific and actionable

## Example

**Problem:** [From traces or similar]

**Solution:** [Step-by-step]

**Output:** [Result]

## Common pitfalls

- Common mistakes from traces
```

Separate each skill with: ===SKILL_SEPARATOR===

Return skills directly. No JSON.
"""


def _format_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return "  (none — this is a pure-reasoning dataset)"
    lines = []
    for t in tools:
        name = t.get("name") or t.get("function", {}).get("name", "unknown")
        desc = t.get("description") or t.get("function", {}).get("description", "")
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


def _format_samples(sample_tasks: list[str]) -> str:
    if not sample_tasks:
        return "  (no samples)"
    lines = []
    for i, s in enumerate(sample_tasks[:5], 1):
        truncated = s[:500] + ("…" if len(s) > 500 else "")
        lines.append(f"  {i}. {truncated}")
    return "\n".join(lines)


def _summarize_trace(trace: EpisodeTrace) -> str:
    status = "SUCCESS" if trace.success else "FAILED"
    tools_used = [e.tool_name for e in trace.entries if e.tool_name]
    final = (trace.final_answer or "")[:120]
    return (
        f"- task={trace.task_id} [{status}] "
        f"steps={len(trace.entries)} tools={tools_used[:10]} "
        f"final_answer={final!r}"
    )


def _detailed_trace_summary(trace: EpisodeTrace, max_chars: int = 2000) -> str:
    """Rich trace summary that includes the problem text and reasoning chain.

    Used by TD skill generation so the LLM can see *what* the agent
    actually reasoned about, not just a one-line task-id.
    """
    status = "SUCCESS" if trace.success else "FAILED"
    parts: list[str] = [f"### Attempt: {trace.task_id} [{status}]"]

    if trace.task_instruction:
        instr = trace.task_instruction[:600]
        parts.append(f"**Problem:** {instr}")

    reasoning_parts: list[str] = []
    for entry in trace.entries:
        if entry.action == "final_answer":
            continue
        chunk = entry.observation or ""
        if not chunk and entry.action == "generate":
            chunk = entry.action
        if entry.tool_name:
            chunk = f"[{entry.tool_name}] {chunk}"
        if chunk:
            reasoning_parts.append(chunk.strip())

    reasoning = "\n".join(reasoning_parts)
    budget = max_chars - sum(len(p) for p in parts) - 50
    if len(reasoning) > budget:
        reasoning = reasoning[:budget] + "..."
    if reasoning:
        parts.append(f"**Reasoning chain:**\n{reasoning}")

    final = (trace.final_answer or "")[:200]
    parts.append(f"**Final answer:** {final!r}")
    return "\n\n".join(parts)


def _format_traces(traces: list[EpisodeTrace], max_traces: int = 15) -> str:
    if not traces:
        return "  (no traces)"
    return "\n".join(_summarize_trace(t) for t in traces[:max_traces])


def _format_traces_detailed(
    traces: list[EpisodeTrace], max_traces: int = 10,
) -> str:
    """Format traces with full problem text and reasoning for TD prompts."""
    if not traces:
        return "  (no traces)"
    return "\n\n---\n\n".join(
        _detailed_trace_summary(t) for t in traces[:max_traces]
    )


def _sanitize_json_escapes(raw: str) -> str:
    r"""Fix invalid JSON escape sequences that LLMs produce from LaTeX.

    JSON only allows \", \\, \/, \b, \f, \n, \r, \t, and \uXXXX.
    LLMs embedding LaTeX write things like \( \frac \alpha etc.
    We double-escape any backslash that isn't part of a valid JSON escape.
    """
    return re.sub(
        r'\\(?!["\\/bfnrtu])',
        r"\\\\",
        raw,
    )


def _parse_skill_markdown(text: str, max_skills: int) -> list[dict[str, Any]]:
    """Parse multiple SKILL.md files separated by ===SKILL_SEPARATOR===.

    Each skill has simple YAML frontmatter (name, description) and a markdown body.
    Returns a list of dicts with keys: name, description, tool_calls, template.
    """
    skills: list[dict[str, Any]] = []
    chunks = text.split("===SKILL_SEPARATOR===")

    for chunk in chunks[:max_skills]:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Find YAML frontmatter between --- delimiters
        # Handle case where chunk starts with --- or has content before it
        if chunk.startswith("---"):
            # Standard frontmatter format
            parts = chunk.split("---", 2)
            if len(parts) < 3:
                logger.warning("Malformed SKILL.md chunk: missing closing ---")
                continue
            frontmatter_text = parts[1].strip()
            body = parts[2].strip()
        else:
            # Try to find --- anywhere in the chunk
            first_delim = chunk.find("---")
            if first_delim == -1:
                logger.warning("No YAML frontmatter found in chunk")
                continue
            rest = chunk[first_delim + 3:]
            second_delim = rest.find("---")
            if second_delim == -1:
                logger.warning("Malformed SKILL.md chunk: missing closing ---")
                continue
            frontmatter_text = rest[:second_delim].strip()
            body = rest[second_delim + 3:].strip()

        # Parse simple frontmatter (just name: and description: lines)
        frontmatter: dict[str, Any] = {}
        current_key = None
        current_val: list[str] = []
        for line in frontmatter_text.split("\n"):
            if ":" in line and not line.startswith(" "):
                # Save previous key if any
                if current_key:
                    frontmatter[current_key] = " ".join(current_val).strip()
                key, val = line.split(":", 1)
                current_key = key.strip()
                current_val = [val.strip()]
            elif current_key:
                # Continuation of previous value
                current_val.append(line.strip())
        # Save last key
        if current_key:
            frontmatter[current_key] = " ".join(current_val).strip()

        if not frontmatter.get("name"):
            logger.warning("Skill has no name in frontmatter")
            continue

        skills.append({
            "name": frontmatter.get("name", ""),
            "description": frontmatter.get("description", ""),
            "tool_calls": [],
            "template": body,
        })

    return skills


def _parse_skill_array(text: str, max_skills: int) -> list[dict[str, Any]]:
    """Parse a JSON array from the LLM response, tolerating code fences."""
    cleaned = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    # Sometimes the LLM prefixes with an intro sentence; find the first [.
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in LLM output")
    payload = cleaned[start : end + 1]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        # Retry after sanitizing LaTeX backslash escapes.
        payload = _sanitize_json_escapes(payload)
        parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise ValueError("LLM output is not a JSON array")
    return parsed[:max_skills]


def _coerce_to_str(value: Any) -> str:
    """Ensure a value is a string. If the LLM returned a dict or list instead
    of a flat string (common with smaller models), flatten it to readable text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            heading = k.replace("_", " ").title()
            parts.append(f"## {heading}\n{v}")
        return "\n\n".join(parts)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value).strip() if value else ""


def _spec_from_dict(d: dict[str, Any], index: int, origin: SkillOrigin) -> SkillSpec:
    prefix = "sd" if origin == SkillOrigin.SPEC_DERIVED else "td"
    name = (d.get("name") or f"skill_{index}").strip()
    safe_name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or f"skill_{index}"
    skill_id = f"{prefix}_{safe_name}"
    return SkillSpec(
        skill_id=skill_id,
        name=name,
        description=_coerce_to_str(d.get("description", "")) or name,
        origin=origin,
        level=2,
        tool_calls=list(d.get("tool_calls", []) or []),
        template=_coerce_to_str(d.get("template", "")),
    )


def generate_library_from_spec(
    writer_model: BaseModel,
    domain: str,
    tools: list[dict[str, Any]],
    sample_tasks: list[str],
    max_skills: int,
    prompt_template: str | None = None,
    extra_context: dict[str, Any] | None = None,
) -> list[SkillSpec]:
    """SD: LLM generates `max_skills` skills from dataset context.

    ``prompt_template`` overrides the built-in SD_PROMPT when provided.
    It must accept the same ``{domain}``, ``{tools_block}``,
    ``{sample_block}``, ``{max_skills}`` substitutions. ``extra_context``
    is merged in for recipe-specific placeholders.
    """
    template = prompt_template or SD_PROMPT
    format_kwargs = {
        "domain": domain,
        "tools_block": _format_tools(tools),
        "sample_block": _format_samples(sample_tasks),
        "max_skills": max_skills,
    }
    if extra_context:
        format_kwargs.update(extra_context)
    try:
        prompt = template.format(**format_kwargs)
    except KeyError as exc:
        logger.warning(
            "SD prompt template missing placeholder %s — falling back to default.",
            exc,
        )
        prompt = SD_PROMPT.format(
            domain=domain,
            tools_block=_format_tools(tools),
            sample_block=_format_samples(sample_tasks),
            max_skills=max_skills,
        )

    logger.info("SKILL_WRITER SD PROMPT:\n%s", prompt)
    try:
        response = writer_model.generate([{"role": "user", "content": prompt}])
        raw = response.text or ""
        logger.info("SKILL_WRITER SD RAW RESPONSE:\n%s", raw)

        # Try markdown parsing first (new format), fall back to JSON (legacy)
        if "===SKILL_SEPARATOR===" in raw or (raw.strip().startswith("---") and "\n---" in raw):
            items = _parse_skill_markdown(raw, max_skills)
            if items:
                logger.info("Parsed %d skills using markdown format", len(items))
            else:
                logger.info("Markdown parsing returned empty, trying JSON")
                items = _parse_skill_array(raw, max_skills)
        else:
            items = _parse_skill_array(raw, max_skills)
    except Exception as exc:
        logger.warning("Skill-writer (SD) failed: %s", exc)
        return []

    specs: list[SkillSpec] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        specs.append(_spec_from_dict(item, i, SkillOrigin.SPEC_DERIVED))
    logger.info("Skill-writer (SD) produced %d/%d skills", len(specs), max_skills)
    return specs


def generate_library_from_traces(
    writer_model: BaseModel,
    traces: list[EpisodeTrace],
    max_skills: int,
    prompt_template: str | None = None,
    trace_summarizer: Any = None,
    extra_context: dict[str, Any] | None = None,
) -> list[SkillSpec]:
    """TD: LLM generates `max_skills` skills by generalizing over probe traces.

    ``prompt_template`` overrides TD_PROMPT; ``trace_summarizer`` is a
    callable that takes ``list[EpisodeTrace]`` and returns a string —
    overrides the default one-line-per-trace summary.
    """
    template = prompt_template or TD_PROMPT
    if trace_summarizer:
        traces_block = trace_summarizer(traces)
    else:
        # Use detailed summaries when task_instruction is available.
        has_instructions = any(t.task_instruction for t in traces)
        if has_instructions:
            traces_block = _format_traces_detailed(traces)
        else:
            traces_block = _format_traces(traces)
    format_kwargs = {
        "traces_block": traces_block,
        "max_skills": max_skills,
    }
    if extra_context:
        format_kwargs.update(extra_context)
    try:
        prompt = template.format(**format_kwargs)
    except KeyError as exc:
        logger.warning(
            "TD prompt template missing placeholder %s — falling back to default.",
            exc,
        )
        prompt = TD_PROMPT.format(
            traces_block=_format_traces(traces),
            max_skills=max_skills,
        )

    logger.info("SKILL_WRITER TD PROMPT:\n%s", prompt)
    try:
        response = writer_model.generate([{"role": "user", "content": prompt}])
        raw = response.text or ""
        logger.info("SKILL_WRITER TD RAW RESPONSE:\n%s", raw)

        # Try markdown parsing first (new format), fall back to JSON (legacy)
        if "===SKILL_SEPARATOR===" in raw or (raw.strip().startswith("---") and "\n---" in raw):
            items = _parse_skill_markdown(raw, max_skills)
            if items:
                logger.info("Parsed %d skills using markdown format", len(items))
            else:
                logger.info("Markdown parsing returned empty, trying JSON")
                items = _parse_skill_array(raw, max_skills)
        else:
            items = _parse_skill_array(raw, max_skills)
    except Exception as exc:
        logger.warning("Skill-writer (TD) failed: %s", exc)
        return []

    specs: list[SkillSpec] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        specs.append(_spec_from_dict(item, i, SkillOrigin.TRACE_DERIVED))
    logger.info("Skill-writer (TD) produced %d/%d skills", len(specs), max_skills)
    return specs
