"""GAIA-specific skill-creation recipe.

Tailors SD/TD prompts for general AI assistant tasks that require web
search, file parsing (PDF, CSV, Excel, etc.), and Python code execution.

Seeds the library with the 9 hand-authored SKILL.md guides from
``gaia/skills/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from skilleval.core.types import SkillOrigin, SkillSpec
from skilleval.skills.creation.recipe import SkillRecipe

logger = logging.getLogger(__name__)

# Root of gaia/skills/ relative to project root
_GAIA_SKILLS_ROOT = Path(__file__).resolve().parents[3] / "gaia" / "skills"

_SKILL_DIRS = [
    "search", "pdf", "csv", "excel", "txt", "xml", "json", "docx", "pdb",
]


def _load_seed_skills() -> list[SkillSpec]:
    """Load all SKILL.md files from gaia/skills/ as seed SkillSpecs."""
    seeds: list[SkillSpec] = []
    for skill_name in _SKILL_DIRS:
        skill_path = _GAIA_SKILLS_ROOT / skill_name / "SKILL.md"
        if not skill_path.exists():
            logger.warning("Seed skill not found: %s", skill_path)
            continue
        content = skill_path.read_text(encoding="utf-8")

        description = _extract_description(content, skill_name)

        seeds.append(SkillSpec(
            skill_id=f"sd_{skill_name}",
            name=skill_name,
            description=description,
            origin=SkillOrigin.SPEC_DERIVED,
            level=2,
            tool_calls=["code_executor"],
            template=content,
        ))
    logger.info("Loaded %d GAIA seed skills from %s", len(seeds), _GAIA_SKILLS_ROOT)
    return seeds


def _extract_description(content: str, fallback_name: str) -> str:
    """Extract description from YAML frontmatter or first paragraph."""
    import re
    fm = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if fm:
        for line in fm.group(1).splitlines():
            if line.strip().startswith("description:"):
                return line.split(":", 1)[1].strip()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("---"):
            return stripped[:200]
    return f"Skill guide for {fallback_name} file processing."


GAIA_SD_PROMPT = """\
You are designing a skill library for an AI agent that solves GAIA \
benchmark tasks — general-purpose questions requiring web search, \
file parsing, code execution, and multi-step reasoning.

## Dataset context

- Domain: {domain}
- Available tools:
{tools_block}
- Sample task instructions:
{sample_block}

## Your task

Write exactly {max_skills} reusable skills that help solve these tasks. \
Skills should cover:
- Information retrieval (web search, academic search)
- File parsing (PDF, CSV, Excel, JSON, XML, DOCX, TXT)
- Data analysis and computation (pandas, numpy, regex)
- Multi-step reasoning with tool composition

Each skill describes HOW to use a tool or technique. Include concrete \
Python code examples with the actual libraries (exa_py, pandas, \
pdfplumber, openpyxl, etc.).

Skills must be distinct and non-overlapping. Do not prescribe the \
reply's output format — the agent's system prompt owns that contract.

Return ONLY a JSON array of exactly {max_skills} skill objects. Each \
object must have:

- "name": short identifier, snake_case
- "description": 1-2 sentences stating WHEN to use this skill
- "tool_calls": ["code_executor"] (all skills use Python execution)
- "template": multi-section markdown with:
  ## When to use
  ## Quick Start (working Python code example)
  ## Procedure (numbered steps)
  ## Common pitfalls

Return JSON only. No markdown fences, no prose.
"""


GAIA_TD_PROMPT = """\
You are extracting reusable skills for GAIA benchmark tasks from agent \
interaction traces. GAIA tasks involve web search, file processing, \
and multi-step reasoning.

## Probe traces

{traces_block}

## Your task

Write exactly {max_skills} skills that generalize patterns visible in \
the traces. Focus on:
- Effective search strategies and query formulation
- File parsing patterns that appeared across multiple tasks
- Error recovery when tools fail or return unexpected results
- Data extraction and transformation patterns

Each skill should include concrete Python code examples.

Return ONLY a JSON array of exactly {max_skills} skill objects:

- "name" (snake_case), "description" (1-2 sentences)
- "tool_calls": ["code_executor"]
- "template" with sections: ## When to use, ## Quick Start, \
## Procedure, ## Common pitfalls

Return JSON only.
"""


def build_recipe() -> SkillRecipe:
    return SkillRecipe(
        sd_prompt=GAIA_SD_PROMPT,
        td_prompt=GAIA_TD_PROMPT,
        seed_skills=_load_seed_skills(),
        sample_task_count=3,
        max_skills=0,  # use only the hand-authored gaia/skills/* SKILL.md files
    )
