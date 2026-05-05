"""WebWalker-specific skill-creation recipe.

Tailors SD/TD prompts for web navigation tasks that require crawling
pages, extracting links, and synthesising information across multiple
hops.

Seeds the library with the 9 hand-authored SKILL.md guides from
``webwalker/skills/``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from skilleval.core.types import SkillOrigin, SkillSpec
from skilleval.skills.creation.recipe import SkillRecipe

logger = logging.getLogger(__name__)

_WEBWALKER_SKILLS_ROOT = Path(__file__).resolve().parents[3] / "webwalker" / "skills"

_SKILL_DIRS = [
    "search",
    "web_crawling",
    "html_parsing",
    "navigation",
    "url",
    "info_extraction",
    "markdown",
    "screenshot",
    "evaluation",
]


def _load_seed_skills() -> list[SkillSpec]:
    """Load all SKILL.md files from webwalker/skills/ as seed SkillSpecs."""
    seeds: list[SkillSpec] = []
    for skill_name in _SKILL_DIRS:
        skill_path = _WEBWALKER_SKILLS_ROOT / skill_name / "SKILL.md"
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
    logger.info(
        "Loaded %d WebWalker seed skills from %s",
        len(seeds), _WEBWALKER_SKILLS_ROOT,
    )
    return seeds


def _extract_description(content: str, fallback_name: str) -> str:
    """Extract description from YAML frontmatter or first paragraph."""
    fm = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if fm:
        for line in fm.group(1).splitlines():
            if line.strip().startswith("description:"):
                return line.split(":", 1)[1].strip()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("---"):
            return stripped[:200]
    return f"Skill guide for {fallback_name} in web navigation."


WEBWALKER_SD_PROMPT = """\
You are designing a skill library for an AI agent that solves \
WebWalkerQA tasks — questions that require navigating from a root URL \
through multiple pages using web crawling, link extraction, and \
information synthesis.

## Dataset context

- Domain: {domain}
- Available tools:
{tools_block}
- Sample task instructions:
{sample_block}

## Your task

Write exactly {max_skills} reusable skills that help solve these tasks. \
Skills should cover:
- Web crawling and page content extraction (crawl4ai, requests)
- Link extraction and navigation strategy (BeautifulSoup, urllib.parse)
- Search and information retrieval (exa_py, DuckDuckGo)
- Multi-hop information synthesis across pages
- Answer extraction from accumulated page observations

Each skill describes HOW to use a library or pattern. Include working \
Python code with actual libraries (crawl4ai, exa_py, beautifulsoup4, \
requests, certifi).

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


WEBWALKER_TD_PROMPT = """\
You are extracting reusable skills from WebWalkerQA agent interaction \
traces. Tasks require navigating from a root URL to find specific \
information.

## Probe traces

{traces_block}

## Your task

Write exactly {max_skills} skills that generalize patterns visible in \
the traces. Focus on:
- Effective crawling strategies for multi-hop navigation
- Link selection heuristics (which buttons/links lead to answers)
- Handling dynamic pages, redirects, and JavaScript-rendered content
- Information accumulation and early-stopping when answer is found
- Error recovery when a page returns no useful content

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
        sd_prompt=WEBWALKER_SD_PROMPT,
        td_prompt=WEBWALKER_TD_PROMPT,
        seed_skills=_load_seed_skills(),
        sample_task_count=3,
        max_skills=0,  # use only the hand-authored webwalker/skills/* SKILL.md files
    )
