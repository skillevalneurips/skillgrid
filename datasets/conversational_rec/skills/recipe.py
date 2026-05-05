"""Skill-creation recipe for Reddit-V2 conversational recommendation."""

from __future__ import annotations

from skilleval.core.types import SkillOrigin, SkillSpec
from skilleval.skills.creation.recipe import SkillRecipe


CONVREC_SD_PROMPT = """\
You are designing a skill library for a movie recommendation agent evaluated \
on Reddit-V2 conversational recommendation tasks.

## Dataset context

- Domain: {domain}
- Available tools:
{tools_block}
- Sample task instructions:
{sample_block}

## Task

Write exactly {max_skills} reusable recommendation skills. Focus on procedures \
that help an agent infer user taste from conversation history, avoid repeating \
movies already mentioned in the context, rank multiple candidate movies, and \
emit valid recommendation JSON.

Each skill must be a complete SKILL.md file using this exact structure:

---
name: skill_name_snake_case
description: 1-2 sentences stating WHEN to use this skill.
---

# Skill Name

## Overview

Briefly state what recommendation failure this skill prevents.

## Workflow

1. Identify explicit liked/disliked movies and user constraints.
2. Convert those into recommendation criteria.
3. Rank candidates by fit, novelty relative to context, and specificity.
4. Return valid JSON with a ranked "recommendations" array and do not include
   titles already mentioned by the user.

## Example

**Input:** A short movie request.

**Solution:** Show how the skill narrows taste and selects titles.

**Output:** {"recommendations":[{"title":"Movie Title","imdb_id":null}]}

## Common pitfalls

- Recommending a movie already present in the context.
- Ignoring exclusions such as no horror, no foreign films, or date ranges.

Separate each skill with a line containing only: ===SKILL_SEPARATOR===
Return the skills directly. No JSON and no prose outside the SKILL.md files.
"""


CONVREC_TD_PROMPT = """\
You are extracting reusable skills from Reddit-V2 movie recommendation traces.

## Probe traces

{traces_block}

## Task

Write exactly {max_skills} skills that generalize successful recommendation \
behaviors and repair failed behaviors. Focus on: preference extraction, context \
novelty, multiple-gold recommendation ranking, and clean JSON output.

Use complete SKILL.md files with frontmatter:

---
name: skill_name_snake_case
description: 1-2 sentences stating WHEN to use this skill.
---

# Skill Name

## Overview

## Workflow

## Example

## Common pitfalls

Separate each skill with: ===SKILL_SEPARATOR===
Return skills directly. No JSON.
"""


AVOID_CONTEXT_REPEATS = SkillSpec(
    skill_id="sd_avoid_context_repeats",
    name="avoid_context_repeats",
    description=(
        "Use when a movie recommendation request mentions examples, prior "
        "favorites, or already-suggested titles; recommend novel titles instead."
    ),
    origin=SkillOrigin.SPEC_DERIVED,
    level=2,
    tool_calls=[],
    template="""\
# Avoid Context Repeats

## Overview

Reddit-V2 contexts often include movies the user already likes, dislikes, or has
already been recommended. Repeating those titles usually hurts recommendation
quality even when the movie matches the theme.

## Workflow

1. List every movie title mentioned in the user and system turns.
2. Treat those titles as context, not as recommendations to output.
3. Infer the taste signal from those examples: genre, tone, era, pacing,
   language, intensity, and exclusions.
4. Recommend different movies that satisfy the inferred criteria.
5. Return valid JSON with recommendations ranked from strongest fit to weakest.

## Example

**Input:** User likes "Good Will Hunting" and wants feel-good movies.

**Solution:** Use "Good Will Hunting" as a taste anchor, but do not output it.

**Output:** {"recommendations":[{"title":"Singin' in the Rain","imdb_id":null},{"title":"Legally Blonde","imdb_id":null}]}

## Common pitfalls

- Copying a liked example as the recommendation.
- Ignoring a user's explicit exclusion such as no horror or no foreign films.
""",
)


def build_recipe() -> SkillRecipe:
    return SkillRecipe(
        sd_prompt=CONVREC_SD_PROMPT,
        td_prompt=CONVREC_TD_PROMPT,
        seed_skills=[AVOID_CONTEXT_REPEATS],
        sample_task_count=4,
    )
