"""GSM8K-specific skill-creation recipe.

Tailors the SD/TD prompts for grade-school math word problems and seeds
the library with one hand-authored verification skill that's reliably
useful across GSM8K sub-patterns.
"""

from __future__ import annotations

from skilleval.core.types import SkillOrigin, SkillSpec
from skilleval.skills.creation.recipe import SkillRecipe


GSM8K_SD_PROMPT = """\
You are designing a skill library for an AI agent that solves grade-school \
math word problems (GSM8K-style).

## Dataset context

- Domain: {domain}
- Available tools:
{tools_block}
- Sample task instructions:
{sample_block}

## Your task

Write exactly {max_skills} reusable skills that help solve these problems. \
Prefer procedures that actually help a weaker backbone (Qwen/Llama) stay \
disciplined: parsing givens, writing equations, computing step-by-step, \
and verifying the final number. Do not propose "use calculator" skills — \
treat arithmetic as CoT. Skills must be distinct and non-overlapping.

Skills describe HOW to reason. Do not prescribe the reply's output format \
— the agent's system prompt owns that contract separately.

Return ONLY a JSON array of exactly {max_skills} skill objects. Each object \
must have these fields:

- "name": short semantic identifier, snake_case
- "description": 1-2 sentences stating WHEN to use this skill
- "tool_calls": list of tool names this skill uses (usually empty for GSM8K)
- "template": multi-section markdown (<= 500 words) with exactly these \
sections in this order:
  ## When to use
  ## Procedure   (numbered steps)
  ## Tool invocation examples   (omit entire section if tool_calls is empty)
  ## Worked example
  ## Common pitfalls

Return JSON only. No markdown fences, no prose around it.
"""


GSM8K_TD_PROMPT = """\
You are extracting reusable skills for grade-school math word problems \
(GSM8K) from agent interaction traces.

## Probe traces

{traces_block}

## Your task

Write exactly {max_skills} skills that generalize patterns visible in the \
traces. Focus on arithmetic decomposition, careful carry of intermediate \
values, and recovery from off-by-one or sign errors. Do not propose \
overlapping skills.

Skills describe HOW to reason. Do not prescribe the reply's output format \
— the agent's system prompt owns that contract separately.

Return ONLY a JSON array of exactly {max_skills} skill objects, same schema \
as the SD version:

- "name" (snake_case), "description" (1-2 sentences)
- "tool_calls" (list, usually empty for GSM8K)
- "template" with sections: ## When to use, ## Procedure, ## Worked example, \
## Common pitfalls.

Return JSON only.
"""


VERIFY_BY_SUBSTITUTION_SEED = SkillSpec(
    skill_id="sd_verify_by_substitution",
    name="verify_by_substitution",
    description=(
        "Sanity-check a candidate numeric answer by plugging it back into "
        "the problem's constraints before committing to it."
    ),
    origin=SkillOrigin.SPEC_DERIVED,
    level=2,
    tool_calls=[],
    template="""\
## When to use
After you have computed a candidate final number for a GSM8K problem, \
use this skill as a last step before committing to the answer. Especially \
valuable for problems with words like "if", "after", "remaining", or \
multi-step conditionals.

## Procedure
1. Restate the question in one sentence and name the quantity asked for.
2. Write the candidate answer next to that quantity.
3. Walk through every numeric constraint in the problem and check the \
candidate is consistent with each.
4. If any constraint is violated, re-derive from the original givens; \
don't adjust the candidate by guessing.

## Worked example
Problem: "Jen had 12 apples. She gave 3 to each of 2 friends, then bought \
5 more. How many does she have?"
Candidate: 11. Check: 12 - (3*2) + 5 = 12 - 6 + 5 = 11. Consistent.

## Common pitfalls
- Committing to the candidate before checking a constraint (off-by-one slips).
- Silently adjusting the candidate when a check fails instead of redoing \
the derivation.
""",
)


def build_recipe() -> SkillRecipe:
    return SkillRecipe(
        sd_prompt=GSM8K_SD_PROMPT,
        td_prompt=GSM8K_TD_PROMPT,
        seed_skills=[VERIFY_BY_SUBSTITUTION_SEED],
        sample_task_count=3,
    )
