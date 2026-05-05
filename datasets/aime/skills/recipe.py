"""AIME-specific skill-creation recipe.

Tailors SD/TD prompts for competition-level math (AIME). Problems require
multi-step algebraic reasoning, number theory, combinatorics, and geometry
— far harder than GSM8K.
"""

from __future__ import annotations

from skilleval.core.types import SkillOrigin, SkillSpec
from skilleval.skills.creation.recipe import SkillRecipe


AIME_SD_PROMPT = """\
You are an expert mathematician designing a skill library for an AI agent \
that solves AIME (American Invitational Mathematics Examination) problems.

## Dataset context

- Domain: {domain}
- AIME answers are always integers in [0, 999].
- Problems span algebra, number theory, combinatorics, geometry, and probability.
- Sample problems:
{sample_block}

## Your task

Write exactly {max_skills} **professional mathematical skills** as SKILL.md files.

**Professional skills must:**
- Be written with mathematical rigor and precision
- Include exact formulas, equations, and conditions
- Show concrete worked examples with step-by-step calculations
- Describe specific procedures, not vague advice

## Format for EACH skill

```
---
name: technique_name_snake_case
description: When to apply this technique. Include keywords/problem patterns that signal this skill.
---

# Technique Name In Title Case

## Overview

Brief description of what this mathematical technique accomplishes.

## Workflow

1. Step with formula: \\( formula \\)
2. Next step with specific action
3. Continue with clear, actionable steps...

## Example

**Problem:** [Actual AIME-style problem with numbers]

**Solution:**
1. [Apply step 1 with actual values]
2. [Apply step 2]
...

**Output:** [Integer 0-999]

## Common pitfalls

- Mistake 1 and how to prevent it
- Mistake 2 and how to prevent it
```

Separate each skill with: ===SKILL_SEPARATOR===

Return skills directly. No JSON.
"""


AIME_TD_PROMPT = """\
You are an expert mathematician extracting reusable problem-solving skills \
from AIME (American Invitational Mathematics Examination) agent traces.

## Probe traces

{traces_block}

## Your task

Write exactly {max_skills} **professional mathematical skills** as SKILL.md files \
that generalize techniques visible in the traces.

**Professional skills must:**
- Be written with mathematical rigor and precision
- Include exact formulas, equations, and conditions
- Show concrete worked examples with step-by-step calculations
- Describe specific procedures, not vague advice

## Format for EACH skill

```
---
name: technique_name_snake_case
description: When to apply this technique. Include keywords/problem patterns.
---

# Technique Name In Title Case

## Overview

Brief description based on patterns from the traces.

## Workflow

1. Step with formula or method
2. Next step
3. Continue...

## Example

**Problem:** [From traces or similar AIME problem]

**Solution:**
1. [Step-by-step solution]
...

**Output:** [Integer 0-999]

## Common pitfalls

- Mistakes observed in traces
- How to avoid them
```

Separate each skill with: ===SKILL_SEPARATOR===

Return skills directly. No JSON.
"""


MODULAR_ARITHMETIC_SEED = SkillSpec(
    skill_id="sd_modular_arithmetic",
    name="modular_arithmetic",
    description=(
        "Apply modular arithmetic to simplify computations involving "
        "remainders, divisibility, or cyclic patterns."
    ),
    origin=SkillOrigin.SPEC_DERIVED,
    level=2,
    tool_calls=[],
    template="""\
## When to use
When a problem asks for a remainder, involves divisibility conditions, \
or has expressions that simplify modulo a small number. Also useful \
for checking answers since AIME answers are integers in [0, 999].

## Procedure
1. Identify the modulus from the problem (explicit remainder, or choose \
a convenient one like 10, 100, or a prime factor).
2. Reduce all large expressions modulo that value before combining.
3. Apply properties: (a+b) mod m = ((a mod m)+(b mod m)) mod m; \
same for multiplication.
4. If the problem asks "find the remainder when divided by 1000", \
compute everything mod 1000 from the start.
5. Verify the final integer is in [0, 999].

## Worked example
Problem: "Find the remainder when 2^100 is divided by 7."
2^1=2, 2^2=4, 2^3=1 (mod 7). Cycle length 3.
100 = 33*3 + 1, so 2^100 = 2^1 = 2 (mod 7). Answer: 2.

## Common pitfalls
- Forgetting to reduce intermediate products, causing overflow in reasoning.
- Assuming a cycle length without verifying the full cycle.
- Off-by-one in exponent decomposition.
""",
)


def build_recipe() -> SkillRecipe:
    return SkillRecipe(
        sd_prompt=AIME_SD_PROMPT,
        td_prompt=AIME_TD_PROMPT,
        seed_skills=[MODULAR_ARITHMETIC_SEED],
        sample_task_count=3,
    )
