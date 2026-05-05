---
name: recommendation-diversity
description: Ensure recommendation lists include deliberate diversity across subgenres, eras, and popularity tiers rather than clustering all picks in one narrow pocket.
type: reasoning-skill
dataset: redial
---

# Skill: Recommendation Diversity & Strategic Risk-Taking

## Problem
Most recommendation lists are homogeneous — all from the same subgenre, era, and popularity tier. This means if the correct answer sits slightly outside the model's narrow focus, it's missed entirely. Deliberately diverse lists cover more ground.

## Instructions

### The Diversity Budget
Structure every recommendation list with intentional slot allocation:

| Slots | Category | Purpose |
|-------|----------|---------|
| ~40% | **Core matches** | Direct subgenre/tone matches to stated preferences |
| ~30% | **Adjacent exploration** | Same broad genre but different subgenre, OR same tone in a different genre |
| ~20% | **Era diversification** | Titles from a different era than the core picks |
| ~10% | **Wild card** | A structurally or thematically similar movie from an unexpected angle |

### Era Distribution Rules
- If the user discusses movies from a specific era, weight toward that era but don't limit yourself to it
- If no era signal exists, distribute across decades rather than clustering in one
- Never let all recommendations come from the same decade

### Popularity Distribution Rules
- Cap the number of "everyone knows this movie" picks at ~30%
- Include at least a couple of titles that only genre enthusiasts would know
- Match the obscurity level of the movies the user mentioned

### The Self-Check
After building your list, ask: "If the user's actual taste is slightly different from my primary interpretation, would any of my recommendations still hit?" If the answer is no, your list is too narrow.
