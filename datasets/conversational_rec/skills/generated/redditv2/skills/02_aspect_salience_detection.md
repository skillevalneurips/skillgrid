---
name: aspect-salience-detection
description: Identify which specific attribute of reference movies the user cares about most — a setting, narrative structure, emotional arc, or visual style — rather than defaulting to broad genre matching.
type: reasoning-skill
dataset: reddit_v2
---

# Skill: Aspect Salience Detection

## Problem
Reddit posts often highlight a specific aspect of a movie they want more of — a setting, a narrative structure, an emotional arc, a visual style — but models default to matching the broadest genre tag instead of the salient attribute.

## Instructions

### Step 1: Parse the Request for Salient Aspects

| Aspect Type | What to Look For |
|------------|-----------------|
| **Setting** | Geographic location, time period, single-location, environment type |
| **Narrative structure** | Twist endings, nonlinear timelines, unreliable narrators, frame stories |
| **Emotional arc** | Redemption, descent, catharsis, bittersweet endings |
| **Visual/sensory** | Specific phobias (heights, water, claustrophobia), cinematographic style |
| **Character type** | Protagonist archetype (con artist, underdog, antihero, ensemble) |
| **Relationship dynamic** | Enemies-to-friends, mentor-student, betrayal, unlikely partners |
| **Theme** | Obsession, grief, identity, class, corruption |

### Step 2: Rank Aspects by Salience
If multiple aspects are present, rank them:

1. **Title/subject line aspect** — The post title is the strongest signal. A word in the title is the non-negotiable core constraint.
2. **Repeated aspect** — If the user mentions something twice, it's critical.
3. **Qualified aspect** — "specifically the part where..." or "what I really liked was..." marks high salience.
4. **Listed-movie intersection** — What do ALL their reference movies have in common? That shared attribute is the core aspect.

### Step 3: Filter by Salient Aspect First
Instead of the typical approach (match genre → then filter by aspect), reverse the order:

1. Match salient aspect → then filter by genre/tone

This reversal ensures the core attribute the user cares about is preserved in every recommendation, even if it means some picks come from unexpected genres.

### Step 4: Verify Each Recommendation
For every pick, ask: "Does this movie satisfy the salient aspect?" If you can't confidently say yes, replace it — even if it's a great movie in the right genre.
