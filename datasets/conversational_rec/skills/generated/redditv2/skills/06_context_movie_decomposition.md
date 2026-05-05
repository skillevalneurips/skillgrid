---
name: context-movie-decomposition
description: When the user lists multiple reference movies, compute the attribute intersection rather than the union. Find what ALL references share to identify the core request.
type: reasoning-skill
dataset: reddit_v2
---

# Skill: Context Movie Decomposition & Intersection

## Problem
When users list multiple reference movies, models tend to either focus on the first/most prominent movie and ignore the rest, or treat each independently and recommend a scattered union. The correct approach is to find the INTERSECTION — what all references share — and recommend from that narrow overlap.

## When to Apply
Any request that mentions 3 or more reference movies or shows.

## Instructions

### Step 1: Decompose Each Reference Movie
For each movie the user mentions, list its key attributes across these dimensions:

| Dimension | What to Assess |
|-----------|---------------|
| Genre / Subgenre | Primary and secondary genre classifications |
| Tone | Dark, light, ironic, earnest, campy, gritty |
| Era / Setting | Time period of the film and its setting |
| Pacing | Slow burn, fast-paced, episodic |
| Themes | Core thematic concerns |
| Character type | Ensemble, lone hero, antihero, underdog, young protagonists |
| Visual style | Stylized, realistic, animated, found-footage |

### Step 2: Compute the Intersection
Lay the attributes side by side and find what ALL references share. The intersection is the set of attributes common to every listed movie — this defines the user's core request.

Be especially careful about attributes that appear in MOST but not ALL references. If one reference has a horror element but the other three don't, horror is NOT in the intersection.

### Step 3: Recommend from the Intersection
Every recommendation must satisfy the intersection criteria. A movie matching most intersection attributes but violating one is a weaker pick than one matching all of them.

### Step 4: Handle Outlier References
Sometimes one reference doesn't fit the pattern. If 3+ movies cluster together and 1 doesn't, weight the cluster. The outlier may be a stray mention or a secondary preference.

### The Intersection Hierarchy
When the intersection is too narrow (few or no movies match all criteria), relax in this order:
1. Keep: Theme + Tone (most important — these define the viewing experience)
2. Relax: Setting, Era (less critical)
3. Last to relax: Genre (only if absolutely necessary)
