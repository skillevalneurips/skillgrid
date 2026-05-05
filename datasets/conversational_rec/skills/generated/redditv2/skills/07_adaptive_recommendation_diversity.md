---
name: adaptive-recommendation-diversity
description: Ensure recommendation lists include deliberate diversity across subgenres, eras, and interpretive angles rather than clustering all picks in one narrow pocket.
type: reasoning-skill
dataset: reddit_v2
---

# Skill: Adaptive Recommendation Diversity

## Problem
Models cluster all recommendations in one narrow pocket — all from the same subgenre, era, and tone. When the correct answer sits even slightly outside that pocket, it's missed. Diversifying the list across multiple valid interpretations of the request substantially improves coverage.

## Instructions

### Step 1: Generate Multiple Interpretations
Before building your list, generate 2-3 different valid interpretations of the request. Most requests can be read multiple ways — a "betrayal movie" could mean dark psychological thrillers, entertaining heist films, or paranoid conspiracy thrillers.

### Step 2: Allocate Slots Across Interpretations

| Interpretation | Slot Allocation | Rationale |
|---------------|----------------|-----------|
| Primary (most likely) | ~50% | Best-fit interpretation of the request |
| Secondary | ~30% | Plausible alternative reading |
| Tertiary / Wild card | ~20% | Unexpected angle that still genuinely matches |

### Step 3: Diversify Within Each Interpretation
Even within a single interpretation, vary along these axes:
- **Popularity**: Mix well-known and lesser-known titles
- **Era**: Span at least 2 decades
- **Origin**: Include non-Hollywood titles when relevant
- **Tone gradient**: Include slightly lighter and darker variations

### Step 4: The Coverage Test
After building your list, ask: "If I grouped these recommendations into clusters, are there at least 2-3 distinct clusters?"

Good list: Multiple distinct clusters with bridging titles
Bad list: All titles in one tight cluster

### Calibrate Diversity to Request Type

| Request Type | Diversity Level | Reasoning |
|-------------|----------------|-----------|
| "Movies like X" (single reference) | HIGH | Many valid aspects to match |
| Genre request ("horror movies") | MEDIUM | Genre is clear but subgenre is open |
| Specific criteria ("one-room movies") | LOW | Tight constraint — focus on matching it |
| Mood request ("feel-good") | HIGH | Mood is subjective — cast a wide net |
| List-based ("more like these 10") | MEDIUM | Intersection narrows space but multiple angles exist |
