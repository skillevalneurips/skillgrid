---
name: cross-genre-lateral-thinking
description: Match structural and narrative patterns across genres rather than staying rigidly within one genre. Correct answers sometimes come from unexpected genres that share a deeper structural similarity.
type: reasoning-skill
dataset: reddit_v2
---

# Skill: Cross-Genre Lateral Thinking

## Problem
Models stay rigidly within the genre they identify — if the user says "thriller", all 10 recommendations are thrillers. But correct answers sometimes come from adjacent or unexpected genres that share a narrative structure, emotional arc, or thematic core with the request.

## When to Apply
- When the user's request focuses on a structural/thematic element rather than a specific genre
- When the user lists reference movies that span multiple genres
- When the request uses non-genre language ("movies that make you think", "movies with great twists", "movies about obsession")

## Instructions

### Step 1: Identify the Structural Core
Look past genre to find what the user is REALLY asking for. Separate the surface genre from the deeper structural pattern:

- "Betrayal movies" → The structural core is "elaborate deception with twists", which can appear in thrillers, heist films, comedies, and sci-fi
- "Courtroom dramas" → The structural core is "argument, confrontation, truth-seeking", which can appear in political dramas, journalism films, and debate movies
- "Mind-bending movies" → The structural core is "cognitive challenge, reality questioning", which spans horror, sci-fi, thriller, and animation
- "Movies about isolation" → Solitude and confinement appear across space films, survival dramas, horror, and relationship dramas

### Step 2: Generate Cross-Genre Candidates
After your primary genre-matched list, explicitly ask: "What movies from OTHER genres share this structural DNA?"

**Attribute transplant technique:**
1. Extract the 2-3 key narrative attributes from the request
2. Search for those attributes across every genre
3. Add 2-3 cross-genre matches to your list

### Step 3: Validate the Connection
A cross-genre pick must satisfy: "If someone loved the reference movies, they would enjoy this recommendation for the same structural reason, even though it's a different genre."

If you can't articulate the structural connection in one sentence, drop the pick.

### The 70/30 Rule
- ~70% of your recommendations should be from the expected genre
- ~30% should be deliberate cross-genre picks that match the structural core
- This 30% often contains surprising but correct answers
