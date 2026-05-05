---
name: implicit-preference-detection
description: Extract meaningful preferences from sparse or ambiguous conversations where the user provides few explicit signals, avoiding the trap of defaulting to generic blockbusters.
type: reasoning-skill
dataset: redial
---

# Skill: Implicit Preference Detection

## Problem
In sparse-context conversations where the user provides little explicit guidance — sometimes just a single genre mention or a vague request — models default to universally acclaimed classics. These safe picks are almost never what the user is looking for because they're too generic to match any specific taste profile.

## When to Apply
When the conversation has:
- Few turns with minimal detail
- Few or no explicitly mentioned movie titles
- Vague requests like "something good", "a fun movie", "I'm bored"

## Instructions

### Step 1: Mine Every Available Signal
Even sparse conversations contain implicit signals:

- **Adjectives and qualifiers**: Words like "edgy", "feel-good", "quirky", "intense", "chill", "fun" each imply a different taste profile
- **Conversational tone**: Is the user enthusiastic and excitable (likely wants high-energy films) or subdued (may want something contemplative)?
- **What they DON'T say**: If asked about genre and they mention "comedies or rom coms" without action/horror, they may actively avoid those
- **Demographic hints**: References to "my kids", "date night", "with the boys" signal audience context
- **Recency cues**: "I just watched..." or "lately I've been into..." signals current mood, not all-time preferences

### Step 2: Build a Probabilistic Taste Profile
Instead of committing to a single genre, build a weighted profile with primary, secondary, and tertiary genre/subgenre probabilities, along with estimated energy level and era preference.

### Step 3: Diversify Under Uncertainty
When you don't have strong signals, DO NOT converge on one type. Instead:
- Spread your recommendations across 3-4 subgenre pockets
- Include a mix of eras (some recent, some older)
- Include titles at varying popularity levels
- AVOID all-time-greatest-hits lists — they are the least informative recommendations possible
