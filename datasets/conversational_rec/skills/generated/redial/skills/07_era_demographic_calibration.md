---
name: era-demographic-calibration
description: Calibrate recommendations to the era and demographic signals present in the conversation instead of defaulting to recent mainstream releases regardless of context.
type: reasoning-skill
dataset: redial
---

# Skill: Era & Demographic Calibration

## Problem
Models default to recommending recent (2010s-2020s) movies regardless of conversation context. When a user talks about films from a particular era, they often want more from that era. Similarly, demographic signals (family, date, teens) are frequently ignored, leading to inappropriate recommendations.

## When to Apply
Whenever the conversation contains:
- Movies predominantly from a specific era
- Demographic signals (kids, teens, date night, family)
- Era-specific language ("classic", "old school", "retro", "like they used to make")

## Instructions

### Step 1: Identify the Era Signal
Map every mentioned movie to its release decade and compute the concentration:

| Decade Concentration | Interpretation |
|---------------------|---------------|
| 70%+ in one decade | Strong era preference — anchor recommendations there |
| 50%+ in one decade | Moderate — weight toward that decade but allow spread |
| Evenly distributed | No era signal — use a balanced distribution |

### Step 2: Match Era in Recommendations
- Strong era signal: at least half your recommendations from that decade (±5 years)
- Moderate: at least a third
- Always include 1-2 from adjacent decades for discovery

### Step 3: Demographic Calibration

| Signal | Adjustment |
|--------|-----------|
| "my kids", "family", "children" | Family-friendly ratings only, age-appropriate themes |
| "date night", "partner" | Romance-adjacent, avoid extreme content |
| "with friends", group context | Higher tolerance for genre films, action, humor |
| "teens", "young adult" | Coming-of-age themes, YA-adjacent content |
| No signal | Default to general audience |

### Step 4: Cultural Context Matching
- If the user references a specific national cinema (Bollywood, anime, K-drama, etc.), weight recommendations toward that space
- If the user uses culturally specific references, match that cultural context
- Don't assume Hollywood-only unless the conversation signals it
