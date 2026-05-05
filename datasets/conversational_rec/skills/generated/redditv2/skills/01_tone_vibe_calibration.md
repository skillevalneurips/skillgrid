---
name: tone-vibe-calibration
description: Infer the emotional tone, energy level, and vibe the user is seeking — not just the genre. Match the feeling of the reference movies, not just their plot categories.
type: reasoning-skill
dataset: reddit_v2
---

# Skill: Tone & Vibe Calibration

## Problem
Models correctly identify genre keywords but completely miss the tonal quality the user is seeking. Matching genre but not tone is the most common failure mode — for example, recommending dark horror when the user wants fun mystery-adventure, or recommending grim thrillers when the user wants entertaining heist films.

## Instructions

### Step 1: Extract Tone Signals
Identify all tone indicators in the post:

| Signal Type | What to Look For |
|------------|-----------------|
| **Explicit adjectives** | "fun", "dark", "cozy", "intense", "light", "eerie", "heartwarming" |
| **Reference movie tone** | What does each referenced movie FEEL like to watch? Match that feeling. |
| **Energy words** | "exciting", "chill", "wild", "relaxing", "edge-of-seat" |
| **Audience context** | "with my kids", "late night", "feel-good" constrains acceptable tone |
| **Negation signals** | "not too heavy", "nothing depressing" sets hard tone boundaries |

### Step 2: Build a Tone Profile
Create a 3-axis tone profile before generating recommendations:

1. **Lightness**: Dark ←→ Light (heavy/disturbing vs fun/uplifting?)
2. **Energy**: Low ←→ High (contemplative and slow vs fast-paced and thrilling?)
3. **Sincerity**: Earnest ←→ Ironic (played straight vs with wink-and-nod humor?)

### Step 3: Validate Recommendations Against Tone
For EACH recommendation, ask: "Does this movie's tone match the user's tone profile on all three axes?"

A mismatch on even one axis can make a recommendation feel wrong to the user, even if genre and theme match perfectly.

### Step 4: Use Reference Movies as Tone Anchors
The most reliable tone signal is the reference movies themselves. Before recommending, ask: "Would watching my recommendation feel like watching their reference movies?" If the emotional experience is fundamentally different, the pick is wrong regardless of surface-level genre match.
