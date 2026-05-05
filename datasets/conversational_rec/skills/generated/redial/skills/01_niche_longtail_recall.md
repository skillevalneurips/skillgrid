---
name: niche-longtail-recall
description: Retrieve lesser-known and long-tail movie titles instead of defaulting to the same popular blockbusters every time.
type: reasoning-skill
dataset: redial
---

# Skill: Niche & Long-Tail Movie Recall

## Problem
Recommender models over-index on a small set of universally popular movies, recommending the same safe defaults regardless of conversation context. These generic picks rarely match what the user actually wants because they aren't tailored to the specific taste signals in the dialog.

## Instructions

1. **Detect the popularity trap**: After generating an initial candidate list, flag any title that would appear on a "top 100 most popular movies" list. If more than 3 of your 10 recommendations are mainstream blockbusters, you are likely in the popularity trap.

2. **Go deeper in each genre pocket**: When you identify the user's preferred genre or subgenre, don't stop at the most universally known titles in that space. Think about what *specific kind* of film the user wants and find titles 2-3 layers deeper — movies that dedicated fans of that subgenre would know, but casual viewers might not.

3. **Use mentioned movies as specificity anchors**: The obscurity level of the movies the user references should calibrate the obscurity level of your recommendations. If they mention niche titles, recommend at a similar depth. If they mention mainstream titles, you can include some well-known picks but still push slightly deeper.

4. **Allocate slots deliberately**:
   - Slots 1-3: Strong matches at moderate popularity
   - Slots 4-7: Deep cuts that precisely match the stated preferences
   - Slots 8-10: Lateral/surprise picks from adjacent subgenres

5. **Avoid the "greatest hits" reflex**: Never recommend a movie solely because it's widely regarded as great. It must match the specific taste signals in the conversation.
