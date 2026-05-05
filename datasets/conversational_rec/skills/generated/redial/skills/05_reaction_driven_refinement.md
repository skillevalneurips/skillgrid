---
name: reaction-driven-refinement
description: Leverage the user's reactions to previously suggested movies in the conversation — enthusiasm, lukewarm responses, dismissals — as strong signals to refine what to recommend next.
type: reasoning-skill
dataset: redial
---

# Skill: Reaction-Driven Refinement

## Problem
In conversational recommendation, the system often suggests movies and the user reacts — with enthusiasm, indifference, or rejection. These reactions are among the strongest preference signals available, yet models typically treat the entire conversation as a flat context and fail to weight reaction signals appropriately.

## When to Apply
Any conversation where the system has already made one or more suggestions and the user has responded to them.

## Instructions

### Step 1: Identify Reaction Signals
Scan the conversation for how the user responded to each previously mentioned or suggested movie:

| Reaction Type | Signal Words / Patterns | Implication |
|--------------|------------------------|-------------|
| **Strong positive** | "I love that one!", "one of my favorites", "great movie" | Recommend more like THIS specific title |
| **Mild positive** | "yeah that's good", "I've seen it", "not bad" | Noted but not a strong anchor — don't over-index |
| **Neutral/indifferent** | No reaction, topic change, "hmm" | This direction may not resonate — try a different angle |
| **Negative** | "didn't like it", "not really my thing", "eh" | Avoid this subgenre/tone/style going forward |
| **Conditional** | "I liked it but...", "it was okay except for..." | The qualifier tells you exactly what to adjust |

### Step 2: Build a Reaction Map
For every movie that received a reaction, note:
- What the movie is (genre, subgenre, tone, era)
- What the reaction was (positive / negative / conditional)
- What aspect the reaction targeted (if identifiable)

### Step 3: Use Reactions as Stronger Signals Than Stated Preferences
A user who says "I like comedies" gives you a broad signal. A user who says "I LOVE that movie!" about a specific comedy gives you a precise signal. The specific reaction always outweighs the general statement.

Priority order for recommendation signals:
1. Strong positive reactions to specific titles (highest weight)
2. Negative reactions / rejections (hard constraints)
3. Conditional reactions (refinement signals)
4. Explicitly stated preferences
5. Inferred preferences from conversation context

### Step 4: Adjust in Real Time
Each new reaction should shift your recommendation strategy:
- After a strong positive: Narrow toward movies similar to the positively received one
- After a rejection: Pivot away from that style/subgenre
- After a conditional ("liked it but too long"): Keep the core but adjust the flagged attribute
- After indifference: Broaden your approach — try a different angle entirely
