---
name: conversational-preference-synthesis
description: Track and synthesize evolving user preferences across multi-turn conversations where preferences shift, narrow, or contradict earlier statements.
type: reasoning-skill
dataset: redial
---

# Skill: Conversational Preference Synthesis

## Problem
In multi-turn movie recommendation conversations, user preferences evolve — they narrow their request, change direction, or contradict earlier statements. A fixed reasoning template that treats the conversation as a flat bag of keywords fails to capture this evolution, producing worse results as conversations get longer.

## Instructions

### Phase 1: Preference Timeline Construction
Before reasoning about recommendations, construct an explicit preference timeline:

- **Turn-by-turn extraction**: For each turn, note what the user explicitly said they like, dislike, or are looking for.
- **Recency weighting**: Later turns override earlier ones. If an early turn says "I like action" but a later turn says "actually something more chill", the active preference is "chill".
- **Narrowing vs broadening**: Track whether the user is narrowing their request ("actually, more specifically...") or broadening it ("or maybe something totally different").

### Phase 2: Preference Reconciliation
Identify and resolve conflicts:

- **Explicit contradictions**: Two statements that conflict → the later one wins, but the earlier one may still constrain the space (e.g., "I like horror" + later "not too scary" → mild horror / thriller territory).
- **Implicit evolution**: User discusses one genre at length, then asks for "something different" → they want contrast, not more of the same.
- **Rejection signals**: If the recommender suggested something and the user dismissed it, that entire category may be off the table.

### Phase 3: Final Preference Profile
Synthesize a single coherent preference profile:
- Primary genre/mood the user is seeking RIGHT NOW (not many turns ago)
- Hard constraints (dislikes, already-seen, explicit rejections)
- Soft preferences (actors, eras, tones mentioned approvingly)

Use this profile — not a flat summary of the entire conversation — to drive your recommendations.
