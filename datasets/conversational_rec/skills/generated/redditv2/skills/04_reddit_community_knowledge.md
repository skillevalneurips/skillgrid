---
name: reddit-community-knowledge
description: Understand what experienced film enthusiasts on Reddit would recommend versus casual viewers. Model the community's anti-obvious bias, hidden-gem premium, and specificity culture.
type: reasoning-skill
dataset: reddit_v2
---

# Skill: Reddit Community Knowledge

## Problem
Gold answers come from actual Reddit threads where community members upvoted recommendations. Reddit film communities have distinct recommendation patterns — an anti-obvious bias, a premium on hidden gems, and a culture of specificity — that differ significantly from generic "top movies" lists.

## Instructions

### Step 1: Understand Reddit Film Community Patterns

1. **Anti-obvious bias**: The community rarely upvotes the single most obvious recommendation for a given request. If the answer is too easy, Redditors assume the poster has already considered it.

2. **"Hidden gem" premium**: Recommendations that introduce the poster to films they likely haven't encountered get more engagement than safe, universally known picks.

3. **Specificity rewarded**: A recommendation that matches the EXACT request criteria gets upvoted over a generically good movie in the right genre. Precision matters more than prestige.

4. **Recency awareness**: The community tends to favor films from the last 10-15 years, unless the request specifically targets older cinema.

5. **Director/auteur awareness**: Redditors value recommendations from notable directors — including lesser-known works by well-known directors.

6. **International cinema enthusiasm**: Foreign films get upvoted when they genuinely match the request. The community appreciates breadth beyond Hollywood.

### Step 2: Model the Upvote Dynamic
Ask yourself: "If I posted this recommendation as a reply in the Reddit thread, would it get upvotes?"

Upvote-worthy characteristics:
- Specific match to the request + not the most obvious choice
- Introduces the poster to something they likely haven't seen
- Shows you actually read and understood their post

Downvote-worthy characteristics:
- Too obvious / something everyone has seen
- Generically good but not relevant to the specific request
- Doesn't engage with the nuance of what they asked for

### Step 3: The "Reddit Reply" Test
Before finalizing your list, imagine posting each recommendation as a reply. Would it add value that other replies haven't? Would it demonstrate that you carefully read the post?
