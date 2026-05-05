---
name: deep-catalog-longtail
description: Retrieve lesser-known, niche, and long-tail movie titles calibrated to the user's knowledge level, rather than defaulting to the same safe mainstream picks.
type: reasoning-skill
dataset: reddit_v2
---

# Skill: Deep Catalog & Long-Tail Retrieval

## Problem
Models default to the same well-known titles regardless of request specificity. Reddit gold answers tend to span a huge range of obscurity levels, with most correct titles appearing only once — this is fundamentally a long-tail recall problem that requires deep catalog knowledge.

## Instructions

### Step 1: Assess the User's Knowledge Level
Reddit posters on movie recommendation subreddits are typically film enthusiasts who have already seen the obvious picks.

| Signal | Knowledge Level | Implication |
|--------|----------------|-------------|
| Lists 5+ reference movies | High | They've seen the obvious ones — go deeper |
| References directors by name | High | They know cinema — recommend auteur-level |
| Uses genre jargon ("noir", "giallo", "mumblecore") | Expert | Match their specificity level |
| Simple request with 1 movie | Moderate | Mix popular and deep cuts |
| First-timer language ("beginner to...") | Beginner | Start accessible, include stepping stones |

### Step 2: Match Obscurity Level to User
Your recommendations should be at the SAME obscurity level as the user's listed movies, plus slightly deeper:

- If they list mainstream hits → recommend well-known but not top-10 titles
- If they list well-known genre films → recommend respected but less mainstream titles
- If they list deep cuts → recommend equally niche titles

### Step 3: Avoid Reflexive Defaults
Before including any extremely well-known title, verify it specifically matches the request — don't include it just because it's a safe, popular choice. If you find yourself reaching for the same handful of famous movies across different requests, you're not engaging with the specific query.

### Step 4: Confidence Check
If you're uncertain a movie title is real or correct:
- Do NOT recommend it
- Replace with a title you are confident about
- Never guess at identifiers — leave them null rather than fabricate
