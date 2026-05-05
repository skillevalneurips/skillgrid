---
name: genre-granularity-subgenre-mapping
description: Distinguish between subgenres within broad categories like comedy, horror, romance, and drama instead of treating each genre as a monolithic bucket.
type: reasoning-skill
dataset: redial
---

# Skill: Genre Granularity & Subgenre Mapping

## Problem
The model collapses rich genre signals into broad categories. "Comedy" becomes one bucket, producing recommendations that span wildly different subgenres — rom-coms mixed with satire mixed with slapstick. Users who want a specific subgenre get generic cross-subgenre results that miss the mark.

## Instructions

### Step 1: Identify the Subgenre from Signals
Don't stop at the broad genre. Use conversation signals to pinpoint the subgenre. Key signal types:

- **Adjectives**: "raunchy" vs "witty" vs "silly" vs "dark" all point to completely different comedy subgenres
- **Referenced movies**: The specific titles a user mentions are the strongest subgenre signal. Map them to their subgenre and stay in that lane.
- **Actor preferences**: Certain actors are strongly associated with specific subgenres. Use this as a signal.
- **Mood language**: "Something that'll make me cry" vs "something lighthearted" vs "something twisted" — these refine genre into subgenre.

### Step 2: Common Subgenre Splits to Watch For

**Comedy**: Raunchy/adult, smart/dry, romantic, dark/satire, slapstick, family, parody, stoner, cringe
**Horror**: Mainstream, slow-burn/atmospheric, slasher/gore, psychological, horror-comedy, found footage, supernatural, body horror
**Romance**: Fantasy romance, classic/period, tragic, indie/quirky, rom-com, erotic thriller
**Action**: Martial arts, heist, spy/espionage, war, superhero, survival, revenge
**Drama**: Legal, medical, sports, biopic, family, social/political, coming-of-age

### Step 3: Recommend Within the Correct Subgenre
Your top 5 recommendations should be from the identified subgenre. Only slots 6-10 should venture into adjacent subgenres for variety.

### Step 4: Never Cross Major Subgenre Boundaries for Top Picks
A rom-com and a dark satire serve completely different audiences despite both being "comedies". A slasher and a psychological horror film appeal to different tastes despite both being "horror". Treat subgenre boundaries as meaningful.
