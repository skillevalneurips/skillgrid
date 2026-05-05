---
name: combinatorial-counting
description: Techniques for combinatorial counting and bijection problems in competition mathematics. Use this skill whenever a problem asks "how many ways," "how many arrangements," "in how many configurations," or involves counting objects subject to constraints. Also use when a problem involves selections, distributions, permutations, derangements, partitions, or any question where the answer is a count of discrete objects. Trigger on problems mentioning committees, teams, paths on grids, coloring, seating arrangements, or distributing items into bins. If a problem can be restated as "count the size of a set satisfying conditions," this skill applies.
---

# Combinatorial Counting & Bijections

## First steps with any counting problem

Before choosing a technique, do three things:

1. **Identify exactly what you're counting.** Write it out: "I'm counting the number of [specific objects] satisfying [specific conditions]." Ambiguity here leads to wrong answers.
2. **Check for symmetry or ordering constraints.** Are you counting ordered or unordered selections? If the problem doesn't specify, look for context clues — "committees" are unordered, "arrangements" are ordered.
3. **Estimate the answer's magnitude.** A quick upper bound (like C(n,k) without constraints) tells you whether your final answer is plausible.

## Complementary counting (total minus invalid)

When the valid configurations are hard to count directly but the invalid ones form a small, structured set, count the complement.

**Pattern:** Answer = Total − Invalid

This works especially well when "invalid" means some degeneracy condition. For example, counting triangles from n points: Total = C(n,3), then subtract collinear triples. The key is to enumerate all sources of invalidity exhaustively — check rows, columns, diagonals, and any other lines through 3+ points.

When using this approach, pause after listing invalid cases and ask: "Have I found all of them?" Geometric problems especially tend to have diagonal or anti-diagonal collinear sets that are easy to miss.

## Pigeonhole principle

When a problem asks for the minimum number of selections to guarantee a property, think pigeonhole.

**Formula:** To guarantee at least k items in some pigeonhole, you need (number of pigeonholes) × (k − 1) + 1 selections.

The hard part is usually identifying the pigeonholes correctly. The pigeonholes are the categories, and you want to force at least k items into one category. Count the categories carefully — they might be combinations of properties (e.g., "at most 2 interest groups from 4" gives C(4,0)+C(4,1)+C(4,2) = 11 categories, not 4).

## Stars and bars

For distributing n identical items into k distinct bins (allowing empty bins):

C(n + k − 1, k − 1)

For distributing with each bin having at least 1 item: substitute n' = n − k, giving C(n − 1, k − 1).

When items are not identical, you cannot use stars and bars — you need product rules or generating functions instead.

## Inclusion-exclusion

When you need to count elements in a union of sets where the sets overlap:

|A₁ ∪ A₂ ∪ ... ∪ Aₙ| = Σ|Aᵢ| − Σ|Aᵢ ∩ Aⱼ| + Σ|Aᵢ ∩ Aⱼ ∩ Aₖ| − ...

Use this when direct counting would require tracking which constraints are satisfied simultaneously. Common applications: counting surjections, derangements, numbers not divisible by any of a set of primes.

The sign alternates: add singles, subtract pairs, add triples, etc. For k constraints, you need 2^k terms in the worst case, so this is practical only when k is small.

## Reformulation as a known structure

Many counting problems become standard once you recognize the underlying structure:

- **Grid paths:** Count lattice paths from (0,0) to (m,n) using only right/up steps → C(m+n, m). With forbidden regions, use reflection or Lindström-Gessel-Viennot.
- **Graph coloring:** Assign labels to vertices with adjacency constraints → chromatic polynomial.
- **Extremal graph theory:** "Maximum edges with no triangle" → Turán's theorem. "No K₄-subgraph" → Turán number ex(n,K₄).
- **Dominating/independent sets:** "Choose vertices so every vertex is chosen or adjacent to a chosen one."

When you spot such a reformulation, state it explicitly before applying the result — this makes the solution verifiable.

## Avoiding overcounting

The most common source of errors in counting is overcounting identical configurations. Guard against this by:

1. **Imposing a canonical ordering.** When counting unordered selections of k items, enforce a₁ ≤ a₂ ≤ ... ≤ aₖ (or strict inequality if items are distinct). This prevents counting the same set multiple times.
2. **Dividing by symmetry factor.** If each configuration appears exactly s! times due to symmetry, divide by s!. But be careful — this only works when every configuration has the same symmetry, which fails for multisets with repeated elements.
3. **Checking with small cases.** Compute the answer for n = 3 or n = 4 by hand-listing all configurations, then verify your formula agrees.

## Circular vs. linear arrangements

Circular arrangements have fewer symmetries than you might think. For n distinct items in a circle:
- If rotations are equivalent: (n−1)! arrangements
- If rotations and reflections are equivalent: (n−1)!/2 arrangements
- If the circle has labeled positions: n! arrangements (same as linear)

State which convention the problem uses before counting. If a "garland" or "necklace" is mentioned, rotations (and possibly reflections) are usually equivalent.

## Verification

After obtaining a count:
1. Check small cases by exhaustive listing
2. Verify the answer has the right order of magnitude
3. If the answer is a binomial coefficient, check that it equals the expected value for boundary cases (n=0, k=0, k=n)
