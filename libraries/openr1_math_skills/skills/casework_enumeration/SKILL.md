---
name: casework-enumeration
description: Techniques for systematic casework and enumeration in competition mathematics. Use this skill whenever a problem requires splitting into cases based on parity, residue classes, sign, size, or structural type. Also use when a problem needs exhaustive enumeration of possibilities subject to constraints — such as counting integer solutions to inequalities, finding all triangles with integer sides and given perimeter, or checking which configurations satisfy multiple simultaneous conditions. Trigger when you see problems that say "find all," "how many integers satisfy," "list all solutions," or any problem where the solution requires methodically checking multiple scenarios. If a problem has natural case splits (even/odd, positive/negative, different residue classes) or requires generating all possibilities and filtering, this skill applies.
---

# Casework & Systematic Enumeration

## When to use casework

Casework is the right approach when:
- A formula changes form depending on a discrete parameter (even vs. odd, sign, residue class)
- The problem asks "find all" and the solution set is finite
- Direct algebraic manipulation produces expressions that depend on case-specific conditions
- The problem domain naturally partitions into non-overlapping regions

Casework is the wrong approach when a uniform formula exists — so before splitting into cases, spend a moment looking for one. But when no uniform approach is apparent after brief consideration, commit to casework rather than spending too long searching for elegance.

## Setting up cases correctly

The two requirements for a valid case split are:

1. **Exhaustive:** Every possible input falls into at least one case. Missing a case means missing solutions.
2. **Mutually exclusive:** No input falls into two cases. Overlapping cases cause double-counting.

State your cases explicitly before solving any of them: "I'll split into cases based on [variable]'s parity: Case 1 (even), Case 2 (odd)." This makes it easy to verify exhaustiveness.

Common case-splitting variables:
- **Parity** (even/odd) — useful when squared terms or modular conditions are involved
- **Residue class** (mod 3, mod 4, etc.) — when the problem has modular constraints
- **Sign** (positive/negative/zero) — when absolute values or inequalities appear
- **Size ordering** (a ≤ b ≤ c) — when counting unordered tuples
- **Structural type** (which equation branch applies, which geometric configuration)

## Canonical ordering to avoid overcounting

When counting unordered sets {a, b, c}, impose a canonical ordering like a ≤ b ≤ c. This converts an unordered problem into an ordered one with constraints, preventing permutations of the same set from being counted multiple times.

**Example: Integer triangles with perimeter N**
- Impose a ≤ b ≤ c, so a + b + c = N with 1 ≤ a ≤ b ≤ c
- Triangle inequality reduces to a + b > c (the other two are automatic when a ≤ b ≤ c)
- Range for c: ⌈N/3⌉ ≤ c ≤ ⌊(N−1)/2⌋ (since c ≥ N/3 from being the largest, and c < N/2 from triangle inequality)
- For each c, count valid (a,b) pairs: a ranges from max(1, N − 2c + 1) to ⌊(N−c)/2⌋

Check the boundaries explicitly — off-by-one errors are the most common mistake in enumeration problems.

## Fast elimination of impossible cases

Before solving a case in detail, check if it can yield valid solutions at all:

1. **Integrality check:** If the case leads to n = 101/3, and n must be an integer, skip immediately.
2. **Parity check:** If the case requires an even number to equal an odd expression, skip.
3. **Positivity/range check:** If a variable must be positive but the case forces it negative, skip.
4. **Divisibility check:** If the case requires 7 | (3k + 2), check if any k satisfies this modular equation.

This saves significant time — many cases in competition problems are designed to be quickly eliminable, with only one or two yielding actual solutions.

## Structured enumeration

When you need to list all solutions:

1. **Fix the most constrained variable first.** If c is bounded to a small range, iterate over c values.
2. **Derive bounds on remaining variables** from the constraint and the fixed variable.
3. **Count or list valid combinations** for each value of the fixed variable.
4. **Sum across all values** of the fixed variable.

Keep a running tally. At the end, cross-check the total against an independent estimate (a rough upper bound, or the total without constraints minus eliminated cases).

## Parity-based case splits

Many competition problems have formulas that differ for even and odd values:

- Sum of first n odd numbers: n² (always a perfect square)
- Sum of first n even numbers: n(n+1) (always an oblong number)
- ⌊n/2⌋ behaves differently for even and odd n
- (−1)ⁿ flips sign

When a formula involves ⌊⌋, (−1)ⁿ, or similar parity-sensitive operations, split immediately rather than trying to handle both cases in one expression.

## Residue class enumeration

When a problem involves conditions mod m, partition all candidates into residue classes 0, 1, ..., m−1 and analyze each:

1. For each residue class r, substitute the representative (e.g., n = mk + r) into the constraints
2. Simplify — the modular condition becomes automatic, and what remains is a bound on k
3. Count valid values of k in each class
4. Sum across classes

This is especially useful for problems like "how many integers in [1, 1000] satisfy n² ≡ 1 (mod 7)?" — factor as (n−1)(n+1) ≡ 0 (mod 7), find valid residues, then count how many times each appears in the range.

## Sign-based case splits

When expressions involve absolute values or when you need to determine the sign of a product/sum:

1. Identify the expressions whose signs matter
2. List all sign combinations: (+,+), (+,−), (−,+), (−,−)
3. For each combination, remove the absolute values (replacing |x| with x or −x as appropriate) and solve
4. Check that each solution actually has the assumed signs — reject solutions that violate the sign assumption

## Aggregation and cross-checking

After completing all cases:

1. **List the results from each case** side by side
2. **Sum the counts** (for counting problems) or **collect the solution set** (for "find all" problems)
3. **Cross-check** against:
   - A simpler bound (e.g., total without constraints)
   - A small-case computation (compute the answer for a smaller parameter by brute force)
   - An independent method (if available)

## Common traps

- **Missing a case.** Before solving, verify your cases are exhaustive. If splitting on n mod 3, you need cases 0, 1, and 2 — not just "divisible by 3" and "not divisible by 3."
- **Off-by-one in range bounds.** When c ranges from ⌈N/3⌉ to ⌊(N−1)/2⌋, be precise about ceiling vs. floor, and strict vs. non-strict inequalities. Verify boundary values by substitution.
- **Counting permutations of the same solution.** If you forgot to impose canonical ordering, you might count {1,2,3} and {3,2,1} as different. Divide by k! only if every solution has exactly k! permutations (fails for solutions with repeated elements like {2,2,3}).
- **Forgetting to handle equality cases in ordering.** When imposing a ≤ b ≤ c, the case a = b = c is valid and must not be excluded.
