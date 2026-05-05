---
name: sequence-series-patterns
description: Techniques for sequence, series, and recurrence problems in competition mathematics. Use this skill whenever a problem involves arithmetic or geometric progressions, linear recurrences, telescoping sums or products, pattern detection in sequences, nested radicals, continued fractions, or running averages. Also use when a problem defines a sequence by a recurrence relation and asks for the nth term or a sum of terms. Trigger on problems mentioning "a_{n+1} = f(a_n)," "find a_100," "sum of the first n terms," "the sequence 1, 1, 2, 3, 5, ...," or any problem where recognizing a pattern in sequential data is central. If a problem involves an infinite nested structure (like √(a + √(a + √(a + ...))) or continued fractions), this skill applies.
---

# Sequence & Series Patterns

## First move: compute terms explicitly

For any recurrence or pattern-based problem, compute the first 8–10 terms before trying to find a closed form. This is not a shortcut — it's the most reliable way to discover the pattern, and it gives you concrete values to verify any formula against.

Write out the terms in a table or list. Look for:
- **Periodicity:** Do the values repeat? If a_7 = a_1 and a_8 = a_2, the period is 6.
- **Constant differences:** Is a_{n+1} − a_n constant? → Arithmetic progression.
- **Constant ratios:** Is a_{n+1}/a_n constant? → Geometric progression.
- **Recognizable sequences:** Fibonacci, triangular numbers, powers of 2, factorials.
- **Sign patterns:** Does the sequence alternate? Does it stabilize?

## Periodicity detection and index reduction

Many competition recurrences are periodic. Once you've computed enough terms to see the pattern:

1. Identify the period length p (the smallest p such that a_{n+p} = a_n for all n beyond some point)
2. Reduce the target index: a_N = a_{N mod p}, but be careful about the alignment
3. **Index alignment trap:** If your sequence starts at a_1 and has period p, then:
   - If N mod p = 0, the answer is a_p (not a_0, which may not exist)
   - If N mod p = r > 0, the answer is a_r
   - Always verify by checking: does a_{p+1} = a_1? Does a_{p+2} = a_2?

**Example:** a_{n+1} = a_n − a_{n−1} with a_1 = 5, a_2 = 8
- Terms: 5, 8, 3, −5, −8, −3, 5, 8, 3, ...
- Period 6. a_100: 100 mod 6 = 4, so a_100 = a_4 = −5

## Arithmetic and geometric series

**Arithmetic series** (constant difference d):
- nth term: a_n = a_1 + (n−1)d
- Sum of first n terms: S_n = n(a_1 + a_n)/2 = n(2a_1 + (n−1)d)/2
- Number of terms from a_1 to a_n: n = (a_n − a_1)/d + 1

**Geometric series** (constant ratio r):
- nth term: a_n = a_1 × r^(n−1)
- Sum of first n terms: S_n = a_1(r^n − 1)/(r − 1) for r ≠ 1
- Infinite sum (|r| < 1): S = a_1/(1 − r)

When summing an arithmetic series, always verify the number of terms independently. A common competition technique is pairing terms from opposite ends: a_1 + a_n = a_2 + a_{n−1} = ... = constant.

## Telescoping sums

A sum telescopes when each term can be written as the difference of consecutive values of some function:

a_k = f(k) − f(k−1)

Then Σ_{k=1}^{n} a_k = f(n) − f(0).

**How to spot telescoping:**
- Partial fractions: 1/(k(k+1)) = 1/k − 1/(k+1)
- Differences of roots: √(k+1) − √k = 1/(√(k+1) + √k), so the reciprocal telescopes
- Factorial ratios: k × k! = (k+1)! − k!

After telescoping, verify by computing the first 2–3 partial sums manually to confirm they match f(k) − f(0).

## Telescoping products

A product telescopes when each factor can be written as a ratio of consecutive values:

a_k = g(k)/g(k−1)

Then Π_{k=1}^{n} a_k = g(n)/g(0).

**Common in running-average problems:** If S_{n+1} = S_n × (n+1)/n, then:
S_n = S_1 × (2/1) × (3/2) × ... × (n/(n−1)) = S_1 × n

The intermediate factors all cancel, leaving a simple closed form.

## Nested radicals and continued fractions

These infinite structures are solved by self-similarity. The key insight: the infinite structure is equal to a finite piece plus a copy of itself.

**Nested radical:** x = √(a + √(a + √(a + ...)))
- Observe: x = √(a + x)
- Square: x² = a + x → x² − x − a = 0
- Solve: x = (1 + √(1 + 4a))/2 (take positive root since x > 0)

**Continued fraction:** y = a + 1/(a + 1/(a + ...))
- Observe: y = a + 1/y
- Rearrange: y² − ay − 1 = 0
- Solve: y = (a + √(a² + 4))/2 (take positive root)

After solving, verify by computing the first few levels of nesting numerically and checking convergence toward your answer.

## Sums of standard sequences

Useful formulas that appear frequently:

- **Sum of first n positive integers:** n(n+1)/2
- **Sum of first n squares:** n(n+1)(2n+1)/6
- **Sum of first n cubes:** [n(n+1)/2]² (equals the square of the sum of first n integers)
- **Sum of first n triangular numbers:** T_k = k(k+1)/2, and Σ_{k=1}^{n} T_k = n(n+1)(n+2)/6
- **Sum of geometric series:** a(r^n − 1)/(r − 1)

When a sum doesn't match a standard form directly, try decomposing it. For example, Σ k(k+1)/2 = (1/2)(Σ k² + Σ k) = (1/2)(n(n+1)(2n+1)/6 + n(n+1)/2).

## Pairing terms

When two sums have the same number of terms and a term-by-term relationship, pair them before summing separately.

**Example:** (2+4+6+...+2022) − (1+3+5+...+2021)
- Pair: (2−1) + (4−3) + (6−5) + ... + (2022−2021)
- Each pair equals 1, and there are 1011 pairs
- Answer: 1011

This is faster and less error-prone than computing each sum separately and subtracting. Look for pairing opportunities whenever you see a difference of two structured sums.

## Fixed-point behavior

Some recurrences converge to a fixed point where a_n = a_{n+1} = L:
- Set L = f(L) where f is the recurrence function
- Solve for L
- The sequence converges to L if |f'(L)| < 1 (for differentiable f)

Running averages often exhibit this: once the average stabilizes, all subsequent terms equal the average. If the average after k terms is c, and a_{n+1} is defined to maintain the running average, then a_n = c for all n ≥ k.

## Common traps

- **Off-by-one in period reduction.** When N mod p = 0, the answer is a_p, not a_0. Always verify the alignment.
- **Assuming a pattern continues without proof.** Computing 5 terms that follow a pattern doesn't prove the pattern holds forever. For competition purposes, compute enough terms (8–10) to be confident, then prove it via the recurrence or induction if the problem requires justification.
- **Negative roots from squaring.** When solving x = √(a + x), squaring gives two solutions. The negative root is always extraneous — reject it.
- **Misidentifying the starting index.** Some sequences start at a_0, others at a_1. Mismatching the index changes the period-reduction formula.
- **Forgetting the convergence condition** for infinite sums/products. A geometric series only converges for |r| < 1. An infinite nested radical only converges if a ≥ 0.

## Verification

1. Check the first 3–4 terms of your closed form against the recurrence
2. For periodic sequences, verify the period by checking a_{p+1} = a_1, a_{p+2} = a_2
3. For sums, verify with n = 1, 2, 3
4. For nested structures, compute 3–4 levels of nesting numerically and check convergence
