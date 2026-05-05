---
name: number-theory-fundamentals
description: Techniques for number theory problems involving GCD, LCM, primes, and Diophantine equations in competition mathematics. Use this skill whenever a problem involves greatest common divisors, least common multiples, prime factorization, coprimality, the Euclidean algorithm, Diophantine equations, or the Frobenius/coin problem. Also use when a problem asks about representability of integers as sums of specific denominations, or when divisibility by multiple numbers simultaneously is a key constraint. Trigger on problems mentioning "relatively prime," "coprime," "greatest common factor," "least common multiple," or any problem where the multiplicative structure of integers is central. If a problem involves factoring large numbers, finding all divisors, or determining whether a number can be expressed in a particular form, this skill applies.
---

# Number Theory Fundamentals: GCD, LCM, and Primes

## Prime factorization is the foundation

Most number theory problems in competitions become straightforward once you have the prime factorization. Make factoring your first move — don't skip it in favor of trial-and-error.

**Systematic factoring procedure:**
1. Divide out small primes (2, 3, 5, 7, 11, 13) in order
2. After removing all small prime factors, check if the remaining number is prime (test divisibility up to its square root)
3. Write the result in canonical form: N = p₁^a₁ × p₂^a₂ × ... × pₖ^aₖ

From the prime factorization, everything follows:
- **Number of divisors:** (a₁+1)(a₂+1)...(aₖ+1)
- **Sum of divisors:** (1+p₁+...+p₁^a₁)(1+p₂+...+p₂^a₂)...(1+pₖ+...+pₖ^aₖ)
- **GCD(m,n):** Take the minimum exponent of each prime
- **LCM(m,n):** Take the maximum exponent of each prime

## GCD and the Euclidean algorithm

The Euclidean algorithm computes gcd(a,b) efficiently:
- gcd(a, b) = gcd(b, a mod b), repeat until remainder is 0
- The last non-zero remainder is the GCD

Beyond computing GCD, the algorithm reveals structure. For polynomial or algebraic expressions, use the identity:

gcd(a+b, a) = gcd(b, a)

This is useful for expressions like gcd(x+y, x²−xy+y²): substitute to get gcd(x+y, 3y²) or similar, which factors the problem into manageable pieces.

**Key property:** gcd(a,b) = 1 (coprimality) means the Euclidean algorithm reaches 1. Consecutive Fibonacci numbers are the "hardest" inputs — they require the most steps to reach gcd = 1. This fact appears in problems asking for the pair with largest norm that is still coprime.

## LCM for simultaneous divisibility

When "N must be divisible by each of a, b, c, ...," the answer is: N must be a multiple of lcm(a, b, c, ...).

Compute LCM via prime factorizations (take max exponents) or using:
lcm(a,b) = a × b / gcd(a,b)

For multiple values, compute iteratively: lcm(a,b,c) = lcm(lcm(a,b), c).

After finding the LCM, generate all valid multiples within the problem's range. Often there are very few — a problem asking for N < 100 where N is divisible by 2, 5, and 7 gives lcm = 70, and the only valid N is 70.

## The Frobenius (coin) problem

"What is the largest integer that cannot be represented as a non-negative integer combination of a and b?"

**If gcd(a,b) = 1:** The Frobenius number is ab − a − b.

This formula only works for two coprime values. For three or more values, there is no simple closed form — enumerate or use dynamic programming.

**Procedure:**
1. Verify gcd(a,b) = 1 — if not, infinitely many integers are unrepresentable (all those not divisible by gcd(a,b))
2. Apply the formula: F = ab − a − b
3. Verify by checking that F is indeed unrepresentable (show no non-negative solution to ax + by = F exists) and that F + 1, F + 2, ..., F + a are all representable

## GCD of polynomial expressions

For expressions like gcd(x³ + y³, x + y), use algebraic factoring:

x³ + y³ = (x+y)(x²−xy+y²)

So gcd(x³+y³, x+y) = (x+y) × gcd(x²−xy+y², 1) = x+y (assuming x+y > 0).

For gcd(x+y, x²−xy+y²), note that x²−xy+y² = (x+y)² − 3xy, so:
gcd(x+y, x²−xy+y²) = gcd(x+y, 3xy)

If gcd(x,y) = 1, this simplifies to gcd(x+y, 3), which is either 1 or 3.

This technique — reducing GCDs of polynomial expressions using algebraic identities — is a recurring theme in competition number theory.

## Diophantine equations

For equations in integers:

1. **Linear Diophantine ax + by = c:** Solvable iff gcd(a,b) | c. Find one solution using the extended Euclidean algorithm, then the general solution is x = x₀ + (b/d)t, y = y₀ − (a/d)t where d = gcd(a,b).
2. **Pell equations x² − Dy² = 1:** Find the fundamental solution by continued fraction expansion of √D, then generate all solutions via (x₁ + y₁√D)ⁿ.
3. **Sum of squares, cubes, etc.:** Use modular arithmetic to rule out impossible residues before searching for solutions. For instance, squares mod 4 are 0 or 1 — so x² + y² ≡ 3 (mod 4) has no solution.

## Common traps

- **Applying Frobenius formula when gcd ≠ 1.** The formula ab − a − b requires coprimality. If gcd(a,b) = d > 1, first note that only multiples of d can be represented, then apply the formula to a/d and b/d and multiply back.
- **Forgetting that lcm can be large.** lcm(a,b) can be as large as a×b (when they're coprime). Check that the LCM fits within the problem's constraints.
- **Assuming gcd(a+b, ab) = 1.** This is not generally true. Compute it from the prime factorizations of a and b.

## Verification

For number theory problems:
1. Check the answer satisfies all divisibility conditions stated in the problem
2. For Frobenius-type problems, verify both that F is unrepresentable and that all integers > F are representable (checking the next a values suffices)
3. For GCD/LCM results, verify using the original numbers, not derived quantities
