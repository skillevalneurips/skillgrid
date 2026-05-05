---
name: modular-arithmetic
description: Techniques for modular arithmetic and divisibility problems in competition mathematics. Use this skill whenever a problem involves remainders, divisibility conditions, congruences, modular exponentiation, finding residues, or determining divisibility by specific numbers. Also use when the problem asks about last digits, periodicity of powers mod n, or constraints like "leaves remainder r when divided by d." Trigger on phrases like "remainder," "divisible by," "mod," "congruent," "last digit," or any problem where reducing modulo a number simplifies the structure.
---

# Modular Arithmetic & Divisibility

## Core approach

Before diving into computation, identify the modular structure of the problem. Most competition problems involving remainders or divisibility have a clean solution once you frame them in the right modular setting.

## Remainder decomposition

When told "N divided by D leaves remainder R," immediately write:

N = D × Q + R, where 0 ≤ R < D

This gives you D × Q = N − R, which converts a remainder problem into a divisibility/factoring problem. The constraint R < D is a filter that eliminates candidates — always apply it.

**Example:**
- "491 divided by a two-digit integer leaves remainder 59"
- D × Q = 491 − 59 = 432, and D must be a two-digit number greater than 59
- Factor 432 = 2⁴ × 3³, enumerate two-digit divisors, filter by D > 59

## Modular exponentiation

For computing aⁿ mod m, follow this decision tree:

1. **Check if gcd(a, m) = 1.** If yes, Euler's theorem applies: a^φ(m) ≡ 1 (mod m).
2. **If m is prime**, use Fermat's little theorem: a^(p−1) ≡ 1 (mod p). This is simpler and should be preferred when applicable.
3. **Reduce the exponent:** Write n = φ(m) × q + r, then aⁿ ≡ aʳ (mod m).
4. **If gcd(a, m) ≠ 1**, factor m and use CRT, or compute the cycle of powers directly.

Always verify by computing the first few powers manually to confirm the cycle length matches what the theorem predicts. This cross-check catches errors in the totient computation.

**Example:** 7^2023 mod 9
- gcd(7, 9) = 1, φ(9) = 6
- 2023 = 6 × 337 + 1, so 7^2023 ≡ 7¹ ≡ 7 (mod 9)
- Verify: 7¹=7, 7²=49≡4, 7³≡28≡1, wait — cycle length is actually 3, not 6. But 2023 mod 3 = 1, so 7^2023 ≡ 7 mod 9. Same answer, but the manual check revealed the true cycle length is a divisor of φ(m), not necessarily φ(m) itself.

## Cycle detection for residues

When Euler/Fermat feels heavy or gcd(a,m) ≠ 1, just compute successive powers mod m until a value repeats:

1. List a¹ mod m, a² mod m, a³ mod m, ...
2. When you see a repetition, you've found the cycle length c.
3. Reduce: aⁿ ≡ a^(n mod c) (mod m), being careful about indexing — if the cycle starts at a¹, then n mod c = 0 corresponds to a^c, not a⁰.

This approach is robust and avoids the risk of misapplying theorems. Use it as a cross-check even when you also use a theorem.

## Prime factorization before divisor enumeration

Never enumerate divisors by trial. Always factor first, then generate divisors systematically from the prime factorization.

For N = p₁^a₁ × p₂^a₂ × ... × pₖ^aₖ:
- Total number of divisors: (a₁+1)(a₂+1)...(aₖ+1)
- Generate all divisors by choosing an exponent for each prime from 0 to aᵢ
- Apply any filters (two-digit, greater than R, etc.) as you enumerate

## Common traps

- **Remainder must be less than divisor.** If you find D divides (N − R) but D ≤ R, that D is invalid. This filter is easy to forget.
- **Fermat's little theorem requires prime modulus.** For composite m, you need Euler's theorem (with coprimality) or direct cycle computation.
- **Cycle length divides φ(m) but may not equal it.** Always verify the actual cycle length rather than assuming it equals φ(m).
- **Index alignment when reducing exponents.** If the cycle length is c and n mod c = 0, the answer is a^c mod m (not a⁰ = 1). Write out the mapping explicitly for the first few terms.

## Verification

After obtaining your answer, verify it by:
1. Plugging back into the original divisibility condition
2. Computing the residue directly for a small related case
3. Using an independent method (e.g., cycle detection to cross-check a theorem-based answer)
