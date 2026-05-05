---
name: algebraic-manipulation
description: Techniques for algebraic manipulation, substitution, and equation-solving in competition mathematics. Use this skill whenever a problem requires simplifying expressions, solving polynomial equations, optimizing functions, manipulating inequalities algebraically, or finding values of parameters. Also use when a problem involves Vieta's formulas, symmetric functions of roots, functional equations, nested radicals, or telescoping. Trigger on problems that say "find the value of," "simplify," "solve the equation," "determine the parameter," or any problem where algebraic rearrangement is the main challenge. If a problem has equations with parameters and asks for which parameter values certain conditions hold, this skill applies.
---

# Algebraic Manipulation & Substitution

## General approach

Competition algebra problems reward recognizing structure before computing. Spend a moment looking at the equation's symmetry, degree, and special values before manipulating. A well-chosen substitution or factoring can collapse an intimidating problem into a routine one.

## Vieta's formulas — the bridge from coefficients to roots

For a quadratic ax² + bx + c = 0 with roots x₁, x₂:
- x₁ + x₂ = −b/a
- x₁ × x₂ = c/a

These let you compute symmetric functions of roots without solving the equation. The most useful identity:

**x₁² + x₂² = (x₁ + x₂)² − 2x₁x₂**

This converts a sum-of-squares (which Vieta doesn't give directly) into quantities Vieta does give. Similarly:
- x₁³ + x₂³ = (x₁ + x₂)³ − 3x₁x₂(x₁ + x₂)
- |x₁ − x₂| = √((x₁+x₂)² − 4x₁x₂)

After using Vieta to find parameter values, always check two things:
1. **Discriminant ≥ 0** — the equation must have real roots. Reject parameter values that give negative discriminants.
2. **Any domain constraints** — if roots represent lengths, they must be positive. Check the sign of the sum and product.

## Test obvious values first

Before deploying heavy algebra, plug in simple values: 0, 1, −1, and any value suggested by the problem's structure. This often finds solutions immediately, and at minimum gives you a known solution to verify against later.

After finding solutions by testing, prove completeness — show no other solutions exist. Common tools for this:
- **Monotonicity argument:** If f(x) is strictly increasing on the domain, it can equal any value at most once.
- **Degree argument:** A polynomial of degree n has at most n roots.
- **Derivative analysis:** Show the function has at most k critical points, limiting the number of intersections with a horizontal line.

## Symmetry exploitation

When an expression is symmetric in its variables (unchanged when you swap any two variables), the extremum often occurs when all variables are equal. This is a heuristic, not a theorem — but it's correct surprisingly often in competition settings.

**Procedure:**
1. Observe the symmetry: "f(x,y) = f(y,x)"
2. Set x = y and solve the reduced problem
3. Verify this is actually an extremum (check the second-order condition or boundary values)
4. If the problem asks for the minimum/maximum over a constrained domain, also check the boundary

## Telescoping

Telescoping simplifies sums or products where consecutive terms cancel.

**Sums:** If you can write aₖ = f(k) − f(k−1), then Σaₖ = f(n) − f(0). Look for partial fraction decompositions:
- 1/(k(k+1)) = 1/k − 1/(k+1)
- 1/(k(k+2)) = (1/2)(1/k − 1/(k+2))

**Products:** If you can write aₖ = g(k)/g(k−1), then Πaₖ = g(n)/g(0). Running-average recurrences often telescope this way:
- S_{n+1} = S_n × (n+1)/n → product telescopes to S_n = S₁ × n

After telescoping, verify the first and last few terms manually to confirm the pattern holds at the boundaries.

## Squaring to remove radicals

When an equation contains square roots, isolate one radical on one side, then square both sides. Repeat if multiple radicals remain.

**Critical warning:** Squaring can introduce extraneous solutions. After solving the squared equation, substitute every candidate solution back into the original (unsquared) equation. Reject any that fail. This is not optional — it's where most errors in radical equations occur.

## Substitution strategies

Good substitutions reduce the number of variables or the degree of the equation:

- **Reciprocal substitution:** For equations symmetric in x and 1/x, set t = x + 1/x. This reduces degree by half.
- **Trigonometric substitution:** For expressions involving √(a² − x²), set x = a sin θ. For √(x² + a²), set x = a tan θ.
- **Homogenization:** For equations where all terms have the same total degree, divide by one variable's power to reduce to fewer variables.
- **u = x + y, v = xy:** For symmetric systems in two variables, reduce to Vieta-like form.

## Polynomial factoring

Before expanding or using the quadratic formula, check if the polynomial factors nicely:

1. **Rational root theorem:** If p/q is a rational root of a polynomial with integer coefficients, then p divides the constant term and q divides the leading coefficient. Test these candidates.
2. **Grouping:** ax³ + bx² + cx + d might factor as x²(ax + b) + (cx + d) if the ratios work out.
3. **Special forms:** Recognize x² − y² = (x−y)(x+y), x³ ± y³ = (x±y)(x²∓xy+y²), and sum/difference of nth powers.

## Parameter problems

When a problem asks "for which values of parameter m does [condition] hold":

1. Solve the equation/system in terms of m, getting roots as functions of m
2. Apply the required condition (real roots → discriminant ≥ 0; positive roots → sum > 0 and product > 0; roots are side lengths of a triangle → triangle inequality)
3. Solve the resulting inequality/equation for m
4. **Check each candidate m** by substituting back into the original equation — ensure the equation's degree doesn't degenerate (e.g., if m = 0 makes the leading coefficient vanish, the equation becomes linear, not quadratic)

## Common traps

- **Forgetting to check discriminant** after using Vieta's formulas — you might find m values that give complex roots.
- **Degree collapse** — when a parameter value makes the leading coefficient zero, the equation changes type. Handle this case separately.
- **Domain restrictions** — if the original problem constrains variables (e.g., x > 0, or x represents a length), enforce these after solving.
- **Extraneous roots from squaring** — always substitute back into the original equation.

## Verification

Always plug your final answer back into the original problem statement (not a derived equation). For parameter problems, verify that the claimed solution actually satisfies all stated conditions.
