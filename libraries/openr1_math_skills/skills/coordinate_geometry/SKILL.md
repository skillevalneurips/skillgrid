---
name: coordinate-geometry
description: Techniques for coordinate geometry and geometric transformation problems in competition mathematics. Use this skill whenever a problem involves distances, angles, or positions of geometric objects (points, lines, circles, conics) in the plane or space. Also use for problems about ellipses, parabolas, hyperbolas, tangent lines, reflections, rotations, or any problem where assigning coordinates to geometric objects could simplify the analysis. Trigger on problems mentioning triangles with specific side lengths, circles with given radii and centers, loci of points, or any "find the area/length/angle" problem. If a problem mentions foci, eccentricity, directrix, or tangency, this skill applies.
---

# Coordinate Geometry & Transformations

## When to use coordinates

Coordinates are powerful when a problem involves specific lengths, positions, or distance constraints. They convert geometric reasoning into algebra, which is more mechanical and less error-prone for complex configurations. However, for problems with heavy symmetry or those that ask for ratios/angles only, synthetic approaches or trigonometric identities may be cleaner.

**Default strategy:** Set up coordinates first. If the algebra becomes unwieldy, switch to synthetic methods. The reverse (trying synthetic first, then switching to coordinates) tends to waste more time because synthetic attempts that fail give you less reusable work.

## Setting up coordinates well

The quality of your coordinate system determines the difficulty of the algebra. Invest time here.

1. **Place the origin at a center of symmetry** — the circumcenter of a triangle, the center of a circle, the midpoint of a segment.
2. **Align an axis with a key line** — put a side of the triangle along the x-axis, or the major axis of an ellipse along the x-axis.
3. **Exploit given right angles** — if two segments are perpendicular, make them the axes.
4. **Minimize the number of free variables.** If a triangle has vertices A, B, C and side AB has known length, place A at the origin and B at (c, 0). Then C has only two unknowns.

## Distance and metric definitions

Before writing distance equations, confirm which metric the problem uses:
- **Euclidean distance:** d = √((x₂−x₁)² + (y₂−y₁)²)
- **Chebyshev (chessboard) distance:** d = max(|x₂−x₁|, |y₂−y₁|)
- **Manhattan (taxicab) distance:** d = |x₂−x₁| + |y₂−y₁|

Chebyshev distance appears in chessboard/grid problems where kings can move diagonally. Chebyshev "circles" are squares rotated 45° relative to Manhattan "circles." State the metric explicitly before setting up equations.

## Conic sections

### Ellipses
- **Definition:** Sum of distances to two foci equals 2a (the major axis length). Use this directly rather than the standard equation when the ellipse might not be axis-aligned.
- **Reflection property:** The tangent at any point P bisects the external angle between the lines PF₁ and PF₂. Equivalently, reflecting one focus across the tangent gives a point collinear with P and the other focus.
- **Standard equation (axis-aligned):** x²/a² + y²/b² = 1, where c² = a² − b² and c is the focal distance.
- For tilted ellipses, work from the distance-sum definition rather than trying to write a rotated standard equation.

### Circles
- **Tangent line is perpendicular to radius at point of tangency.** If a line is tangent to a circle at point M, then the center-to-M segment is perpendicular to the tangent.
- **Power of a point:** For a point P outside circle with center O and radius r, the power is PO² − r². This equals PA × PB for any secant through P.

### Parabolas
- **Definition:** Distance to focus equals distance to directrix.
- **Reflection property:** Rays parallel to the axis reflect through the focus.

## Perpendicularity and collinearity

- **Perpendicularity:** Two vectors (a,b) and (c,d) are perpendicular iff ac + bd = 0. Two lines with slopes m₁ and m₂ are perpendicular iff m₁m₂ = −1 (be careful with vertical lines — use the dot product form instead).
- **Collinearity of three points:** Points A, B, C are collinear iff the area of triangle ABC is 0, i.e., (B−A) × (C−A) = 0 (cross product equals zero).

## Symmetry reductions

Before solving, check for symmetries that reduce the number of unknowns:

- **Equidistant from two points** → the locus is the perpendicular bisector of the segment joining them.
- **Problem unchanged under reflection** → the solution likely lies on the axis of symmetry. Try imposing this constraint and see if it yields a valid solution.
- **Rotational symmetry** → use polar coordinates or reduce to one angular variable.

## When coordinates get messy: fallback strategies

If the coordinate algebra produces expressions with many terms or nested radicals:

1. **Law of Cosines:** c² = a² + b² − 2ab cos C. Use this to find angles or sides when you know three elements of a triangle.
2. **Law of Sines:** a/sin A = b/sin B = c/sin C = 2R. Use this when you know an angle and its opposite side.
3. **Stewart's Theorem, Angle Bisector Theorem, or Power of a Point** — these give direct relationships without coordinates.
4. **Area formulas:** Heron's formula, shoelace formula, or (1/2)|ab sin C|.

## Common traps

- **Assuming axis-alignment.** An ellipse or parabola may be tilted. Check whether the problem specifies orientation before using x²/a² + y²/b² = 1.
- **Forgetting to check both intersection points.** A line intersecting a circle or conic typically gives two solutions. Verify which satisfies the original geometric constraints.
- **Sign errors in distance formulas.** When squaring both sides of a distance equation, extraneous solutions can appear. Always verify the final answer satisfies the original (unsquared) condition.
- **Confusing different distance metrics.** Chebyshev vs. Euclidean vs. Manhattan give different "circles" and different nearest-point results.

## Verification

After finding a geometric quantity:
1. Check dimensional consistency (lengths are positive, areas are non-negative)
2. Verify with a special case or degenerate configuration
3. If possible, compute using two independent methods (e.g., coordinates + area formula)
