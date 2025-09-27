# Partial Derivatives: Complete Mathematical Guide

## What is a Partial Derivative?

A **partial derivative** measures how a function changes when you vary **only one** of its input variables, while keeping all other variables **constant**.

### Mathematical Notation

For a function `f(x, y, z)` with multiple variables:
- `∂f/∂x` = partial derivative with respect to x (y and z held constant)
- `∂f/∂y` = partial derivative with respect to y (x and z held constant)  
- `∂f/∂z` = partial derivative with respect to z (x and y held constant)

The symbol `∂` (partial) distinguishes it from `d` (ordinary derivative).

## Intuitive Understanding

### Geometric Interpretation

Imagine a 3D surface `z = f(x, y)`:
- `∂f/∂x` = slope of the surface when you move in the x-direction
- `∂f/∂y` = slope of the surface when you move in the y-direction

It's like taking a "slice" of the surface and measuring the slope of that slice.

### Physical Analogy

Think of temperature `T(x, y, t)` in a room:
- `∂T/∂x` = how temperature changes as you move east-west
- `∂T/∂y` = how temperature changes as you move north-south
- `∂T/∂t` = how temperature changes over time

Each partial derivative isolates one dimension of change.

## Mathematical Examples

### Example 1: Simple Polynomial

**Function:** `f(x, y) = x² + 3xy + y²`

**Partial derivatives:**
```
∂f/∂x = ∂/∂x(x² + 3xy + y²) = 2x + 3y + 0 = 2x + 3y
∂f/∂y = ∂/∂y(x² + 3xy + y²) = 0 + 3x + 2y = 3x + 2y
```

**Key insight:** When differentiating with respect to x, treat y as a constant (and vice versa).

### Example 2: Exponential Function

**Function:** `f(x, y) = e^(xy)`

**Partial derivatives:**
```
∂f/∂x = ∂/∂x(e^(xy)) = e^(xy) × y = y × e^(xy)
∂f/∂y = ∂/∂y(e^(xy)) = e^(xy) × x = x × e^(xy)
```

**Note:** Chain rule applies - derivative of `e^u` is `e^u × du/dx`.

### Example 3: Mixed Functions

**Function:** `f(x, y, z) = x²y + sin(z) + xyz`

**Partial derivatives:**
```
∂f/∂x = 2xy + 0 + yz = 2xy + yz
∂f/∂y = x² + 0 + xz = x² + xz  
∂f/∂z = 0 + cos(z) + xy = cos(z) + xy
```

## Rules for Computing Partial Derivatives

### 1. **Constant Rule**
If a term doesn't contain the variable you're differentiating with respect to, it becomes 0.
```
∂/∂x(y²) = 0    (y² is constant with respect to x)
∂/∂y(5x) = 0    (5x is constant with respect to y)
```

### 2. **Power Rule**
```
∂/∂x(x^n) = n × x^(n-1)
∂/∂x(x²y³) = 2x × y³    (treat y³ as constant)
```

### 3. **Product Rule**
```
∂/∂x(u(x,y) × v(x,y)) = (∂u/∂x) × v + u × (∂v/∂x)
```

### 4. **Chain Rule**
```
∂/∂x(f(g(x,y))) = f'(g(x,y)) × (∂g/∂x)
```

## Detailed Step-by-Step Example

Let's compute partial derivatives for: `f(x, y) = x³y² + 2xy + y⁴`

### Computing ∂f/∂x

**Step 1:** Identify all terms
- Term 1: `x³y²`
- Term 2: `2xy`  
- Term 3: `y⁴`

**Step 2:** Differentiate each term with respect to x (treat y as constant)
- `∂/∂x(x³y²) = 3x² × y² = 3x²y²`
- `∂/∂x(2xy) = 2 × y = 2y`
- `∂/∂x(y⁴) = 0` (no x in this term)

**Step 3:** Combine results
```
∂f/∂x = 3x²y² + 2y + 0 = 3x²y² + 2y
```

### Computing ∂f/∂y

**Step 1:** Same terms as above

**Step 2:** Differentiate each term with respect to y (treat x as constant)
- `∂/∂y(x³y²) = x³ × 2y = 2x³y`
- `∂/∂y(2xy) = 2x × 1 = 2x`
- `∂/∂y(y⁴) = 4y³`

**Step 3:** Combine results
```
∂f/∂y = 2x³y + 2x + 4y³
```

## Numerical Verification

Let's verify our results numerically:

**Given:** `f(x, y) = x³y² + 2xy + y⁴` at point `(x=2, y=1)`

**Function value:** `f(2,1) = 2³×1² + 2×2×1 + 1⁴ = 8 + 4 + 1 = 13`

### Verifying ∂f/∂x = 3x²y² + 2y

**At (2,1):** `∂f/∂x = 3×2²×1² + 2×1 = 12 + 2 = 14`

**Numerical check:** 
- `f(2.001, 1) = (2.001)³×1² + 2×(2.001)×1 + 1⁴ ≈ 13.014`
- Rate of change = `(13.014 - 13) / 0.001 = 14` ✓

### Verifying ∂f/∂y = 2x³y + 2x + 4y³

**At (2,1):** `∂f/∂y = 2×2³×1 + 2×2 + 4×1³ = 16 + 4 + 4 = 24`

**Numerical check:**
- `f(2, 1.001) = 2³×(1.001)² + 2×2×(1.001) + (1.001)⁴ ≈ 13.024`
- Rate of change = `(13.024 - 13) / 0.001 = 24` ✓

## Applications in Neural Networks

### 1. **Loss Function Gradients**

For MSE loss: `L(ŷ, y) = (1/n) × Σᵢ(ŷᵢ - yᵢ)²`

```
∂L/∂ŷᵢ = (2/n) × (ŷᵢ - yᵢ)
```

This partial derivative tells us how the loss changes when we adjust prediction `i`.

### 2. **Weight Gradients**

For a linear layer: `output = Σⱼ(wⱼ × inputⱼ)`

```
∂output/∂wₖ = inputₖ
```

This shows how the output changes when we adjust weight `k`.

### 3. **Chain Rule in Backpropagation**

```
∂Loss/∂weight = ∂Loss/∂output × ∂output/∂weight
```

This is the fundamental equation for training neural networks!

## Common Mistakes and How to Avoid Them

### Mistake 1: Forgetting to Treat Other Variables as Constants

**Wrong:** `∂/∂x(xy) = y × ∂y/∂x` ❌

**Correct:** `∂/∂x(xy) = y` ✓ (y is constant with respect to x)

### Mistake 2: Confusing Partial and Total Derivatives

**Partial:** `∂f/∂x` (only x varies)
**Total:** `df/dx` (all variables may depend on x)

### Mistake 3: Not Applying Chain Rule

**Function:** `f(x,y) = sin(x²y)`

**Wrong:** `∂f/∂x = cos(x²y)` ❌

**Correct:** `∂f/∂x = cos(x²y) × 2xy` ✓ (chain rule: multiply by derivative of inside)

## Higher-Order Partial Derivatives

You can take partial derivatives of partial derivatives:

### Second-Order Partials

For `f(x,y) = x³y² + 2xy`:
```
∂f/∂x = 3x²y² + 2y
∂²f/∂x² = ∂/∂x(3x²y² + 2y) = 6xy²

∂f/∂y = 2x³y + 2x  
∂²f/∂y² = ∂/∂y(2x³y + 2x) = 2x³
```

### Mixed Partials

```
∂²f/∂x∂y = ∂/∂x(∂f/∂y) = ∂/∂x(2x³y + 2x) = 6x²y + 2
∂²f/∂y∂x = ∂/∂y(∂f/∂x) = ∂/∂y(3x²y² + 2y) = 6x²y + 2
```

**Note:** Mixed partials are equal for well-behaved functions (Clairaut's theorem).

## The Gradient Vector

The **gradient** is the vector of all partial derivatives:

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**Properties:**
- Points in direction of steepest increase
- Magnitude indicates rate of steepest increase
- Perpendicular to level curves/surfaces

### Example: Gradient of f(x,y) = x² + y²

```
∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]
```

At point (3,4): `∇f = [6, 8]`
- Direction: toward point (3,4) from origin
- Magnitude: `√(6² + 8²) = 10`

## Summary

**Partial derivatives** are the foundation of:
1. **Multivariable calculus**
2. **Neural network training** (backpropagation)
3. **Optimization algorithms** (gradient descent)
4. **Physics and engineering** (field equations)

**Key takeaways:**
- Vary one variable, hold others constant
- Use standard derivative rules
- Apply chain rule when needed
- Verify numerically when in doubt
- Gradients point toward steepest increase

Understanding partial derivatives is essential for deep learning, as they enable us to compute how small changes in weights affect the loss function, which is the core mechanism behind neural network training.
