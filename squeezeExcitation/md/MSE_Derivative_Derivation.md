# MSE Derivative Derivation: Step-by-Step Mathematical Proof

## The Question

Given:
```
MSE = (1/n) × Σᵢ(ŷᵢ - yᵢ)²
```

Why is the derivative:
```
∂MSE/∂ŷᵢ = (2/n) × (ŷᵢ - yᵢ)
```

Let's derive this rigorously using calculus.

## Step-by-Step Derivation

### Step 1: Expand the MSE Formula

```
MSE = (1/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)²
```

This can be written as:
```
MSE = (1/n) × [(ŷ₁ - y₁)² + (ŷ₂ - y₂)² + ... + (ŷₙ - yₙ)²]
```

### Step 2: Apply the Partial Derivative

We want to find `∂MSE/∂ŷⱼ` for a specific prediction `ŷⱼ`.

Using the partial derivative:
```
∂MSE/∂ŷⱼ = ∂/∂ŷⱼ [(1/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)²]
```

### Step 3: Factor Out the Constant

Since `(1/n)` is a constant:
```
∂MSE/∂ŷⱼ = (1/n) × ∂/∂ŷⱼ [Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)²]
```

### Step 4: Apply Derivative to the Sum

The derivative of a sum is the sum of derivatives:
```
∂MSE/∂ŷⱼ = (1/n) × Σᵢ₌₁ⁿ [∂/∂ŷⱼ (ŷᵢ - yᵢ)²]
```

### Step 5: Evaluate Each Term in the Sum

For each term `(ŷᵢ - yᵢ)²`, we need `∂/∂ŷⱼ (ŷᵢ - yᵢ)²`.

**Case 1: When i ≠ j**
- The term `(ŷᵢ - yᵢ)²` doesn't contain `ŷⱼ`
- Therefore: `∂/∂ŷⱼ (ŷᵢ - yᵢ)² = 0`

**Case 2: When i = j**
- The term becomes `(ŷⱼ - yⱼ)²`
- We need: `∂/∂ŷⱼ (ŷⱼ - yⱼ)²`

### Step 6: Apply Chain Rule for i = j

For the term `(ŷⱼ - yⱼ)²`, using the chain rule:

Let `u = ŷⱼ - yⱼ`, then we have `u²`

```
∂/∂ŷⱼ (ŷⱼ - yⱼ)² = ∂/∂ŷⱼ (u²) = ∂u²/∂u × ∂u/∂ŷⱼ
```

Where:
- `∂u²/∂u = 2u = 2(ŷⱼ - yⱼ)`
- `∂u/∂ŷⱼ = ∂(ŷⱼ - yⱼ)/∂ŷⱼ = 1` (since `yⱼ` is constant)

Therefore:
```
∂/∂ŷⱼ (ŷⱼ - yⱼ)² = 2(ŷⱼ - yⱼ) × 1 = 2(ŷⱼ - yⱼ)
```

### Step 7: Combine the Results

From Steps 5 and 6:
- Only the term where `i = j` contributes to the derivative
- All other terms contribute 0

So:
```
∂MSE/∂ŷⱼ = (1/n) × [0 + 0 + ... + 2(ŷⱼ - yⱼ) + ... + 0]
∂MSE/∂ŷⱼ = (1/n) × 2(ŷⱼ - yⱼ)
∂MSE/∂ŷⱼ = (2/n) × (ŷⱼ - yⱼ)
```

## Verification with Concrete Example

Let's verify with a simple example:

**Given:**
- n = 3 samples
- Predictions: `ŷ = [1, 2, 3]`
- Targets: `y = [1.5, 1.8, 2.9]`

**MSE Calculation:**
```
MSE = (1/3) × [(1-1.5)² + (2-1.8)² + (3-2.9)²]
MSE = (1/3) × [0.25 + 0.04 + 0.01]
MSE = (1/3) × 0.3 = 0.1
```

**Derivative for ŷ₁:**
Using our formula: `∂MSE/∂ŷ₁ = (2/3) × (1 - 1.5) = (2/3) × (-0.5) = -1/3`

**Numerical Verification:**
If we increase `ŷ₁` by a small amount `ε = 0.001`:
- New `ŷ₁ = 1.001`
- New MSE = `(1/3) × [(1.001-1.5)² + (2-1.8)² + (3-2.9)²]`
- New MSE = `(1/3) × [0.249001 + 0.04 + 0.01] = 0.099667`
- Change in MSE = `0.099667 - 0.1 = -0.000333`
- Rate of change = `-0.000333 / 0.001 = -0.333 ≈ -1/3` ✓

## Alternative Derivation Using Matrix Notation

For a more compact derivation, we can use matrix notation:

**Vector form:**
```
MSE = (1/n) × ||ŷ - y||²
```

Where `||·||²` is the squared L2 norm.

**Expanding:**
```
MSE = (1/n) × (ŷ - y)ᵀ(ŷ - y)
```

**Taking the gradient:**
```
∇ŷ MSE = (1/n) × ∇ŷ [(ŷ - y)ᵀ(ŷ - y)]
```

Using the matrix derivative rule `∇ₓ (xᵀAx) = 2Ax` when A is symmetric:
```
∇ŷ MSE = (1/n) × 2(ŷ - y) = (2/n) × (ŷ - y)
```

## Why This Form Makes Intuitive Sense

The derivative `∂MSE/∂ŷᵢ = (2/n) × (ŷᵢ - yᵢ)` has several intuitive properties:

1. **Sign**: 
   - If `ŷᵢ > yᵢ` (overestimate), derivative is positive → increase loss
   - If `ŷᵢ < yᵢ` (underestimate), derivative is negative → decrease loss

2. **Magnitude**: 
   - Larger errors `|ŷᵢ - yᵢ|` lead to larger gradients
   - Proportional response to error size

3. **Scaling**: 
   - `(1/n)` factor averages over all samples
   - `2` factor comes from the quadratic nature (power rule)

4. **Gradient Descent**:
   ```
   ŷᵢ_new = ŷᵢ_old - learning_rate × (2/n) × (ŷᵢ_old - yᵢ)
   ```
   This pushes predictions toward targets!

## Summary

The derivative `∂MSE/∂ŷᵢ = (2/n) × (ŷᵢ - yᵢ)` arises from:

1. **Chain rule** application to the squared term
2. **Power rule** giving the factor of 2
3. **Constant factor** (1/n) carrying through
4. **Partial derivative** isolating only the relevant term

This fundamental result is why MSE gradients are linear in the prediction error, making it a well-behaved loss function for gradient-based optimization.
