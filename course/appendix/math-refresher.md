# Math Refresher

Everything you need to know for this course — and nothing more.

---

## 1. Exponents

**What:** Repeated multiplication.

```
2³ = 2 × 2 × 2 = 8
5² = 5 × 5 = 25
```

**Rules:**
```
a^m × a^n = a^(m+n)      Example: 2³ × 2² = 2⁵ = 32
a^(-n) = 1 / a^n          Example: 2^(-3) = 1/8
a^(1/2) = √a              Example: 9^(1/2) = 3
a^(-1/2) = 1/√a           Example: 4^(-0.5) = 1/2
a^0 = 1                   Example: 5⁰ = 1
```

**Where it appears in microgpt.py:**
- `(ms + 1e-5) ** -0.5` — computing 1/√(mean square) in RMSNorm
- `self.data**other` — the power operation in the `Value` class

---

## 2. The Exponential Function (e^x)

**What:** The number `e ≈ 2.718` raised to the power x.

```
e⁰ = 1
e¹ ≈ 2.718
e² ≈ 7.389
e^(-1) ≈ 0.368
```

Key properties:
- Always positive: e^x > 0 for all x
- Grows very fast for large x
- Approaches zero for large negative x
- Its derivative is itself: d(e^x)/dx = e^x

**Where it appears:**
- `math.exp(self.data)` — used in softmax to make all values positive

---

## 3. The Logarithm (ln or log)

**What:** The inverse of the exponential. `ln(x)` answers: "what power do I raise e to, to get x?"

```
ln(1) = 0        because e⁰ = 1
ln(e) = 1        because e¹ = e
ln(7.389) ≈ 2    because e² ≈ 7.389
```

Key properties:
- Only defined for positive numbers: ln(x) requires x > 0
- ln(1) = 0
- ln(x) < 0 when 0 < x < 1
- Grows very slowly for large x
- Derivative: d(ln x)/dx = 1/x

**Where it appears:**
- `-probs[target_id].log()` — the cross-entropy loss function

---

## 4. Summation (Σ)

**What:** Shorthand for "add up a bunch of things."

```
Σᵢ₌₁³ xᵢ = x₁ + x₂ + x₃
```

In Python: `sum(x[i] for i in range(3))`

**Where it appears:**
- `sum(wi * xi for wi, xi in zip(wo, x))` — dot products in linear layers
- `sum(losses)` — averaging losses

---

## 5. Multiplication Symbol (×)

Used interchangeably with `*` in this course:
- `a × b` = `a * b`
- `3 × 4 = 12`

---

## 6. Derivatives (Basics)

The derivative `dy/dx` tells you: "if x changes by a tiny bit, how much does y change?"

Common derivatives used in this course:

| Function | Derivative | In English |
|----------|-----------|------------|
| y = c (constant) | dy/dx = 0 | Constants don't change |
| y = x | dy/dx = 1 | 1-to-1 relationship |
| y = c × x | dy/dx = c | Scales the change |
| y = x² | dy/dx = 2x | |
| y = x^n | dy/dx = n × x^(n-1) | Power rule |
| y = e^x | dy/dx = e^x | Exponential is its own derivative |
| y = ln(x) | dy/dx = 1/x | |
| y = max(0, x) | dy/dx = 1 if x > 0, else 0 | Step function |

**The Chain Rule:** For composed functions:
```
d/dx f(g(x)) = f'(g(x)) × g'(x)
```

"Multiply the derivatives along the chain."

---

## 7. Vectors (Lists of Numbers)

A **vector** is just a list of numbers:
```
v = [3.0, -1.0, 2.5]    (a 3-dimensional vector)
```

Operations on vectors:
```
Addition:     [1, 2] + [3, 4] = [4, 6]           (element-wise)
Scalar mult:  2 × [3, 4] = [6, 8]                (multiply each element)
Dot product:  [1, 2] · [3, 4] = 1×3 + 2×4 = 11  (multiply + sum)
```

**Where they appear:** Every embedding, every layer input/output is a vector.

---

## 8. Matrices (Grids of Numbers)

A **matrix** is a 2D grid:
```
M = [[1, 2, 3],
     [4, 5, 6]]    (a 2×3 matrix: 2 rows, 3 columns)
```

**Matrix-vector multiplication:** The `linear()` function in microgpt.py:
```
M × v = [row₀ · v, row₁ · v]

[[1, 2, 3],    [1]     [1×1 + 2×2 + 3×3]   [14]
 [4, 5, 6]]  × [2]  =  [4×1 + 5×2 + 6×3] = [32]
               [3]
```

Each row's dot product with the input gives one output element.

---

## 9. Probability

A **probability distribution** assigns a number between 0 and 1 to each possible outcome, with all probabilities summing to 1.

```
P('a') = 0.3
P('b') = 0.5
P('c') = 0.2
───────────
Sum    = 1.0 ✓
```

**Random sampling:** Choosing an outcome where each option's chance equals its probability.

---

## 10. Square Root (√)

```
√4 = 2       (because 2² = 4)
√9 = 3       (because 3² = 9)
√2 ≈ 1.414
```

In code: `x ** 0.5` or `math.sqrt(x)`

**Where it appears:**
- `head_dim**0.5` — scaling in attention
- `v_hat ** 0.5` — in Adam optimizer

---

## 11. Mean (Average)

```
mean([2, 4, 6]) = (2 + 4 + 6) / 3 = 4
```

In code: `sum(x) / len(x)`

**Where it appears:**
- `sum(xi * xi for xi in x) / len(x)` — mean of squares in RMSNorm

---

That's all the math. Every formula in microgpt.py uses only these concepts.
