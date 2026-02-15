# Math Refresher

Everything you need to know for this course — and nothing more.

---

## 1. Exponents

**What:** Repeated multiplication.

$$2^3 = 2 \times 2 \times 2 = 8, \quad 5^2 = 5 \times 5 = 25$$

**Rules:**

| Rule | Example |
|------|---------|
| $a^m \times a^n = a^{m+n}$ | $2^3 \times 2^2 = 2^5 = 32$ |
| $a^{-n} = 1 / a^n$ | $2^{-3} = 1/8$ |
| $a^{1/2} = \sqrt{a}$ | $9^{1/2} = 3$ |
| $a^{-1/2} = 1/\sqrt{a}$ | $4^{-0.5} = 1/2$ |
| $a^0 = 1$ | $5^0 = 1$ |

!!! info "Where it appears in microgpt.py"

    - `(ms + 1e-5) ** -0.5` — computing $1/\sqrt{\text{mean square}}$ in RMSNorm
    - `self.data**other` — the power operation in the `Value` class

---

## 2. The Exponential Function ($e^x$)

The number $e \approx 2.718$ raised to the power $x$:

$$e^0 = 1, \quad e^1 \approx 2.718, \quad e^2 \approx 7.389, \quad e^{-1} \approx 0.368$$

Key properties:

- Always positive: $e^x > 0$ for all $x$
- Grows very fast for large $x$
- Approaches zero for large negative $x$
- Its derivative is itself: $\frac{d}{dx}e^x = e^x$

!!! info "Where it appears"

    `math.exp(self.data)` — used in softmax to make all values positive.

---

## 3. The Logarithm ($\ln$ or $\log$)

The inverse of the exponential. $\ln(x)$ answers: "what power do I raise $e$ to, to get $x$?"

$$\ln(1) = 0, \quad \ln(e) = 1, \quad \ln(7.389) \approx 2$$

Key properties:

- Only defined for positive numbers
- $\ln(1) = 0$
- $\ln(x) < 0$ when $0 < x < 1$
- Derivative: $\frac{d}{dx}\ln(x) = 1/x$

!!! info "Where it appears"

    `-probs[target_id].log()` — the cross-entropy loss function.

---

## 4. Summation ($\Sigma$)

Shorthand for "add up a bunch of things":

$$\sum_{i=1}^{3} x_i = x_1 + x_2 + x_3$$

In Python: `sum(x[i] for i in range(3))`

---

## 5. Derivatives (Basics)

The derivative $\frac{dy}{dx}$ tells you: "if $x$ changes by a tiny bit, how much does $y$ change?"

| Function | Derivative | In English |
|:--------:|:----------:|------------|
| $y = c$ | $\frac{dy}{dx} = 0$ | Constants don't change |
| $y = x$ | $\frac{dy}{dx} = 1$ | 1-to-1 relationship |
| $y = cx$ | $\frac{dy}{dx} = c$ | Scales the change |
| $y = x^2$ | $\frac{dy}{dx} = 2x$ | |
| $y = x^n$ | $\frac{dy}{dx} = nx^{n-1}$ | Power rule |
| $y = e^x$ | $\frac{dy}{dx} = e^x$ | Its own derivative |
| $y = \ln(x)$ | $\frac{dy}{dx} = 1/x$ | |
| $y = \max(0,x)$ | $\frac{dy}{dx} = \begin{cases}1 & x>0 \\ 0 & x \leq 0\end{cases}$ | Step function |

**The Chain Rule:**

$$\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$$

"Multiply the derivatives along the chain."

---

## 6. Vectors (Lists of Numbers)

A **vector** is a list of numbers:

```python
v = [3.0, -1.0, 2.5]    # a 3-dimensional vector
```

| Operation | Example | Result |
|-----------|---------|--------|
| Addition | $[1, 2] + [3, 4]$ | $[4, 6]$ |
| Scalar multiplication | $2 \times [3, 4]$ | $[6, 8]$ |
| Dot product | $[1, 2] \cdot [3, 4]$ | $1 \times 3 + 2 \times 4 = 11$ |

Every embedding, every layer input/output is a vector.

---

## 7. Matrices (Grids of Numbers)

A **matrix** is a 2D grid:

```python
M = [[1, 2, 3],
     [4, 5, 6]]    # a 2×3 matrix
```

**Matrix-vector multiplication** — the `linear()` function in microgpt.py:

$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 14 \\ 32 \end{bmatrix}$$

Each row's dot product with the input gives one output element.

---

## 8. Probability

A **probability distribution** assigns a number between 0 and 1 to each outcome, with all probabilities summing to 1:

$$P(\text{a}) = 0.3, \quad P(\text{b}) = 0.5, \quad P(\text{c}) = 0.2 \quad \implies \quad \text{Sum} = 1.0$$

**Random sampling:** Choosing an outcome where each option's chance equals its probability.

---

## 9. Square Root ($\sqrt{\phantom{x}}$)

$$\sqrt{4} = 2, \quad \sqrt{9} = 3, \quad \sqrt{2} \approx 1.414$$

In code: `x ** 0.5` or `math.sqrt(x)`

!!! info "Where it appears"

    - `head_dim**0.5` — scaling in attention
    - `v_hat ** 0.5` — in Adam optimizer

---

## 10. Mean (Average)

$$\text{mean}([2, 4, 6]) = \frac{2 + 4 + 6}{3} = 4$$

In code: `sum(x) / len(x)`

!!! info "Where it appears"

    `sum(xi * xi for xi in x) / len(x)` — mean of squares in RMSNorm.

---

That's all the math. Every formula in microgpt.py uses only these concepts.
