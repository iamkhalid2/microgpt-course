# Linear Layers

## The Problem

We have a 16-dimensional embedding vector `x`. Now we need to **transform** it — mix the information in its 16 dimensions to produce a new vector. This is the most fundamental computation in neural networks.

## What Is a Linear Layer?

A linear layer multiplies an input vector by a weight matrix. It's the neural network's way of "mixing and recombining" information.

### Concrete Example

Let's start with a tiny 3→2 example (3 inputs, 2 outputs):

```
Input:  x = [x₀, x₁, x₂]

Weight matrix W (2 rows × 3 columns):
  W = [[w₀₀, w₀₁, w₀₂],
       [w₁₀, w₁₁, w₁₂]]

Output: y = [y₀, y₁]
  y₀ = w₀₀×x₀ + w₀₁×x₁ + w₀₂×x₂
  y₁ = w₁₀×x₀ + w₁₁×x₁ + w₁₂×x₂
```

Each output element is a **weighted sum** of all input elements. The weights determine "how much of each input goes into each output."

### With numbers:

```
x = [2.0, 3.0, 1.0]

W = [[0.5, -0.3, 0.1],
     [0.2,  0.4, -0.2]]

y₀ = 0.5×2 + (-0.3)×3 + 0.1×1 = 1.0 - 0.9 + 0.1 = 0.2
y₁ = 0.2×2 + 0.4×3 + (-0.2)×1 = 0.4 + 1.2 - 0.2 = 1.4

y = [0.2, 1.4]
```

## The Code (Lines 94–95)

```python
# Lines 94-95 of microgpt.py
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

Let's unpack this nested comprehension:

```python
# For each row 'wo' in weight matrix 'w':
for wo in w:
    # Compute the dot product of that row with input 'x':
    sum(wi * xi for wi, xi in zip(wo, x))
```

Step by step for our example:
```
Row 0: wo = [0.5, -0.3, 0.1]
       sum(0.5×2, -0.3×3, 0.1×1) = 0.2

Row 1: wo = [0.2, 0.4, -0.2]
       sum(0.2×2, 0.4×3, -0.2×1) = 1.4

Result: [0.2, 1.4]
```

### What's a Dot Product?

The inner sum is called a **dot product** — you multiply corresponding elements and add them up:

```
dot([a, b, c], [d, e, f]) = a×d + b×e + c×f
```

It's a measure of "similarity" between two vectors. High dot product = vectors point in the same direction. Low dot product = they don't.

## Visual Representation

```
Input x (len 3):     [x₀ , x₁ , x₂]
                       │    │    │
                       ↓    ↓    ↓
Weight row 0:     [w₀₀, w₀₁, w₀₂] ──[dot product]──▶ y₀
Weight row 1:     [w₁₀, w₁₁, w₁₂] ──[dot product]──▶ y₁

Output y (len 2):     [y₀ , y₁]
```

Each row of the weight matrix produces one output element. The number of rows determines the output size.

## Where Linear Layers Appear in microgpt.py

| Line | Usage | Input size → Output size |
|------|-------|--------------------------|
| 118 | `linear(x, attn_wq)` | 16 → 16 (compute queries) |
| 119 | `linear(x, attn_wk)` | 16 → 16 (compute keys) |
| 120 | `linear(x, attn_wv)` | 16 → 16 (compute values) |
| 133 | `linear(x_attn, attn_wo)` | 16 → 16 (output projection) |
| 138 | `linear(x, mlp_fc1)` | 16 → 64 (expand) |
| 140 | `linear(x, mlp_fc2)` | 64 → 16 (compress) |
| 143 | `linear(x, lm_head)` | 16 → 27 (final prediction) |

The `linear` function is the workhorse — used 7 times per layer, plus one more for the output.

## Why "Linear"?

Because the transformation `y = Wx` is a **linear function** — if you double the input, you double the output. There are no curves, no bends. Just scaling and mixing.

This is both the strength and the limitation:
- **Strength:** Easy to compute gradients through
- **Limitation:** Can only represent straight-line relationships

To model complex patterns, we need **non-linearity** — that's what activation functions like ReLU provide (see [the MLP block](./08-the-mlp-block.md)).

## No Bias Terms

In standard neural networks, linear layers often add a **bias**: `y = Wx + b`. Karpathy's implementation skips biases entirely. This is a simplification (and common in modern Transformer architectures). The model can still learn any pattern; biases just give a slight head start.

## Terminology

| Term | Meaning |
|------|---------|
| **Linear layer** | Multiplying input by a weight matrix; y = Wx |
| **Dot product** | Multiply elements pairwise, then sum |
| **Weight matrix** | The grid of learnable parameters in a linear layer |
| **Bias** | An optional additive vector (omitted in microgpt.py) |
| **Projection** | Another word for "linear transformation" |

## Next

A linear layer gives us raw numbers, but we often need **probabilities** (values between 0 and 1 that sum to 1). That's what [softmax](./03-softmax.md) does.
