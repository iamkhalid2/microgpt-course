# Normalization (RMSNorm)

## The Problem

As data flows through layers of linear transformations, the numbers can **drift** — becoming very large or very small. This causes two problems:

1. **Exploding values:** Numbers get so large that `exp()` overflows → training crashes
2. **Vanishing values:** Numbers get so small that they're effectively zero → model stops learning

We need a way to keep the numbers "well-behaved" — centered and of moderate magnitude.

## RMS Normalization

`microgpt.py` uses **RMSNorm** (Root Mean Square Normalization), a simplified version of the more common LayerNorm.

### The Formula

```
RMSNorm(x) = x / RMS(x)

where RMS(x) = √(mean(x²))
             = √((x₀² + x₁² + ... + xₙ₋₁²) / n)
```

In words: divide each element by the "average magnitude" of all elements.

### Step by Step

```
x = [3.0, 4.0, 0.0]

Step 1: Square each element
  [9.0, 16.0, 0.0]

Step 2: Mean of squares
  (9.0 + 16.0 + 0.0) / 3 = 8.333

Step 3: Square root (the RMS)
  √8.333 = 2.887

Step 4: Divide each element by RMS
  [3.0/2.887, 4.0/2.887, 0.0/2.887]
  = [1.039, 1.386, 0.0]
```

The values are now "normalized" — they have a consistent scale regardless of how large or small the original values were.

## The Code (Lines 103–106)

```python
# Lines 103-106 of microgpt.py
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

### Line 104: Mean of squares

```python
ms = sum(xi * xi for xi in x) / len(x)
```

Compute `xi²` for each element, sum them, divide by count. This is the "mean square" (MS in RMS).

### Line 105: The scale factor

```python
scale = (ms + 1e-5) ** -0.5
```

This is `1 / √(ms + ε)`:
- `** -0.5` means "raise to the power of -0.5" = "1 divided by the square root"
- `1e-5` (= 0.00001) is a tiny epsilon (ε) to prevent division by zero if all values are 0

### Line 106: Apply scaling

```python
return [xi * scale for xi in x]
```

Multiply each element by the scale factor. Equivalent to dividing by the RMS.

### Why multiply instead of divide?

Writing `xi * scale` (where `scale = 1/√ms`) is mathematically identical to `xi / √ms`. Computing the reciprocal once and multiplying is slightly more efficient than dividing each element individually.

## Why RMSNorm Instead of LayerNorm?

GPT-2 originally used **LayerNorm**, which also subtracts the mean:

```
LayerNorm: (x - mean(x)) / std(x)
RMSNorm:   x / RMS(x)
```

RMSNorm skips the mean subtraction. Research showed it works nearly as well with less computation. Karpathy uses it here for simplicity.

## Where RMSNorm Is Used

```python
# Line 112
x = rmsnorm(x)              # After combining embeddings

# Line 117
x = rmsnorm(x)              # Before attention

# Line 137
x = rmsnorm(x)              # Before MLP
```

RMSNorm is applied **before each major block** (attention and MLP). This is called **pre-normalization** — it stabilizes the input to each block.

## The Effect Visually

```
Before normalization:     [100.0, -200.0, 50.0, 0.3]
                          (wildly different magnitudes)

After normalization:      [0.77, -1.54, 0.39, 0.002]
                          (well-behaved, similar magnitudes)
```

## Terminology

| Term | Meaning |
|------|---------|
| **Normalization** | Scaling values to have consistent magnitude |
| **RMSNorm** | Dividing by the root mean square of the values |
| **LayerNorm** | Subtract mean, divide by standard deviation (more complex) |
| **Epsilon (ε)** | A tiny number (1e-5) added to prevent division by zero |
| **Pre-normalization** | Normalizing before (not after) each block |

## Next

Now we're ready for the main event: [attention](./05-attention.md) — the mechanism that lets the model look at previous characters to decide the next one.
