# Normalization (RMSNorm)

## The Problem

As data flows through layers of linear transformations, the numbers can **drift** — becoming very large or very small. This causes two problems:

1. **Exploding values:** Numbers get so large that $e^x$ overflows → training crashes
2. **Vanishing values:** Numbers get so small that they're effectively zero → model stops learning

We need a way to keep the numbers "well-behaved."

## RMS Normalization

`microgpt.py` uses **RMSNorm** (Root Mean Square Normalization), a simplified version of the more common LayerNorm.

### The Formula

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}$$

$$\text{where } \text{RMS}(\mathbf{x}) = \sqrt{\frac{x_0^2 + x_1^2 + \cdots + x_{n-1}^2}{n}}$$

In words: divide each element by the "average magnitude" of all elements.

### Step by Step

!!! example "Example: $\mathbf{x} = [3.0, 4.0, 0.0]$"

    | Step | Computation | Result |
    |:----:|-------------|:------:|
    | 1. Square each element | $[9.0, 16.0, 0.0]$ | |
    | 2. Mean of squares | $(9.0 + 16.0 + 0.0) / 3$ | $8.333$ |
    | 3. Square root (RMS) | $\sqrt{8.333}$ | $2.887$ |
    | 4. Divide each by RMS | $[3.0/2.887, 4.0/2.887, 0.0/2.887]$ | $[1.039, 1.386, 0.0]$ |

    The values now have a consistent scale regardless of the original magnitudes.

## The Code (Lines 103–106)

```python title="microgpt.py — Lines 103-106"
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

=== "Line 104: Mean of squares"

    ```python
    ms = sum(xi * xi for xi in x) / len(x)
    ```

    Compute $x_i^2$ for each element, sum them, divide by count. This is the "mean square" (**MS** in RM**S**).

=== "Line 105: Scale factor"

    ```python
    scale = (ms + 1e-5) ** -0.5
    ```

    This computes $\frac{1}{\sqrt{ms + \epsilon}}$:

    - `** -0.5` = "1 divided by the square root"
    - `1e-5` ($= 0.00001$) is a tiny epsilon ($\epsilon$) to prevent division by zero

=== "Line 106: Apply"

    ```python
    return [xi * scale for xi in x]
    ```

    Multiply each element by the scale factor. Equivalent to dividing by the RMS.

    !!! note

        Writing `xi * scale` (where $\text{scale} = 1/\sqrt{ms}$) is identical to `xi / √ms`. Computing the reciprocal once and multiplying is slightly more efficient.

## RMSNorm vs LayerNorm

| | RMSNorm | LayerNorm |
|---|---------|-----------|
| **Formula** | $x / \text{RMS}(x)$ | $(x - \mu) / \sigma$ |
| **Centers at zero?** | No | Yes (subtracts mean) |
| **Used in** | LLaMA, microgpt.py | GPT-2, BERT |
| **Advantage** | Simpler, less computation | Slightly more stable |

RMSNorm skips the mean subtraction. Research showed it works nearly as well with less computation.

## Where RMSNorm Is Used

```python
# Line 112 — after combining embeddings
x = rmsnorm(x)

# Line 117 — before attention
x = rmsnorm(x)

# Line 137 — before MLP
x = rmsnorm(x)
```

!!! info "Pre-normalization"

    RMSNorm is applied **before each major block** (attention and MLP). This is called pre-normalization — it stabilizes the input to each block.

??? note "Terminology"

    | Term | Meaning |
    |------|---------|
    | **Normalization** | Scaling values to have consistent magnitude |
    | **RMSNorm** | Dividing by the root mean square of the values |
    | **LayerNorm** | Subtract mean, divide by standard deviation |
    | **Epsilon ($\epsilon$)** | A tiny number (`1e-5`) to prevent division by zero |
    | **Pre-normalization** | Normalizing before (not after) each block |
