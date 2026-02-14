# The MLP Block

## The Problem

Attention lets tokens **communicate** — each token can gather information from other tokens. But attention is a **linear** operation (weighted sums). It can only compute linear combinations of existing information.

To learn complex patterns (like "after 'qu', the next letter is usually a vowel"), the model needs **non-linear processing** — the ability to compute things that aren't just weighted averages.

## The MLP (Multi-Layer Perceptron)

The MLP is a two-layer network sandwiched around a non-linear activation:

```
16 dims → [Expand to 64] → [Activation] → [Compress to 16] → 16 dims
```

Think of it as:
1. **Expand:** Spread the information into a much wider space (16 → 64)
2. **Think:** Apply non-linear processing (ReLU²)
3. **Compress:** Condense back to the original size (64 → 16)

## The Code (Lines 135–141)

```python
# Lines 135-141 of microgpt.py

# 2) MLP block
x_residual = x                                      # save for residual
x = rmsnorm(x)                                      # normalize
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])     # expand: 16 → 64
x = [xi.relu() ** 2 for xi in x]                    # activation: ReLU²
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])     # compress: 64 → 16
x = [a + b for a, b in zip(x, x_residual)]          # residual connection
```

### Line 137: Normalize

```python
x = rmsnorm(x)
```

Same RMSNorm as before — keep the values well-behaved before processing.

### Line 138: Expand (16 → 64)

```python
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])   # mlp_fc1 is 64 × 16
```

The first linear layer multiplies the 16-dim input by a 64×16 matrix, producing 64 outputs. This **expansion** (4×) gives the model more room to compute complex features.

Why 4×? It's a convention from the original Transformer paper. The expansion ratio of 4 has been found to work well across many tasks.

### Line 139: Activation (ReLU²)

```python
x = [xi.relu() ** 2 for xi in x]
```

This is the key non-linear step. It applies `ReLU(x)²`:

```
ReLU²(x) = (max(0, x))²
         = x²  if x > 0
         = 0   if x ≤ 0
```

Visually:

```
     output                output
      │   /                 │   .·
      │  /      ReLU        │  /    ReLU²
      │ /                   │ /
──────┼────── x         ────┼────── x
      │                     │
      │                     │

    Linear zero gate      Smoother, squared
```

Why `ReLU²` instead of plain `ReLU`?
- Smoother gradient near zero (no sudden kink)
- Empirically works well for language models
- The original GPT-2 used GeLU (another smooth activation); `ReLU²` is simpler and similar

Why **not** just a linear function? Because two linear layers in sequence are equivalent to a single linear layer. The non-linearity between them is what lets the MLP compute complex, non-linear functions.

### Line 140: Compress (64 → 16)

```python
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])   # mlp_fc2 is 16 × 64
```

Project back down from 64 to 16 dimensions. The model had to "decide" what's worth keeping in just 16 numbers.

### Line 141: Residual Connection

```python
x = [a + b for a, b in zip(x, x_residual)]
```

Add the original input back. Same pattern as with attention.

## What Does the MLP Actually Do?

Researchers have found that MLP layers in Transformers act as **"memory banks"**:

- The first layer's weights (64×16) contain **keys** — patterns to match against
- The activation function acts as a **gate** — turning off irrelevant patterns
- The second layer's weights (16×64) contain **values** — the information to inject when a pattern matches

It's like a lookup table:
```
"If the input looks like [pattern A]" → inject [knowledge A]
"If the input looks like [pattern B]" → inject [knowledge B]
...
```

## Visual Summary

```
     x (16 dims)
     │
     ├──── x_residual (saved)
     │
  [rmsnorm]
     │
  [linear 16→64]     "expand"
     │
  [ReLU²]            "gate and transform"
     │
  [linear 64→16]     "compress"
     │
     + x_residual     "add original back"
     │
     ▼
    new x (16 dims)
```

## Terminology

| Term | Meaning |
|------|---------|
| **MLP** | Multi-Layer Perceptron — a two-layer feedforward network |
| **Activation function** | A non-linear function between layers (here: ReLU²) |
| **Expansion ratio** | Factor by which the hidden dimension expands (4× here) |
| **Feedforward** | Information flows in one direction (no loops) |
| **Non-linearity** | Any function that isn't f(x) = ax + b |

## Next

We've now covered every component. Let's see how they all fit together in the [full GPT function](./09-the-full-gpt-function.md).
