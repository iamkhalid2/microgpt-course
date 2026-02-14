# Residual Connections

## The Problem

The attention block transforms `x` into something new. But what if the transformation *loses* important information that was in the original `x`?

Imagine you're editing a document: you highlight a paragraph and replace it entirely. If the replacement is bad, everything from the original is lost. But if you *add a comment* next to the original, you keep both.

## The Solution: Add, Don't Replace

A **residual connection** (or "skip connection") is embarrassingly simple:

```
output = input + transformation(input)
```

Instead of:
```
x → [attention] → new_x          (information lost)
```

We do:
```
x → [attention] → (attention_output + x) → new_x    (original preserved)
```

## The Code (Lines 116, 134)

```python
# Line 116
x_residual = x                                    # save original

# ... attention computation happens here (lines 117-133) ...

# Line 134
x = [a + b for a, b in zip(x, x_residual)]        # add original back
```

And again for the MLP block:

```python
# Line 136
x_residual = x                                    # save original

# ... MLP computation happens here (lines 137-140) ...

# Line 141
x = [a + b for a, b in zip(x, x_residual)]        # add original back
```

Each element of the output is: `output[i] = transformed[i] + original[i]`.

## Why This Works

### 1. Gradient Highway

During backward pass, the gradient needs to flow from the loss all the way back to early parameters. Without residual connections, the gradient passes through *every* operation and can shrink to nearly zero (**vanishing gradients**).

The addition creates a **shortcut** for the gradient:

```
Without residual:   gradient must flow through: gpt → attn → norm → embed
                    gradient shrinks at each step

With residual:      gradient has a DIRECT PATH through the addition
                    d(a + b)/da = 1  ← the gradient flows straight through!
```

Since the derivative of addition w.r.t. its inputs is 1, the gradient passes through *unchanged*. This is like a highway for gradients.

### 2. Starting from Identity

Remember that `attn_wo` and `mlp_fc2` are initialized to zero. This means:
- At the start of training, the attention block outputs zeros
- The residual connection means `x = zeros + x = x`
- **The model starts as the identity function** — it just passes the input through

The model then *gradually* learns to add useful transformations on top. This is much more stable than starting with random transformations.

### 3. Preserving Information

The original information is never lost. Each block can only **add** new information. If the attention block learns nothing useful, it can output zeros, and the input passes through unchanged.

## Visual

```
     x ────────────────────────────────┐
     │                                  │ (skip connection)
     ▼                                  │
  ┌──────────┐                         │
  │ RMSNorm  │                         │
  │ Attention│    attention output      │
  └──────────┘          │              │
                        ▼              ▼
                     ┌──────┐
                     │  +   │  ← element-wise addition
                     └──────┘
                        │
                        ▼
                    new x (has both original and new info)
```

## Terminology

| Term | Meaning |
|------|---------|
| **Residual connection** | Adding the input to the output: y = x + f(x) |
| **Skip connection** | Same thing — the input "skips over" the transformation |
| **Vanishing gradients** | Gradients shrinking to near-zero in deep networks |
| **Identity function** | f(x) = x — output equals input (what residuals default to) |

## Next

We've covered the attention block and its residual connection. Now let's look at the other half of each Transformer layer: the [MLP block](./08-the-mlp-block.md).
