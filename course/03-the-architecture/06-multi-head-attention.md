# Multi-Head Attention

## The Problem

A single attention mechanism computes one set of attention weights — one "perspective" on which tokens are relevant. But different aspects of language require **different types of attention**:

- One perspective might focus on **adjacent characters** (common letter pairs like "th", "er")
- Another might focus on the **first character** (names starting with certain letters tend to end certain ways)
- Another might focus on **repeated patterns** ("mm" → the 'a' often follows)

One attention "head" can only learn one of these perspectives. We need multiple.

## The Solution: Multiple Heads

Split the 16-dimensional Q, K, V vectors into **4 independent groups of 4 dimensions each**. Each group runs its own attention — its own queries, keys, values, and weights. Then concatenate the results.

```
Full Q (16 dims): [q₀, q₁, q₂, q₃, q₄, q₅, q₆, q₇, q₈, q₉, q₁₀, q₁₁, q₁₂, q₁₃, q₁₄, q₁₅]
                   ├── head 0 ──┤├── head 1 ──┤├── head 2 ──┤├── head 3 ──┤
                   [q₀, q₁, q₂, q₃]  [q₄, q₅, q₆, q₇]  ...
                   head_dim = 4
```

Each head:
- Has its own 4-dimensional slice of Q, K, V
- Computes its own attention scores and weights
- Produces a 4-dimensional output

Then all 4 outputs are **concatenated** back to 16 dimensions.

## The Code (Lines 123–132)

```python
# Lines 123-132 of microgpt.py
x_attn = []
for h in range(n_head):                    # n_head = 4
    hs = h * head_dim                       # head start index: 0, 4, 8, 12
    q_h = q[hs:hs+head_dim]                # slice of Q for this head
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]   # slice of each K
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]  # slice of each V

    # Scaled dot-product attention (same as previous lesson, but per head)
    attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                   for t in range(len(k_h))]
    attn_weights = softmax(attn_logits)
    head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)]
    x_attn.extend(head_out)                 # concatenate head outputs
```

### Walking through head 0:

```
hs = 0 × 4 = 0

q_h = q[0:4]           → the first 4 dimensions of q
k_h = [k₀[0:4], k₁[0:4], k₂[0:4]]   → first 4 dims of each past key
v_h = [v₀[0:4], v₁[0:4], v₂[0:4]]   → first 4 dims of each past value

→ attention on dimensions 0-3
→ produces 4-dimensional output
→ x_attn.extend(head_out) appends these 4 values
```

### After all 4 heads:

```
x_attn = [head0_out(4), head1_out(4), head2_out(4), head3_out(4)]
       = 16 dimensions total
```

## Output Projection (Line 133)

```python
# Line 133 of microgpt.py
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
```

After concatenating the 4 heads, one more linear layer mixes them together. This lets the model combine information from different attention perspectives.

```
Head 0: focused on adjacent chars    → [0.3, -0.1, ...]
Head 1: focused on first char        → [0.1, 0.5, ...]
Head 2: focused on repeated chars    → [0.4, 0.2, ...]
Head 3: focused on something else    → [0.0, -0.3, ...]

concatenated: [0.3, -0.1, ..., 0.1, 0.5, ..., 0.4, 0.2, ..., 0.0, -0.3, ...]

linear(concatenated, wo): mixes all heads into final [y₀, y₁, ..., y₁₅]
```

## Why Split Instead of Using 4 Full-Size Heads?

If each of 4 heads used the full 16 dimensions:
- 4 × (16 × 16) = **1024** operations per attention step

By splitting 16 into 4 groups of 4:
- 4 × (4 × 4) = **64** operations per attention step

Same number of total parameters (the Q, K, V matrices are still 16×16), but the attention computations are 16× cheaper. And empirically, multiple small heads learn *better* than one big head.

## Visual Summary

```
        x (16 dims)
        │
  ┌─────┤ linear transformations ├─────┐
  │     │                              │
 Q(16) K(16) V(16)
  │     │     │
  ├──split into 4 heads──┤
  │     │     │     │
  H0    H1    H2    H3      (each 4 dims)
  │     │     │     │
  ├──concatenate──────────┤
  │                       │
  x_attn (16 dims)
  │
  linear (wo) → output (16 dims)
```

## Terminology

| Term | Meaning |
|------|---------|
| **Multi-head attention** | Running multiple attention heads in parallel on subsets of dimensions |
| **Head** | One independent attention mechanism |
| **head_dim** | The dimension of each head (n_embd / n_head) |
| **Concatenation** | Joining head outputs end-to-end |
| **Output projection** | A linear layer that mixes the concatenated heads |

## Next

But wait — we can't just replace the original `x` with the attention output. We need a way to **preserve the original information** while adding new information. That's what [residual connections](./07-residual-connections.md) do.
