# Parameters Are Knowledge

## The Problem

We have an autograd engine that can compute gradients. But gradients of *what*? We need actual numbers to compute with — the model's **parameters**.

Parameters are the thousands of numbers that the model will **tune during training** to get good at prediction. Before training, they're random. After training, they encode everything the model has "learned."

## The Hyperparameters (Lines 75–79)

```python title="microgpt.py — Lines 75-79"
n_embd = 16     # embedding dimension
n_head = 4      # number of attention heads
n_layer = 1     # number of layers
block_size = 8  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head = 4
```

These are **hyperparameters** — settings that the programmer chooses, not things the model learns:

| Hyperparameter | Value | What it controls |
|:--------------:|:-----:|------------------|
| `n_embd` | 16 | How "rich" each token's representation is |
| `n_head` | 4 | How many different "perspectives" in attention |
| `n_layer` | 1 | How many times we repeat the attention+MLP block |
| `block_size` | 8 | Maximum number of characters the model can see |
| `head_dim` | 4 | Size of each attention head ($16 / 4 = 4$) |

!!! info "Scale comparison"

    In real GPT models, these numbers are much larger (GPT-2: `n_embd=768, n_head=12, n_layer=12`). The *structure* is identical.

## Creating Parameter Matrices (Line 80)

```python title="microgpt.py — Line 80"
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
```

This helper creates a 2D grid (matrix) of `Value` objects, each initialized with a small random number from a Gaussian distribution:

$$\text{random.gauss}(0, 0.02) \implies \text{a random number near 0, usually between } -0.06 \text{ and } +0.06$$

!!! example "Example: `matrix(3, 2)`"

    ```python
    [
      [Value(0.01), Value(-0.03)],   # row 0
      [Value(0.02), Value(0.01)],    # row 1
      [Value(-0.01), Value(0.04)],   # row 2
    ]
    ```

    A 3×2 grid of random `Value` objects.

=== "Why random?"

    If all parameters started at the same value, they'd all receive the same gradient and update in lockstep forever. Randomness breaks this symmetry.

=== "Why small (std=0.02)?"

    Large initial values cause numerical instability. Starting near zero is safe.

=== "Why Gaussian?"

    Draws values from a bell curve centered at 0. Most values are close to 0, rarely far from it.

## The State Dictionary (Lines 81–89)

```python title="microgpt.py — Lines 81-89"
state_dict = {
    'wte': matrix(vocab_size, n_embd),   # token embeddings: 27 × 16
    'wpe': matrix(block_size, n_embd),    # position embeddings: 8 × 16
    'lm_head': matrix(vocab_size, n_embd), # output layer: 27 × 16
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)    # 16 × 16
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)    # 16 × 16
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)    # 16 × 16
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)  # 16 × 16
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # 64 × 16
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0)  # 16 × 64
```

| Name | Shape | Purpose |
|------|:-----:|---------|
| `wte` | 27 × 16 | **Token embedding** — gives each of 27 tokens a 16-dimensional "meaning" |
| `wpe` | 8 × 16 | **Position embedding** — encodes position (1st, 2nd, ..., 8th) |
| `lm_head` | 27 × 16 | **Output layer** — converts internal state back to token predictions |
| `attn_wq` | 16 × 16 | **Query** weights for attention |
| `attn_wk` | 16 × 16 | **Key** weights for attention |
| `attn_wv` | 16 × 16 | **Value** weights for attention (not `Value` class — confusing, but standard terminology) |
| `attn_wo` | 16 × 16 | **Output** projection for attention |
| `mlp_fc1` | 64 × 16 | **Expand** layer in the MLP block (16 → 64) |
| `mlp_fc2` | 16 × 64 | **Compress** layer in the MLP block (64 → 16) |

!!! warning "Why `std=0` for some matrices?"

    `attn_wo` and `mlp_fc2` are initialized with `std=0` — **all zeros**. These are output projection matrices. Initializing them to zero means the attention and MLP blocks initially do nothing (they output zeros, so the residual connection just passes the input through). This is a stability trick for training.

## Flattening the Parameters (Line 89)

```python title="microgpt.py — Line 89"
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
```

This flattens all matrices into a single flat list of `Value` objects. The optimizer needs one flat list to loop over all parameters.

### How many parameters?

| Matrix | Shape | Count |
|--------|:-----:|------:|
| `wte` | 27 × 16 | 432 |
| `wpe` | 8 × 16 | 128 |
| `lm_head` | 27 × 16 | 432 |
| `attn_wq` | 16 × 16 | 256 |
| `attn_wk` | 16 × 16 | 256 |
| `attn_wv` | 16 × 16 | 256 |
| `attn_wo` | 16 × 16 | 256 |
| `mlp_fc1` | 64 × 16 | 1,024 |
| `mlp_fc2` | 16 × 64 | 1,024 |
| **Total** | | **4,064** |

4,064 `Value` objects, each a small random number, each tracking its gradient. By comparison, GPT-2 has 124 million parameters, and GPT-4 is rumored to have over a trillion.

??? note "Terminology"

    | Term | Meaning |
    |------|---------|
    | **Parameters** | The learnable numbers in the model (weights and biases) |
    | **Hyperparameters** | Settings chosen by the programmer (`n_embd`, `n_head`, etc.) |
    | **State dict** | A dictionary mapping names to parameter matrices |
    | **Weight matrix** | A 2D grid of parameters used in a linear transformation |
    | **Initialization** | The strategy for setting initial parameter values |
    | **Gaussian** | A bell-curve distribution; most values cluster near the mean |
