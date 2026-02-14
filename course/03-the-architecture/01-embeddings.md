# Embeddings

## The Problem

We have token IDs: `'e'` = 4, `'m'` = 12, etc. But a single number (like 4) doesn't carry much information. The model needs a richer representation.

Consider: is `'e'` (4) more similar to `'d'` (3) than to `'z'` (25)? No — the numeric ordering is arbitrary. But with a single number, the model has no way to tell.

## The Solution: Embedding Vectors

Instead of representing each token as a single number, we represent it as a **list of numbers** (a vector). Each token gets its own 16-dimensional vector:

```
'a' (token 0)  → [0.01, -0.03, 0.02, ..., 0.04]   (16 numbers)
'b' (token 1)  → [-0.02, 0.01, -0.01, ..., 0.03]   (16 numbers)
'e' (token 4)  → [0.03, 0.02, -0.04, ..., -0.01]   (16 numbers)
...
```

These 16 numbers capture a token's "meaning" in a way the model can work with. After training, tokens with similar roles will have similar vectors.

## The Code (Lines 109–111)

```python
# Lines 109-111 of microgpt.py (inside the gpt function)
tok_emb = state_dict['wte'][token_id]  # token embedding
pos_emb = state_dict['wpe'][pos_id]    # position embedding
x = [t + p for t, p in zip(tok_emb, pos_emb)]  # combined embedding
```

### Line 109: Token Embedding

```python
tok_emb = state_dict['wte'][token_id]
```

`state_dict['wte']` is a 27×16 matrix — the **token embedding table**:

```
Token 0 ('a'):  [v₀₀, v₀₁, v₀₂, ..., v₀₁₅]
Token 1 ('b'):  [v₁₀, v₁₁, v₁₂, ..., v₁₁₅]
Token 2 ('c'):  [v₂₀, v₂₁, v₂₂, ..., v₂₁₅]
...
Token 26 (BOS): [v₂₆₀, v₂₆₁, ..., v₂₆₁₅]
```

For `token_id = 4` (the letter 'e'), we just do a table lookup: grab row 4. The result is a list of 16 `Value` objects — the embedding for 'e'.

This is not a computation, just a **lookup**. But the values in this table are parameters — they'll be updated during training.

### Line 110: Position Embedding

```python
pos_emb = state_dict['wpe'][pos_id]
```

`state_dict['wpe']` is an 8×16 matrix — the **position embedding table**:

```
Position 0: [p₀₀, p₀₁, ..., p₀₁₅]    "being first"
Position 1: [p₁₀, p₁₁, ..., p₁₁₅]    "being second"
...
Position 7: [p₇₀, p₇₁, ..., p₇₁₅]    "being eighth"
```

Why do we need this? Because the model processes one token at a time, but **position matters**. The letter 'e' at position 0 (first letter of a name) is very different from 'e' at position 4 (middle of a name).

Without position information, the model would treat every occurrence of 'e' identically regardless of where it appears.

### Line 111: Combine

```python
x = [t + p for t, p in zip(tok_emb, pos_emb)]
```

We **add** the token embedding and position embedding element-wise:

```
tok_emb  = [t₀,  t₁,  t₂,  ..., t₁₅]   (what token is this?)
pos_emb  = [p₀,  p₁,  p₂,  ..., p₁₅]   (where is it?)
x        = [t₀+p₀, t₁+p₁, ..., t₁₅+p₁₅] (combined: what + where)
```

Why addition (not concatenation)? It's simpler and works well in practice. Both embeddings live in the same 16-dimensional space, and their sum captures both "what" and "where."

## Visual Summary

```
token_id = 4 ('e')
pos_id = 2 (third position)

         wte (27 × 16)                wpe (8 × 16)
┌─────────────────────┐         ┌─────────────────────┐
│ [row 0: 'a']        │         │ [row 0: pos 0]      │
│ [row 1: 'b']        │         │ [row 1: pos 1]      │
│ [row 2: 'c']        │         │ [row 2: pos 2] ◀──── pos_emb
│ [row 3: 'd']        │         │ ...                  │
│ [row 4: 'e'] ◀───── tok_emb  │ [row 7: pos 7]      │
│ ...                  │         └─────────────────────┘
│ [row 26: BOS]        │
└─────────────────────┘

x = tok_emb + pos_emb  (element-wise addition)
  = 16 Value objects representing "'e' at position 2"
```

## Why 16 Dimensions?

The choice of 16 is a hyperparameter (`n_embd`). More dimensions = more expressive power, but also more parameters and slower computation. For our tiny names dataset, 16 is sufficient.

In GPT-2, `n_embd = 768`. Each token is a 768-dimensional vector. That's a much richer representation, needed for understanding complex language.

## Terminology

| Term | Meaning |
|------|---------|
| **Embedding** | A vector (list of numbers) representing a token |
| **Embedding table** | A matrix where each row is one token's embedding |
| **Token embedding** | Encodes *what* the token is |
| **Position embedding** | Encodes *where* the token is in the sequence |
| **Lookup** | Selecting a row from a table by index (no arithmetic) |
| **Dimension** | The number of elements in an embedding vector |

## Next

We have a 16-dimensional vector `x` representing our token. Now, how do we transform it? That's what [linear layers](./02-linear-layers.md) do.
