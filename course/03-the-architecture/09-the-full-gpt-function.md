# The Full GPT Function

## Putting It All Together

We've studied every component. Now let's see the complete `gpt()` function — all of them assembled into a single pipeline that takes a token and produces predictions.

## The Code (Lines 108–144)

```python
# Lines 108-144 of microgpt.py
def gpt(token_id, pos_id, keys, values):
    # Step 1: Embed
    tok_emb = state_dict['wte'][token_id]       # token embedding lookup
    pos_emb = state_dict['wpe'][pos_id]          # position embedding lookup
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # combine: what + where
    x = rmsnorm(x)                                # normalize

    for li in range(n_layer):                     # for each layer (just 1 in our case)
        # ─────────────────────────────────────────
        # 1) Multi-head attention block
        # ─────────────────────────────────────────
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])    # query
        k = linear(x, state_dict[f'layer{li}.attn_wk'])    # key
        v = linear(x, state_dict[f'layer{li}.attn_wv'])    # value
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                           for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                        for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]         # residual

        # ─────────────────────────────────────────
        # 2) MLP block
        # ─────────────────────────────────────────
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])    # expand 16→64
        x = [xi.relu() ** 2 for xi in x]                    # activate
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])    # compress 64→16
        x = [a + b for a, b in zip(x, x_residual)]         # residual

    logits = linear(x, state_dict['lm_head'])               # project to vocab
    return logits
```

## The Data Flow

Let's trace what happens to a single token, step by step:

```
Input: token_id = 4 ('e'), pos_id = 0 (first position)

Step 1: Embedding
  tok_emb = wte[4]        → 16 numbers representing 'e'
  pos_emb = wpe[0]        → 16 numbers representing "first position"
  x = tok_emb + pos_emb   → 16 numbers: "'e' at position 0"
  x = rmsnorm(x)          → 16 numbers, normalized

Step 2: Attention Block
  x_residual = x           (save for later)
  x = rmsnorm(x)           (normalize again)
  q, k, v = linear(x, Wq), linear(x, Wk), linear(x, Wv)   → 16 each
  For each of 4 heads:
    Compute attention weights over all past tokens
    Weighted sum of values → 4 numbers
  Concatenate → 16 numbers
  x = linear(concat, Wo) → 16 numbers
  x = x + x_residual      (residual connection)

Step 3: MLP Block
  x_residual = x           (save for later)
  x = rmsnorm(x)
  x = linear(x, fc1)      → 64 numbers (expanded)
  x = ReLU(x)²            → 64 numbers (activated)
  x = linear(x, fc2)      → 16 numbers (compressed)
  x = x + x_residual      (residual connection)

Step 4: Output
  logits = linear(x, lm_head) → 27 numbers (one per possible character)

Return: logits (27 raw scores)
```

## The Architecture Diagram

```
                    token_id    pos_id
                       │           │
                    ┌──┘           └──┐
                    ▼                 ▼
               ┌─────────┐     ┌─────────┐
               │   wte   │     │   wpe   │
               │ (embed) │     │ (embed) │
               └────┬────┘     └────┬────┘
                    │               │
                    └───── + ───────┘
                           │
                      ┌────┴────┐
                      │ rmsnorm │
                      └────┬────┘
                           │
              ┌────────────┼─────────────────┐
              │            │  LAYER 0         │
              │   ┌────────┴────────┐         │
              │   │    Attention    │         │
              │   │  (4 heads)     │         │
              │   └────────┬────────┘         │
              │            │                  │
              │      ┌─────┴─────┐            │
              ├──────┤    +      │ (residual) │
              │      └─────┬─────┘            │
              │            │                  │
              │   ┌────────┴────────┐         │
              │   │      MLP       │         │
              │   │ (expand→act→   │         │
              │   │  compress)     │         │
              │   └────────┬────────┘         │
              │            │                  │
              │      ┌─────┴─────┐            │
              ├──────┤    +      │ (residual) │
              │      └─────┬─────┘            │
              └────────────┼─────────────────┘
                           │
                  ┌────────┴────────┐
                  │    lm_head     │
                  │ (linear 16→27) │
                  └────────┬────────┘
                           │
                           ▼
                  logits (27 scores)
```

## What the Logits Mean

The output is 27 numbers — one for each token in the vocabulary:

```
logits[0]  → raw score for 'a'
logits[1]  → raw score for 'b'
...
logits[25] → raw score for 'z'
logits[26] → raw score for <BOS>
```

These are **not probabilities yet.** They're raw scores that can be negative or very large. To get probabilities, we apply softmax (which happens outside this function, in the training loop).

## Stateless Design

Notice that `gpt()` takes `keys` and `values` as parameters and **modifies them in place** by appending new K/V pairs. This makes the function "stateless" in the sense that all state is external — the function itself doesn't store anything between calls.

This design enables the KV cache pattern: process tokens one at a time, accumulating context in the K/V lists.

## Why "GPT"?

**GPT** = **G**enerative **P**re-trained **T**ransformer

- **Generative:** It generates text (one token at a time)
- **Pre-trained:** It's trained on data before being used
- **Transformer:** The architecture — attention + MLP + residual connections

This function IS the Transformer. That's it. The rest is training and inference.

## Scaling Up

This is a tiny GPT with `n_layer=1` and `n_embd=16`. Real GPTs just increase the numbers:

| | microgpt.py | GPT-2 Small | GPT-3 |
|---|---|---|---|
| n_embd | 16 | 768 | 12,288 |
| n_head | 4 | 12 | 96 |
| n_layer | 1 | 12 | 96 |
| block_size | 8 | 1024 | 2048 |
| Parameters | 4,064 | 124M | 175B |

Same architecture. Same code. Just bigger matrices.

## Checkpoint ✓

You now understand the **entire model architecture**:
- ✅ Parameters: random numbers that encode knowledge
- ✅ Embeddings: representing tokens and positions as vectors
- ✅ Linear layers: mixing information via matrix multiplication
- ✅ Softmax: converting scores to probabilities
- ✅ RMSNorm: keeping values well-behaved
- ✅ Attention: deciding which tokens to focus on
- ✅ Multi-head: multiple attention perspectives
- ✅ Residual connections: preserving original information
- ✅ MLP: non-linear processing and knowledge storage
- ✅ Full GPT: all pieces assembled

## Next

The model can now take a token and produce predictions. But the predictions are garbage because the parameters are random. Time to **train**: [Module 4: Training](../04-training/00-what-is-training.md).
