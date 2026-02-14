# Attention

## The Problem

When predicting the next character in a name, the model needs to consider **all the characters it has seen so far**. For example, in "emm_":

- Seeing 'e' at position 0 might suggest one set of likely next characters
- Seeing 'm' repeated at positions 1 and 2 might suggest a different set
- The model needs to **combine information from all previous positions** to make its prediction

How does the model decide which previous characters are relevant and how much to "listen" to each one?

## The Attention Intuition

Think of attention as a **question-answering** system:

```
Current token ('m' at position 2) asks:
  "Hey, previous tokens — who has information I need?"

Token 'e' (position 0) responds:
  "I'm somewhat relevant — I can tell you about the beginning of the name"

Token 'm' (position 1) responds:
  "I'm very relevant — I'm the same letter as you, and letter pairs matter!"

The model then listens proportionally:
  30% attention to 'e', 60% attention to 'm', 10% to itself
```

This "attention" is computed using three concepts: **Queries**, **Keys**, and **Values**.

## Queries, Keys, and Values

Think of it like a library:
- **Query (Q):** "I'm looking for X" (what the current token is *asking* for)
- **Key (K):** "I contain X" (what each token *advertises* about itself)
- **Value (V):** "Here's my actual information" (the content each token actually provides)

The process:
1. The current token creates a **query** — "what am I looking for?"
2. Each previous token has a **key** — "what do I offer?"
3. Compare the query to each key (dot product) to get **attention scores** — "how relevant is each token?"
4. Convert scores to probabilities (softmax) → **attention weights**
5. Use the weights to take a weighted average of the **values** — the actual information

## The Code (Lines 118–133)

Let's go through the attention block step by step:

### Step 1: Compute Q, K, V (Lines 118–122)

```python
# Lines 118-122 of microgpt.py
q = linear(x, state_dict[f'layer{li}.attn_wq'])  # query
k = linear(x, state_dict[f'layer{li}.attn_wk'])  # key
v = linear(x, state_dict[f'layer{li}.attn_wv'])  # value
keys[li].append(k)
values[li].append(v)
```

- `x` is the current token's embedding (16 numbers)
- Three linear layers transform `x` into Q, K, and V (each 16 numbers)
- K and V are **appended** to lists (`keys[li]`, `values[li]`) that accumulate across positions

Why separate linear layers for Q, K, V? Because they serve different roles:
- **Q** encodes "what I'm looking for" (trained to ask the right questions)
- **K** encodes "what I have to offer" (trained to advertise relevant info)
- **V** encodes "the actual content" (the information to pass forward)

### Step 2: Compute Attention Scores (Line 129)

```python
# Line 129 of microgpt.py
attn_logits = [
    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
    for t in range(len(k_h))
]
```

For each previous token `t`, compute:
```
score(t) = dot(q, k_t) / √(head_dim)
         = (q₀×k₀ + q₁×k₁ + ... + q₃×k₃) / √4
```

The **dot product** measures how similar the query is to each key:
- High dot product → "This token has what I'm looking for!"
- Low dot product → "Not relevant"

The **division by √(head_dim)** is called **scaled attention**. Without it, dot products grow too large (proportional to the dimension), which would push softmax into extreme probabilities. The scaling keeps things balanced.

### Step 3: Convert to Probabilities (Line 130)

```python
# Line 130 of microgpt.py
attn_weights = softmax(attn_logits)
```

The softmax we learned earlier. Converts raw scores into probabilities that sum to 1:

```
scores = [2.1, 0.5, 1.3]
weights = softmax(scores) = [0.56, 0.11, 0.33]
```

Now we know how much attention to pay to each previous token.

### Step 4: Weighted Sum of Values (Line 131)

```python
# Line 131 of microgpt.py
head_out = [
    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
    for j in range(head_dim)
]
```

For each dimension `j` of the output:
```
out[j] = Σ attn_weights[t] × v[t][j]
```

This is a **weighted average** — tokens with high attention weights contribute more to the output.

## Full Example

Let's trace attention with 3 tokens at positions 0, 1, 2:

```
Position 0: 'e'  →  k₀ = [0.2, -0.1, 0.3, 0.0],  v₀ = [1.0, 0.5, -0.2, 0.3]
Position 1: 'm'  →  k₁ = [0.1, 0.4, -0.2, 0.1],  v₁ = [0.3, -0.1, 0.7, 0.2]
Position 2: 'm'  →  q₂ = [0.3, 0.3, 0.1, -0.1]  (current query)

Step 1: Dot products (head_dim = 4, √4 = 2):
  score₀ = (0.3×0.2 + 0.3×(-0.1) + 0.1×0.3 + (-0.1)×0.0) / 2 = 0.06/2 = 0.03
  score₁ = (0.3×0.1 + 0.3×0.4 + 0.1×(-0.2) + (-0.1)×0.1) / 2 = 0.12/2 = 0.06
  score₂ = (0.3×0.3 + 0.3×0.3 + 0.1×0.1 + (-0.1)×(-0.1)) / 2 = 0.20/2 = 0.10

Step 2: Softmax:
  weights = softmax([0.03, 0.06, 0.10]) = [0.316, 0.333, 0.351]

Step 3: Weighted sum of values:
  out[0] = 0.316×1.0 + 0.333×0.3 + 0.351×0.3 = 0.620
  out[1] = 0.316×0.5 + 0.333×(-0.1) + 0.351×0.5 = 0.300
  ... (for all 4 dimensions)
```

## The KV Cache

Notice lines 121–122:
```python
keys[li].append(k)
values[li].append(v)
```

The keys and values are **accumulated** as we process each token. When processing position 2, we already have keys and values from positions 0 and 1. This is the **KV cache** — it avoids recomputing K and V for past positions.

## Causal Masking (Built-in!)

In `microgpt.py`, causal masking happens *automatically* because we process tokens one at a time. When computing attention at position 2, the `keys` list only contains positions 0, 1, and 2 — future tokens haven't been added yet. The model literally can't attend to the future.

In batch implementations (like PyTorch), you need an explicit mask to prevent cheating. Here, the sequential processing does it naturally.

## Terminology

| Term | Meaning |
|------|---------|
| **Attention** | A mechanism for deciding how much to "listen" to each previous token |
| **Query (Q)** | "What am I looking for?" — the current token's request |
| **Key (K)** | "What do I have to offer?" — each token's advertisement |
| **Value (V)** | "Here's my content" — the actual information to share |
| **Attention scores** | Dot products between Q and K — raw relevance measures |
| **Attention weights** | Scores after softmax — probabilities summing to 1 |
| **Scaled attention** | Dividing by √(head_dim) to keep scores moderate |
| **KV cache** | Storing past keys and values to avoid recomputation |
| **Causal masking** | Preventing the model from seeing future tokens |

## Next

We've been looking at a single "head" of attention. But `microgpt.py` uses **4 heads** that attend to different things simultaneously. Let's see how [multi-head attention](./06-multi-head-attention.md) works.
