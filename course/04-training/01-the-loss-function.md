# The Loss Function

## The Problem

The model produced a probability distribution over 27 characters. The target (correct answer) is one specific character. We need a **single number** that says "how wrong was the prediction?"

This number is the **loss**. Lower is better.

## Cross-Entropy Loss

The loss function used in `microgpt.py` is **cross-entropy loss**, which boils down to:

```
loss = -log(probability assigned to the correct answer)
```

That's it. Just one line of math.

### Intuition

If the model assigns a **high probability** to the correct character:
```
Correct answer: 'm'
Model says: P('m') = 0.9 (90%)
Loss = -log(0.9) = 0.105    ← small loss (good!)
```

If the model assigns a **low probability** to the correct character:
```
Correct answer: 'm'
Model says: P('m') = 0.01 (1%)
Loss = -log(0.01) = 4.605   ← large loss (bad!)
```

If the model is **perfectly confident** and correct:
```
P('m') = 1.0 (100%)
Loss = -log(1.0) = 0        ← perfect (zero loss)
```

### The -log curve

```
loss
  5 │·
    │ ·
  4 │  ·
    │   ·
  3 │    ·
    │     ·
  2 │      ·
    │        ·
  1 │          ·
    │             ·
  0 │                 · · · ·
    └───┬───┬───┬───┬───┬───── probability
      0.0 0.2 0.4 0.6 0.8 1.0
```

The loss:
- Goes to **infinity** as probability approaches 0 (extremely wrong → extremely high penalty)
- Equals **0** when probability is 1 (perfect prediction → no penalty)
- Drops steeply for low probabilities (the model is heavily punished for being confidently wrong)

## The Code (Lines 160–169)

```python
# Lines 160-169 of microgpt.py
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
losses = []
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits)
    loss_t = -probs[target_id].log()
    losses.append(loss_t)
loss = (1 / n) * sum(losses)
```

### Line 161: Initialize KV cache

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
```

Fresh KV cache for each training example. Start with no context.

### Lines 163-164: Get input and target

```python
token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
```

For the name "emma" (tokens = `[BOS, 'e', 'm', 'm', 'a', BOS]`):

```
pos  | token_id (input)  | target_id (correct answer)
─────┼───────────────────┼──────────────────────────
  0  | BOS (26)          | 'e' (4)
  1  | 'e'  (4)          | 'm' (12)
  2  | 'm' (12)          | 'm' (12)
  3  | 'm' (12)          | 'a' (0)
  4  | 'a'  (0)          | BOS (26)
```

### Line 165-166: Forward pass and softmax

```python
logits = gpt(token_id, pos_id, keys, values)
probs = softmax(logits)
```

Run the model, get probabilities for all 27 characters.

### Line 167: Compute loss for this position

```python
loss_t = -probs[target_id].log()
```

Index into the probability vector to get the probability assigned to the *correct* character, take -log.

For example, if `target_id = 12` ('m') and `probs[12].data = 0.05`:
```
loss_t = -log(0.05) = 2.996
```

### Line 168-169: Average over the sequence

```python
losses.append(loss_t)
loss = (1 / n) * sum(losses)
```

We compute the loss at every position (BOS→e, e→m, m→m, m→a, a→BOS) and average them. This gives us one single number for the entire name.

## Why -log? Why Not Something Simpler?

You might ask: why not just use `loss = 1 - probability`?

Two reasons:

1. **Infinite punishment for zero probability:** If the model assigns 0% to the correct answer, that's catastrophically wrong. `-log(0) = ∞` captures this severity. `1 - 0 = 1` would understate it.

2. **Information theory:** `-log(p)` has a beautiful interpretation — it measures the "surprise" in bits. A low probability event is surprising; a high probability event is expected.

## The Comment: "May yours be low"

Line 169 in the original:
```python
loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.
```

A blessing from Karpathy. Low loss means the model is predicting well. It's the one number you watch during training to know if things are working.

## Terminology

| Term | Meaning |
|------|---------|
| **Loss** | A single number measuring how wrong the model's predictions were |
| **Cross-entropy loss** | `-log(P(correct answer))` — the standard loss for classification |
| **Logits** | Raw model outputs before softmax |
| **Target** | The correct answer the model should have predicted |
| **Average loss** | Loss averaged over all positions in the sequence |

## Next

We have the loss. Now we need to compute gradients — [backpropagation](./02-backpropagation.md) through the entire computation graph.
