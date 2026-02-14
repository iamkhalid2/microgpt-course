# The Training Loop

## The Complete Code (Lines 151–184)

Here's the full training loop with every piece annotated:

```python
# Lines 151-184 of microgpt.py

# ── SETUP ──
num_steps = 500                         # how many training steps

for step in range(num_steps):

    # ── 1. SAMPLE ──
    doc = docs[step % len(docs)]        # pick a name (cycle through dataset)
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]  # tokenize
    n = min(block_size, len(tokens) - 1) # cap at block_size (8)

    # ── 2. FORWARD PASS + LOSS ──
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)   # model prediction
        probs = softmax(logits)                          # to probabilities
        loss_t = -probs[target_id].log()                 # cross-entropy loss
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)        # average loss over the sequence

    # ── 3. BACKWARD PASS ──
    loss.backward()                     # compute all gradients

    # ── 4. PARAMETER UPDATE (Adam) ──
    lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0                      # reset gradient

    # ── 5. LOG ──
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")
```

## One Complete Training Step, Step by Step

Let's trace step 0 with the name "emma":

### 1. Sample

```
doc = "emma"
tokens = [26, 4, 12, 12, 0, 26]     # [BOS, 'e', 'm', 'm', 'a', BOS]
n = min(8, 6 - 1) = 5               # 5 predictions to make
```

### 2. Forward Pass + Loss

```
Position 0: Input BOS(26), Target 'e'(4)
  logits = gpt(26, 0, ...)       → 27 raw scores
  probs = softmax(logits)        → 27 probabilities
  loss_0 = -log(probs[4])        → e.g., 3.29 (model barely predicted 'e')

Position 1: Input 'e'(4), Target 'm'(12)
  logits = gpt(4, 1, ...)        → 27 scores (now with KV from pos 0)
  probs = softmax(logits)
  loss_1 = -log(probs[12])       → e.g., 3.33

Position 2: Input 'm'(12), Target 'm'(12)
  logits = gpt(12, 2, ...)
  loss_2 = -log(probs[12])       → e.g., 3.30

Position 3: Input 'm'(12), Target 'a'(0)
  logits = gpt(12, 3, ...)
  loss_3 = -log(probs[0])        → e.g., 3.31

Position 4: Input 'a'(0), Target BOS(26)
  logits = gpt(0, 4, ...)
  loss_4 = -log(probs[26])       → e.g., 3.28

loss = (3.29 + 3.33 + 3.30 + 3.31 + 3.28) / 5 = 3.302
```

At step 0, the loss is ~3.3. For a random model with 27 tokens, the expected loss is `-log(1/27) ≈ 3.30`. ✓ The model is at random chance.

### 3. Backward Pass

```python
loss.backward()
```

After this, every parameter has its `.grad` computed.

### 4. Parameter Update

```python
lr_t = 0.01 × 0.5 × (1 + cos(0)) = 0.01 × 0.5 × 2 = 0.01   # full LR at step 0
```

For each of the 4,064 parameters, Adam updates `m[i]`, `v[i]`, and nudges `p.data`. Then resets `p.grad = 0`.

### 5. Log

```
step    1 / 500 | loss 3.3020
```

### Repeat 499 more times.

## The Arc of Training

```
Epoch     | What the model has learned
──────────┼──────────────────────────────────────────────
Step 1    | Nothing. Random predictions.
Step 50   | Common characters are predicted more often.
Step 100  | Frequent letter pairs (th, er, an) are learned.
Step 200  | Name-like structures emerge (consonant-vowel patterns).
Step 300  | The model knows when names should end (predicts BOS).
Step 500  | Reasonable name generation capability.
```

## Data Cycling

```python
doc = docs[step % len(docs)]
```

The `%` (modulo) operator cycles through the dataset. With 32,000 names and 500 steps, we only see 500 names — about 1.5% of the data. A longer training run would cycle through more.

## Block Size Truncation

```python
n = min(block_size, len(tokens) - 1)
```

If a name is longer than `block_size` (8), we truncate it. Most names are shorter than 8 characters, so this rarely matters.

## Checkpoint ✓

You now understand the **entire training process**:
- ✅ Sampling a document and tokenizing it
- ✅ Forward pass: running the model on each token
- ✅ Loss: measuring prediction quality with cross-entropy
- ✅ Backward pass: computing all gradients automatically
- ✅ Adam optimizer: updating parameters with momentum and adaptation
- ✅ Cosine learning rate decay

## Next

The model is trained. Now let's **use it**: [Module 5: Inference](../05-inference/00-generating-text.md).
