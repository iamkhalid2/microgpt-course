# The Training Loop

## The Complete Code (Lines 151–184)

```python title="microgpt.py — Lines 151-184"
# ── SETUP ──
num_steps = 500

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

## Tracing Step 0 with "emma"

=== "1. Sample"

    ```text
    doc = "emma"
    tokens = [26, 4, 12, 12, 0, 26]     # [BOS, 'e', 'm', 'm', 'a', BOS]
    n = min(8, 6 - 1) = 5               # 5 predictions to make
    ```

=== "2. Forward + Loss"

    | Position | Input | Target | Loss |
    |:--------:|:-----:|:------:|:----:|
    | 0 | BOS (26) | 'e' (4) | 3.29 |
    | 1 | 'e' (4) | 'm' (12) | 3.33 |
    | 2 | 'm' (12) | 'm' (12) | 3.30 |
    | 3 | 'm' (12) | 'a' (0) | 3.31 |
    | 4 | 'a' (0) | BOS (26) | 3.28 |
    | **Avg** | | | **3.302** |

    At step 0, the loss is ~3.3. For a random model with 27 tokens, the expected loss is $-\log(1/27) \approx 3.30$. :white_check_mark: The model is at random chance.

=== "3. Backward"

    ```python
    loss.backward()  # every parameter now has .grad set
    ```

=== "4. Update"

    ```python
    lr_t = 0.01 × 0.5 × (1 + cos(0)) = 0.01   # full LR at step 0
    ```

    For each of 4,064 parameters: update `m[i]`, `v[i]`, nudge `p.data`, reset `p.grad = 0`.

=== "5. Log"

    ```text
    step    1 / 500 | loss 3.3020
    ```

## The Arc of Training

| Step | What the model has learned |
|:----:|--------------------------|
| 1 | Nothing. Random predictions. |
| 50 | Common characters predicted more often |
| 100 | Frequent letter pairs (th, er, an) |
| 200 | Name-like structures (consonant-vowel patterns) |
| 300 | When names should end (predicts BOS) |
| 500 | Reasonable name generation capability |

## Data Cycling

```python
doc = docs[step % len(docs)]
```

The `%` (modulo) operator cycles through the dataset. With ~32,000 names and 500 steps, we only see ~1.5% of the data. A longer training run would cycle through more.

## Block Size Truncation

```python
n = min(block_size, len(tokens) - 1)
```

If a name is longer than `block_size` (8), we truncate. Most names are shorter than 8 characters, so this rarely matters.

!!! success "Checkpoint ✓"

    You now understand the **entire training process**:

    - :white_check_mark: Sampling a document and tokenizing it
    - :white_check_mark: Forward pass: running the model on each token
    - :white_check_mark: Loss: measuring prediction quality with cross-entropy
    - :white_check_mark: Backward pass: computing all gradients automatically
    - :white_check_mark: Adam optimizer: updating parameters with momentum and adaptation
    - :white_check_mark: Cosine learning rate decay
