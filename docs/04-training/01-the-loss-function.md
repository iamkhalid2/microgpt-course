# The Loss Function

## What Loss Measures

After the model predicts probabilities for the next character, we need a single number that says: "How wrong were you?"

That number is the **loss**. Lower = better.

## Cross-Entropy Loss

The loss function in `microgpt.py` is **cross-entropy loss**:

$$\text{loss} = -\log(P(\text{correct token}))$$

Just the negative log of the probability assigned to the **correct answer**.

### Why This Works

=== "Confident and correct"

    $$P(\text{correct}) = 0.9 \implies \text{loss} = -\log(0.9) = 0.105$$

    Low loss — the model knew the answer. :white_check_mark:

=== "Uncertain"

    $$P(\text{correct}) = 0.2 \implies \text{loss} = -\log(0.2) = 1.609$$

    Moderate loss — the model wasn't sure.

=== "Confident and wrong"

    $$P(\text{correct}) = 0.01 \implies \text{loss} = -\log(0.01) = 4.605$$

    High loss — the model was confident about the **wrong** answer. :x:

!!! warning

    Cross-entropy **heavily punishes** confident wrong predictions. Going from 0.01 to 0.001 adds more loss than going from 0.5 to 0.1. This forces the model not to be overconfident about wrong answers.

## The Loss Curve

$$y = -\log(x)$$

```text
loss
 5 │ *
   │  *
 4 │   *
   │    *
 3 │     *
   │       *
 2 │         *
   │            *
 1 │                *
   │                       *
 0 │──────────────────────────── *
   0    0.2   0.4   0.6   0.8   1.0
              P(correct)
```

| $P(\text{correct})$ | Loss | Interpretation |
|:---:|:---:|---|
| 1.0 | 0.0 | Perfect prediction |
| 0.5 | 0.693 | 50/50 guess |
| $1/27 \approx 0.037$ | 3.296 | Random chance (27 tokens) |
| 0.01 | 4.605 | Barely considers the correct answer |

## The Code (Lines 166–170)

```python title="microgpt.py — Lines 166-170"
logits = gpt(token_id, pos_id, keys, values)
probs = softmax(logits)
loss_t = -probs[target_id].log()
losses.append(loss_t)
```

=== "Line 166"

    `gpt()` returns 27 raw logits (scores).

=== "Line 167"

    `softmax()` converts logits to 27 probabilities summing to 1.

=== "Line 168"

    `probs[target_id]` grabs the probability of the **correct** next character. `.log()` computes the natural logarithm. The `-` sign makes it a positive loss.

=== "Line 169"

    Append this position's loss. We'll average all positions at the end.

## Averaging Over the Sequence (Line 171)

```python title="microgpt.py — Line 171"
loss = (1 / n) * sum(losses)
```

For a name like "emma" (5 positions), we compute loss at each position and average:

$$\text{loss} = \frac{1}{5}(\text{loss}_0 + \text{loss}_1 + \text{loss}_2 + \text{loss}_3 + \text{loss}_4)$$

!!! info "Why average?"

    Names have different lengths. Without averaging, longer names would have higher loss, biasing the model toward short names.

## What's the Initial Loss?

At step 0, the model assigns roughly equal probability to all 27 tokens:

$$P(\text{correct}) \approx \frac{1}{27} \implies \text{loss} \approx -\log\left(\frac{1}{27}\right) \approx 3.30$$

If your model's first loss is near 3.3, everything is working correctly. If it's much higher, something is wrong.

??? note "Terminology"

    | Term | Meaning |
    |------|---------|
    | **Loss** | A single number measuring prediction error (lower = better) |
    | **Cross-entropy** | $-\log(P(\text{correct}))$ — the standard loss for classification |
    | **Logits** | Raw scores before softmax |
    | **Target** | The correct next token |
    | **Average loss** | Mean loss over all positions in a sequence |
