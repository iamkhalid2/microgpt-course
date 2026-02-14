# Temperature and Sampling

## The Problem

When generating text, the model outputs probabilities for the next token. But how "creative" should the model be?

- **Too predictable:** Always picking the most likely token → boring, repetitive output
- **Too random:** Picking tokens nearly uniformly → nonsensical output

The **temperature** parameter controls this tradeoff.

## What Temperature Does

```python
# Line 195 of microgpt.py
probs = softmax([l / temperature for l in logits])
```

Before softmax, each logit is **divided by the temperature**:

```
adjusted_logit = logit / temperature
```

### Temperature = 1.0 (Normal)

```
logits = [2.0, 1.0, 0.1]
logits / 1.0 = [2.0, 1.0, 0.1]
probs = [0.659, 0.242, 0.099]    (same as original)
```

### Temperature = 0.5 (Sharper — less creative)

```
logits = [2.0, 1.0, 0.1]
logits / 0.5 = [4.0, 2.0, 0.2]     ← differences AMPLIFIED
probs = [0.869, 0.117, 0.014]       ← the best option dominates
```

### Temperature = 2.0 (Flatter — more creative)

```
logits = [2.0, 1.0, 0.1]
logits / 2.0 = [1.0, 0.5, 0.05]    ← differences DAMPENED
probs = [0.424, 0.257, 0.319]       ← more uniform
```

## The Pattern

```
Temperature → 0:  Always picks the most likely token (deterministic)
Temperature = 1:  Standard probabilities (balanced)
Temperature → ∞:  All tokens equally likely (pure random)
```

Visually:

```
Probability
    │
0.9 │ ██                                 temp = 0.5 (sharp)
    │ ██
0.7 │ ██
    │ ██
0.5 │ ██ ▓▓                              temp = 1.0 (normal)
    │ ██ ▓▓
0.3 │ ██ ▓▓ ░░   ░░ ░░ ░░               temp = 2.0 (flat)
    │ ██ ▓▓ ░░   ░░ ░░ ░░
0.1 │ ██ ▓▓ ░░   ░░ ░░ ░░
    └──┴──┴──┴───┴──┴──┴──
       a   b   c    d   e
```

## Why Temperature = 0.5 in microgpt.py?

```python
# Line 187 of microgpt.py
temperature = 0.5
```

A temperature of 0.5 makes the model **fairly confident** — it strongly favors high-probability characters. This produces more "realistic" names. Higher temperatures would produce more creative but potentially nonsensical combinations.

## The Sampling Step

```python
# Line 196 of microgpt.py
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

`random.choices(population, weights)` selects one item from `population`, where each item's chance of being selected is proportional to its weight.

```
population = [0, 1, 2, 3, ..., 26]    (all token IDs)
weights = [0.01, 0.003, ...]           (probability of each token)
```

Even with temperature = 0.5, there's still randomness. The most likely character is *usually* picked, but not always. This is what makes each generation unique.

## Alternatives to Random Sampling

| Method | How it works | Result |
|--------|-------------|--------|
| **Greedy** | Always pick the highest probability token | Deterministic, repetitive |
| **Random sampling** | Pick based on probabilities (what we do) | Varied, sometimes odd |
| **Top-k sampling** | Only consider the top k most likely tokens | Less randomness |
| **Nucleus (top-p)** | Consider tokens until cumulative probability reaches p | Adaptive k |

`microgpt.py` uses simple random sampling with temperature. It's the most straightforward approach.

## Why "Temperature"?

The name comes from **statistical mechanics** in physics. In a physical system:
- At high temperature, particles move randomly (high entropy)
- At low temperature, particles settle into ordered states (low entropy)

The same applies to our probability distribution:
- High temperature → high entropy → more randomness
- Low temperature → low entropy → more order

## Terminology

| Term | Meaning |
|------|---------|
| **Temperature** | A scalar that controls randomness in generation |
| **Sharpening** | Low temperature makes the distribution more peaked |
| **Flattening** | High temperature makes the distribution more uniform |
| **Greedy decoding** | Always choosing the most likely token |
| **Entropy** | A measure of randomness/uncertainty in a distribution |

## Next

Let's tie everything together in [the complete picture](./02-the-complete-picture.md).
