# Temperature and Sampling

## The Problem

When generating text, the model outputs probabilities for the next token. But how "creative" should the model be?

- **Too predictable:** Always picking the most likely token → boring, repetitive
- **Too random:** Picking tokens nearly uniformly → nonsensical

The **temperature** parameter controls this tradeoff.

## What Temperature Does

```python title="microgpt.py — Line 195"
probs = softmax([l / temperature for l in logits])
```

Before softmax, each logit is **divided by the temperature**:

$$\text{adjusted logit} = \frac{\text{logit}}{\text{temperature}}$$

=== "Temperature = 1.0 (Normal)"

    ```text
    logits = [2.0, 1.0, 0.1]
    logits / 1.0 = [2.0, 1.0, 0.1]
    probs = [0.659, 0.242, 0.099]    ← same as original
    ```

=== "Temperature = 0.5 (Sharper)"

    ```text
    logits = [2.0, 1.0, 0.1]
    logits / 0.5 = [4.0, 2.0, 0.2]     ← differences AMPLIFIED
    probs = [0.869, 0.117, 0.014]       ← the best option dominates
    ```

    Less creative — strongly favors the top choice.

=== "Temperature = 2.0 (Flatter)"

    ```text
    logits = [2.0, 1.0, 0.1]
    logits / 2.0 = [1.0, 0.5, 0.05]    ← differences DAMPENED
    probs = [0.424, 0.257, 0.319]       ← more uniform
    ```

    More creative — gives weaker options a better chance.

## The Pattern

| Temperature | Effect | Result |
|:-----------:|--------|--------|
| → 0 | Probabilities become one-hot | Always picks the most likely token |
| = 1 | Standard probabilities | Balanced |
| → ∞ | Probabilities become uniform | Pure random |

!!! info "Why temperature = 0.5 in microgpt.py?"

    A temperature of 0.5 makes the model **fairly confident** — it strongly favors high-probability characters. This produces more "realistic" names. Higher temperatures produce more creative but potentially nonsensical combinations.

## The Sampling Step

```python title="microgpt.py — Line 196"
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

`random.choices(population, weights)` selects one item where each item's chance is proportional to its weight.

Even with temperature = 0.5, there's still randomness. The most likely character is *usually* picked, but not always. This is what makes each generation **unique**.

## Alternative Sampling Methods

| Method | How it works | Tradeoff |
|--------|-------------|----------|
| **Greedy** | Always pick highest probability | Deterministic, repetitive |
| **Random** | Pick based on probabilities (what we do) | Varied, sometimes odd |
| **Top-k** | Only consider the top k most likely tokens | Less randomness |
| **Nucleus (top-p)** | Consider tokens until cumulative prob reaches p | Adaptive k |

`microgpt.py` uses simple random sampling with temperature — the most straightforward approach.

!!! tip "Why is it called 'temperature'?"

    From **statistical mechanics** in physics:

    - High temperature → particles move randomly (high entropy)
    - Low temperature → particles settle into ordered states (low entropy)

    Same idea: high temperature = more randomness, low temperature = more order.

??? note "Terminology"

    | Term | Meaning |
    |------|---------|
    | **Temperature** | A scalar that controls randomness in generation |
    | **Sharpening** | Low temperature makes the distribution more peaked |
    | **Flattening** | High temperature makes the distribution more uniform |
    | **Greedy decoding** | Always choosing the most likely token |
    | **Entropy** | A measure of randomness/uncertainty in a distribution |
