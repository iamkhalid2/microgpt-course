# Softmax

## The Problem

A linear layer outputs a list of raw numbers called **logits**. These can be any values — positive, negative, huge, tiny:

```text
logits = [2.1, -0.5, 1.3, 0.8, -1.2, ...]
```

But we need **probabilities** — numbers between 0 and 1 that sum to 1. For example: "There's a 40% chance the next character is 'a', 30% chance it's 'e', etc."

## The Softmax Formula

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

In plain English:

1. Apply $e^x$ (exponential) to each logit → makes everything positive
2. Divide each by the total → makes everything sum to 1

### Step by Step

=== "1. Exponentiate"

    $$e^{2.0} = 7.389, \quad e^{1.0} = 2.718, \quad e^{0.1} = 1.105$$

=== "2. Sum"

    $$\text{total} = 7.389 + 2.718 + 1.105 = 11.212$$

=== "3. Divide"

    | Logit | $e^z$ | Probability |
    |:-----:|:-----:|:-----------:|
    | 2.0 | 7.389 | $7.389 / 11.212 = 0.659$ (65.9%) |
    | 1.0 | 2.718 | $2.718 / 11.212 = 0.242$ (24.2%) |
    | 0.1 | 1.105 | $1.105 / 11.212 = 0.099$ (9.9%) |
    | | **Total** | **1.000** :white_check_mark: |

The largest logit (2.0) gets the largest probability (65.9%). The exponential function **amplifies** differences.

## The Code (Lines 97–101)

```python title="microgpt.py — Lines 97-101"
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

!!! warning "Wait — what's `max_val` doing there?"

    Line 98 subtracts the maximum logit before exponentiating. This is the **numerical stability trick**.

    Without it:

    - $e^{1000} = \infty$ (overflow!)
    - $e^{-1000} = 0$ (underflow!)

    By subtracting the max, the largest value becomes 0, and $e^0 = 1$. No overflow.

    **The math is unchanged** — subtracting a constant from all logits doesn't change the ratios:

    $$\frac{e^{a-c}}{e^{a-c} + e^{b-c}} = \frac{e^a / e^c}{e^a / e^c + e^b / e^c} = \frac{e^a}{e^a + e^b}$$

## Properties of Softmax

| Property | Why it matters |
|----------|---------------|
| All outputs are positive | Probabilities can't be negative |
| Outputs sum to 1 | They represent a valid probability distribution |
| Preserves ordering | Largest logit → largest probability |
| Differentiable | We can compute gradients through it |

## Where Softmax Is Used

In `microgpt.py`, softmax appears in **two places**:

1. **Attention weights** (line 130): Converting attention scores into probabilities
   > "How much attention should I pay to each previous token?"

2. **Output prediction** (line 166): Converting final logits into character probabilities
   > "What's the probability of each possible next character?"

??? note "Terminology"

    | Term | Meaning |
    |------|---------|
    | **Logits** | Raw, unnormalized scores from a linear layer |
    | **Softmax** | Function that converts logits to probabilities |
    | **Probability distribution** | List of non-negative numbers that sum to 1 |
    | **Numerical stability** | Avoiding overflow/underflow by shifting values |
    | **$e^x$** | The exponential function ($\approx 2.718^x$) |
