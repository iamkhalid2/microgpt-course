# Softmax

## The Problem

A linear layer outputs a list of raw numbers called **logits**. These can be any values — positive, negative, huge, tiny:

```
logits = [2.1, -0.5, 1.3, 0.8, -1.2, ...]
```

But we need **probabilities** — numbers between 0 and 1 that sum to 1. For example: "There's a 40% chance the next character is 'a', 30% chance it's 'e', etc."

How do we convert raw scores into probabilities?

## The Softmax Formula

```
softmax(logits)ᵢ = exp(logitsᵢ) / Σ exp(logitsⱼ)
```

In plain English:
1. Apply `e^x` (exponential) to each logit → makes everything positive
2. Divide each by the total → makes everything sum to 1

### Step by Step

```
logits = [2.0, 1.0, 0.1]

Step 1: Exponentiate
  exp(2.0) = 7.389
  exp(1.0) = 2.718
  exp(0.1) = 1.105

Step 2: Sum
  total = 7.389 + 2.718 + 1.105 = 11.212

Step 3: Divide
  prob₀ = 7.389 / 11.212 = 0.659   (65.9%)
  prob₁ = 2.718 / 11.212 = 0.242   (24.2%)
  prob₂ = 1.105 / 11.212 = 0.099   (9.9%)
                              ─────
                              1.000   ✓ (sums to 1)
```

The largest logit (2.0) gets the largest probability (65.9%). The exponential function **amplifies** differences.

## The Code (Lines 97–101)

```python
# Lines 97-101 of microgpt.py
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

### Wait — what's `max_val` doing there?

Line 98 subtracts the maximum logit before exponentiating:

```python
max_val = max(val.data for val in logits)
exps = [(val - max_val).exp() for val in logits]
```

This is the **numerical stability trick**. Without it:
- `exp(1000)` = ∞ (overflow!)
- `exp(-1000)` = 0 (underflow!)

By subtracting the max, the largest value becomes 0, and `exp(0) = 1`. No overflow.

**The math is unchanged:** subtracting a constant from all logits doesn't change the ratios:

```
Before: exp(a) / (exp(a) + exp(b))
After:  exp(a-c) / (exp(a-c) + exp(b-c))
      = exp(a)/exp(c) / (exp(a)/exp(c) + exp(b)/exp(c))
      = exp(a) / (exp(a) + exp(b))  ← same!
```

### Line by Line

```python
# Example: logits = [Value(2.0), Value(1.0), Value(0.1)]

max_val = 2.0                           # the largest logit value

exps = [(val - 2.0).exp() for val in logits]
     = [exp(0.0), exp(-1.0), exp(-1.9)]
     = [1.0, 0.368, 0.150]

total = 1.0 + 0.368 + 0.150 = 1.518

result = [1.0/1.518, 0.368/1.518, 0.150/1.518]
       = [0.659, 0.242, 0.099]          # same probabilities as before ✓
```

## Properties of Softmax

| Property | Why it matters |
|----------|---------------|
| All outputs are positive | Probabilities can't be negative |
| Outputs sum to 1 | They represent a valid probability distribution |
| Preserves ordering | Largest logit → largest probability |
| Differentiable | We can compute gradients through it |

## Where Softmax Is Used

In `microgpt.py`, softmax appears in two places:

1. **Attention weights** (line 130): Converting attention scores into probabilities
   ```python
   attn_weights = softmax(attn_logits)
   ```
   "How much attention should I pay to each previous token?"

2. **Output prediction** (line 166): Converting final logits into character probabilities
   ```python
   probs = softmax(logits)
   ```
   "What's the probability of each possible next character?"

## Terminology

| Term | Meaning |
|------|---------|
| **Logits** | Raw, unnormalized scores from a linear layer |
| **Softmax** | Function that converts logits to probabilities |
| **Probability distribution** | List of non-negative numbers that sum to 1 |
| **Numerical stability** | Avoiding overflow/underflow by shifting values |
| **exp(x)** | e^x ≈ 2.718^x — the exponential function |

## Next

Now that we can produce probabilities, we need a way to keep numbers from growing too large or too small inside the network. That's what [normalization](./04-normalization.md) does.
