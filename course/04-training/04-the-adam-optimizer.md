# The Adam Optimizer

## The Problem with Plain Gradient Descent

Gradient descent uses the same learning rate for every parameter and has no memory of past gradients. Adam fixes both.

## What Adam Adds

**Adam** (Adaptive Moment Estimation) maintains two extra values for **each parameter**:

1. **m (first moment):** A running average of past gradients → **momentum**
2. **v (second moment):** A running average of past squared gradients → **adaptive learning rate**

## The Code (Lines 146–182)

### Setup (Lines 147–149)

```python
# Lines 147-149 of microgpt.py
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
m = [0.0] * len(params)  # first moment buffer (momentum)
v = [0.0] * len(params)  # second moment buffer (adaptive rates)
```

| Value | What it is | Typical range |
|-------|-----------|---------------|
| `learning_rate = 0.01` | Base step size | 1e-4 to 1e-2 |
| `beta1 = 0.9` | How much past momentum to keep | 0.9 - 0.99 |
| `beta2 = 0.95` | How much past variance to keep | 0.95 - 0.999 |
| `eps_adam = 1e-8` | Tiny number to prevent division by zero | 1e-8 |
| `m[i] = 0.0` | Momentum for parameter i (starts at zero) | |
| `v[i] = 0.0` | Variance for parameter i (starts at zero) | |

### The Update (Lines 175–182)

```python
# Lines 175-182 of microgpt.py
lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
for i, p in enumerate(params):
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad           # update momentum
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2      # update variance
    m_hat = m[i] / (1 - beta1 ** (step + 1))              # bias correction
    v_hat = v[i] / (1 - beta2 ** (step + 1))              # bias correction
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)     # update parameter
    p.grad = 0                                              # reset gradient
```

Let's understand each line.

### Line 175: Cosine Learning Rate Decay

```python
lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
```

The learning rate isn't constant — it follows a **cosine schedule**:

```
lr
0.01 │*
     │ *
     │  *
     │   *
0.005│    *
     │      *
     │        *
     │           *
   0 │              *
     └───┬───┬───┬───┬───
        0  125 250 375 500   step
```

- Starts at `learning_rate` (0.01)
- Smoothly decreases to 0 by the end of training
- Why? Large steps early for fast learning, tiny steps late for fine-tuning

The formula: `cos(π × step/total)` goes from cos(0)=1 to cos(π)=-1. So `0.5×(1+cos(...))` goes from 1 to 0.

### Line 177: Update Momentum (First Moment)

```python
m[i] = beta1 * m[i] + (1 - beta1) * p.grad
     = 0.9 × m[i] + 0.1 × p.grad
```

This is an **exponential moving average** of the gradient:
- Keep 90% of the previous momentum
- Add 10% of the current gradient

**Why?** Individual gradients are noisy (computed from a single name). Averaging over history smooths out the noise and gives a more reliable direction.

It's like a rolling ball — instead of changing direction every instant, it has **inertia**.

```
Without momentum:           With momentum:
      ↗                          ↗
    ↙                          →
      ↗                          →
    ↙     (zigzag!)              →  (smooth!)
      ↗                          →
```

### Line 178: Update Variance (Second Moment)

```python
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
     = 0.95 × v[i] + 0.05 × p.grad²
```

An exponential moving average of the **squared** gradient.

**Why?** This tracks how "noisy" or "variable" each parameter's gradient is:
- Parameters with consistently large gradients → large v → smaller effective step
- Parameters with small gradients → small v → larger effective step

This is the **adaptive** part — each parameter gets its own effective learning rate.

### Lines 179–180: Bias Correction

```python
m_hat = m[i] / (1 - beta1 ** (step + 1))
v_hat = v[i] / (1 - beta2 ** (step + 1))
```

Since `m` and `v` are initialized to 0, they're **biased toward zero** in the early steps. Bias correction inflates them to compensate.

At step 0:
```
m_hat = m[0] / (1 - 0.9¹) = m[0] / 0.1 → multiplied by 10
```

By step 100:
```
m_hat = m[100] / (1 - 0.9¹⁰¹) ≈ m[100] / 1.0 → essentially no correction
```

The correction matters most at the start and fades out over time.

### Line 181: The Actual Update

```python
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
```

Breaking this down:
```
p.data -= lr_t × (smoothed gradient) / (√(gradient variance) + ε)
```

- `m_hat` = smoothed gradient (direction + momentum)
- `v_hat ** 0.5` = √(gradient variance) (adaptive scaling)
- The division: parameters with high variance get **smaller** steps (cautious), parameters with low variance get **larger** steps (confident)
- `eps_adam` prevents division by zero

### Line 182: Reset Gradient

```python
p.grad = 0
```

Zero out the gradient for the next step. Essential, because `backward()` uses `+=`.

## Adam vs. SGD: A Visual Comparison

```
SGD:    Same learning rate for all parameters
        ──▶ ──▶ ──▶ ──▶  (may zigzag)

Adam:   Adapts per parameter + has momentum
        ─────────────────▶  (smooth, efficient path)
```

## The Formula in One Line

```
parameter -= lr × (smoothed_gradient / √smoothed_squared_gradient)
                    ___momentum___      ______adaptive rate______
```

## Terminology

| Term | Meaning |
|------|---------|
| **Adam** | Adaptive Moment Estimation — a popular optimizer |
| **Momentum** | Running average of past gradients (smooths out noise) |
| **Adaptive learning rate** | Different effective step sizes for different parameters |
| **Exponential moving average** | new = β × old + (1-β) × current |
| **Bias correction** | Compensating for zero-initialization in early steps |
| **Cosine decay** | Learning rate schedule that smoothly decreases to zero |
| **Epsilon (ε)** | Tiny constant to prevent division by zero |

## Next

Now let's see [the complete training loop](./05-the-training-loop.md) — all the pieces we've covered assembled into one coherent process.
