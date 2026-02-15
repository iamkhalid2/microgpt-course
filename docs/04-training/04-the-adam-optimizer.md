# The Adam Optimizer

## The Problem with Plain Gradient Descent

Gradient descent uses the same learning rate for every parameter and has no memory of past gradients. Adam fixes both.

## What Adam Adds

**Adam** (Adaptive Moment Estimation) maintains two extra values for **each parameter**:

1. **m (first moment):** A running average of past gradients → **momentum**
2. **v (second moment):** A running average of past squared gradients → **adaptive learning rate**

## Setup (Lines 147–149)

```python title="microgpt.py — Lines 147-149"
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
m = [0.0] * len(params)  # first moment buffer (momentum)
v = [0.0] * len(params)  # second moment buffer (adaptive rates)
```

| Value | What it is | Typical range |
|-------|-----------|:-------------:|
| `learning_rate = 0.01` | Base step size | 1e-4 to 1e-2 |
| `beta1 = 0.9` | How much past momentum to keep | 0.9 – 0.99 |
| `beta2 = 0.95` | How much past variance to keep | 0.95 – 0.999 |
| `eps_adam = 1e-8` | Tiny number to prevent division by zero | 1e-8 |

## The Update (Lines 175–182)

```python title="microgpt.py — Lines 175-182"
lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
for i, p in enumerate(params):
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad           # update momentum
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2      # update variance
    m_hat = m[i] / (1 - beta1 ** (step + 1))              # bias correction
    v_hat = v[i] / (1 - beta2 ** (step + 1))              # bias correction
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)     # update parameter
    p.grad = 0                                              # reset gradient
```

=== "Line 175: Cosine learning rate decay"

    $$\text{lr}_t = \text{lr} \times \frac{1 + \cos(\pi \cdot t / T)}{2}$$

    - Starts at `learning_rate` (0.01)
    - Smoothly decreases to 0 by the end of training
    - Large steps early for fast learning, tiny steps late for fine-tuning

    The formula: $\cos(\pi \cdot t/T)$ goes from $\cos(0)=1$ to $\cos(\pi)=-1$. So $\frac{1+\cos(...)}{2}$ goes from 1 → 0.

=== "Line 177: Momentum (first moment)"

    $$m_i = \beta_1 \cdot m_i + (1 - \beta_1) \cdot g_i = 0.9 \cdot m_i + 0.1 \cdot g_i$$

    An **exponential moving average** of the gradient. Keeps 90% of previous momentum, adds 10% of current gradient.

    Individual gradients are noisy (computed from a single name). Averaging smooths the noise.

    It's like a rolling ball — instead of changing direction every instant, it has **inertia**.

=== "Line 178: Variance (second moment)"

    $$v_i = \beta_2 \cdot v_i + (1 - \beta_2) \cdot g_i^2 = 0.95 \cdot v_i + 0.05 \cdot g_i^2$$

    Tracks how "noisy" each parameter's gradient is:

    - Consistently large gradients → large $v$ → smaller effective step
    - Small gradients → small $v$ → larger effective step

    This is the **adaptive** part — each parameter gets its own effective learning rate.

=== "Lines 179-180: Bias correction"

    $$\hat{m} = \frac{m_i}{1 - \beta_1^{t+1}}, \quad \hat{v} = \frac{v_i}{1 - \beta_2^{t+1}}$$

    Since $m$ and $v$ start at 0, they're biased toward zero early on. Correction inflates them:

    | Step | Correction factor for $m$ | Effect |
    |:----:|:-------------------------:|--------|
    | 0 | $1/(1-0.9^1) = 10\times$ | Large correction |
    | 10 | $\approx 2.9\times$ | Moderate |
    | 100 | $\approx 1.0\times$ | Almost none |

=== "Line 181: The actual update"

    $$\theta_i \leftarrow \theta_i - \text{lr}_t \cdot \frac{\hat{m}_i}{\sqrt{\hat{v}_i} + \epsilon}$$

    - $\hat{m}$ = smoothed gradient (direction + momentum)
    - $\sqrt{\hat{v}}$ = gradient volatility (adaptive scaling)
    - High variance → **smaller** steps (cautious)
    - Low variance → **larger** steps (confident)

## The Formula in One Line

$$\theta \leftarrow \theta - \text{lr} \times \frac{\underbrace{\text{smoothed gradient}}_{\text{momentum}}}{\underbrace{\sqrt{\text{smoothed squared gradient}}}_{\text{adaptive rate}}}$$

??? note "Terminology"

    | Term | Meaning |
    |------|---------|
    | **Adam** | Adaptive Moment Estimation — a popular optimizer |
    | **Momentum** | Running average of past gradients (smooths noise) |
    | **Adaptive learning rate** | Different effective step sizes per parameter |
    | **Exponential moving average** | $\text{new} = \beta \cdot \text{old} + (1-\beta) \cdot \text{current}$ |
    | **Bias correction** | Compensating for zero-initialization in early steps |
    | **Cosine decay** | Learning rate schedule that smoothly decreases to zero |
    | **Epsilon ($\epsilon$)** | Tiny constant to prevent division by zero |
