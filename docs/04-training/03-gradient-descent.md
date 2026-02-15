# Gradient Descent

## The Simplest Optimizer

Now that we have gradients, updating parameters is conceptually simple:

$$\theta \leftarrow \theta - \eta \cdot \nabla\theta$$

Or in plain code:

```python
parameter = parameter - learning_rate * gradient
```

That's **gradient descent** — literally "descending" along the gradient (slope) of the loss surface.

## Why Subtract?

The gradient points in the direction of **steepest increase** of the loss. We want the loss to **decrease**. So we move in the **opposite** direction.

=== "Positive gradient"

    ```text
    gradient > 0
    → increasing the parameter increases the loss
    → so DECREASE the parameter (subtract)

    Example: param=0.5, grad=2.0, lr=0.01
    0.5 - 0.01 × 2.0 = 0.48
    ```

=== "Negative gradient"

    ```text
    gradient < 0
    → increasing the parameter decreases the loss
    → so INCREASE the parameter (subtract a negative = add)

    Example: param=0.5, grad=-2.0, lr=0.01
    0.5 - 0.01 × (-2.0) = 0.52
    ```

## The Learning Rate

The gradient tells us the **direction** to move, but not **how far**. The learning rate ($\eta$) controls the step size:

=== "Too large"

    Steps are so big you **overshoot** the minimum and the loss bounces around.

=== "Too small"

    Steps are so tiny that training takes forever.

=== "Just right"

    Steady progress toward lower loss.

## Why Not Just Use Gradient Descent?

Simple gradient descent (called **SGD** — Stochastic Gradient Descent) works but has problems:

| Problem | Why it matters |
|---------|---------------|
| One learning rate for all parameters | Some parameters might need small steps, others large ones |
| No momentum | Each step only uses the current gradient. If the gradient is noisy, the path zigzags |
| Hard to tune | Learning rate is very sensitive |

!!! info

    This is why `microgpt.py` uses **Adam** — a smarter optimizer that fixes all three problems.

??? note "Terminology"

    | Term | Meaning |
    |------|---------|
    | **Gradient descent** | Update: $\theta \leftarrow \theta - \eta \cdot g$ |
    | **SGD** | Stochastic Gradient Descent — gradient descent on a random subset of data |
    | **Learning rate ($\eta$)** | How big each step is |
    | **Convergence** | When the loss stops decreasing |
    | **Overshoot** | When the learning rate is so large that updates make things worse |
