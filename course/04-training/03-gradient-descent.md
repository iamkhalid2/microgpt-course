# Gradient Descent

## The Simplest Optimizer

Now that we have gradients, updating parameters is conceptually simple:

```
parameter = parameter - learning_rate × gradient
```

That's **gradient descent** — literally "descending" along the gradient (slope) of the loss surface.

## Why Subtract?

The gradient points in the direction of **steepest increase** of the loss. We want the loss to **decrease**. So we move in the **opposite** direction — hence the minus sign.

```
gradient is positive  →  increasing the parameter increases the loss
                      →  so DECREASE the parameter (subtract)

gradient is negative  →  increasing the parameter decreases the loss
                      →  so INCREASE the parameter (subtract a negative = add)
```

## The Learning Rate

The gradient tells us the **direction** to move, but not **how far**. The learning rate controls the step size:

```
learning_rate = 0.01

parameter = 0.5
gradient = -2.0

update: 0.5 - 0.01 × (-2.0) = 0.5 + 0.02 = 0.52
```

- **Too large:** Steps are so big you overshoot the minimum and the loss bounces around
- **Too small:** Steps are so tiny that training takes forever
- **Just right:** Steady progress toward lower loss

```
Too large:                   Too small:                 Just right:
loss                         loss                       loss
  │  *                         │  *                       │  *
  │ *  *                       │   *                      │   *
  │*    *                      │    *                     │    *
  │      *                     │     *                    │     *
  │*      *  (bouncing!)       │      *                   │      *
  └──────────                  │       *  (too slow!)     │       *
                               │        *                 │        * *
                               └──────────                └──────────
```

## Why Not Just Use Gradient Descent?

Simple gradient descent (called **SGD** — Stochastic Gradient Descent) works but has problems:

1. **One learning rate for all parameters:** Some parameters might need small steps, others large ones
2. **No momentum:** Each step only uses the current gradient. If the gradient is noisy (which it is, since we train on one name at a time), the path to the minimum is zigzaggy
3. **Learning rate is hard to tune:** It's very sensitive

This is why `microgpt.py` uses **Adam** — a smarter optimizer that fixes all three problems.

## Terminology

| Term | Meaning |
|------|---------|
| **Gradient descent** | Update: parameter -= lr × gradient |
| **SGD** | Stochastic Gradient Descent — gradient descent on a random subset of data |
| **Learning rate (lr)** | How big each step is |
| **Step** | One parameter update |
| **Convergence** | When the loss stops decreasing (you've found a good minimum) |
| **Overshoot** | When the learning rate is so large that updates make things worse |

## Next

Let's see how the [Adam optimizer](./04-the-adam-optimizer.md) improves on basic gradient descent.
