# Backpropagation

## The One-Line Magic

```python
# Line 172 of microgpt.py
loss.backward()
```

That's it. One line. And it computes the gradient of the loss with respect to **all 4,064 parameters**, simultaneously.

## What Just Happened?

If you went through Module 2, you know exactly what this does:

1. **Topological sort** — Order all `Value` nodes so children come before parents
2. **Seed** — Set `loss.grad = 1`
3. **Propagate** — Walk the graph in reverse, applying the chain rule at each node:
   ```
   child.grad += local_grad × parent.grad
   ```

The result: every `Value` in `params` now has its `.grad` field set.

## The Scale of This Computation

For a single training step on a 5-character name:

```
Forward pass: ~24,000 Value nodes created
              (5 positions × ~4,800 operations each)

Backward pass: ~24,000 nodes visited
               Each node: 1-2 multiplications + additions
               Total: ~50,000 arithmetic operations
```

All of this to fill in 4,064 `.grad` values. Each gradient tells us: "if this parameter increased by a tiny amount, this is how much the loss would change."

## Before and After

```
Before loss.backward():
  params[0].grad = 0     (all gradients are zero)
  params[1].grad = 0
  params[2].grad = 0
  ...
  params[4063].grad = 0

After loss.backward():
  params[0].grad = -0.0023   (decrease this ← nudges loss down)
  params[1].grad = 0.0041    (increase this ← nudges loss up!)
  params[2].grad = -0.0001   (barely matters)
  ...
  params[4063].grad = 0.0015
```

## Why This Is Remarkable

Without autograd, to compute these 4,064 gradients, you'd need to:
1. Run the forward pass once for the original loss
2. Run 4,064 more forward passes — one per parameter — nudging each by a tiny ε
3. Compare: `grad ≈ (loss_nudged - loss_original) / ε`

That's **4,065 forward passes** per training step × 500 steps = **~2 million forward passes**.

With autograd: **1 forward pass + 1 backward pass** per step × 500 steps = **1,000 total passes**.

This is why automatic differentiation was a breakthrough.

## The Gradient Reset (Line 182)

There's one subtle but critical line at the end of each training step:

```python
# Line 182 of microgpt.py
p.grad = 0
```

After using each gradient for the parameter update, it's reset to zero. Why? Because `backward()` uses `+=` to accumulate gradients. Without resetting, gradients from previous steps would leak into the current step, corrupting the computation.

## Terminology

| Term | Meaning |
|------|---------|
| **Backpropagation** | Running `backward()` on the loss to compute all gradients |
| **Gradient** | `∂loss/∂parameter` — how much the loss changes per unit parameter change |
| **Gradient reset** | Setting all gradients to zero before the next step |

## Next

Now we have gradients. The simplest thing to do is [gradient descent](./03-gradient-descent.md) — just subtract the gradient from each parameter.
