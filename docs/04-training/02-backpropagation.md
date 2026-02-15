# Backpropagation

## The One-Line Magic

```python title="microgpt.py — Line 172"
loss.backward()
```

One line. It computes the gradient of the loss with respect to **all 4,064 parameters**, simultaneously.

## What Just Happened?

If you went through Module 2, you know exactly what this does:

1. **Topological sort** — Order all `Value` nodes so children come before parents
2. **Seed** — Set `loss.grad = 1`
3. **Propagate** — Walk the graph in reverse:

$$\text{child.grad} \mathrel{+}= \text{local\_grad} \times \text{parent.grad}$$

The result: every `Value` in `params` now has its `.grad` field set.

## The Scale of This Computation

!!! info "For a single training step on a 5-character name"

    ```text
    Forward pass:  ~24,000 Value nodes created
                   (5 positions × ~4,800 operations each)

    Backward pass: ~24,000 nodes visited
                   Each node: 1-2 multiplications + additions
                   Total: ~50,000 arithmetic operations
    ```

    All of this to fill in 4,064 `.grad` values.

## Before and After

=== "Before `loss.backward()`"

    ```text
    params[0].grad = 0       (all gradients are zero)
    params[1].grad = 0
    params[2].grad = 0
    ...
    params[4063].grad = 0
    ```

=== "After `loss.backward()`"

    ```text
    params[0].grad = -0.0023   ← decrease this → nudges loss down
    params[1].grad = 0.0041    ← increase this → nudges loss UP!
    params[2].grad = -0.0001   ← barely matters
    ...
    params[4063].grad = 0.0015
    ```

Each gradient means: "if this parameter increased by a tiny amount, this is how much the loss would change."

## Why This Is Remarkable

Without autograd, to compute 4,064 gradients, you'd need:

| Method | Forward passes per step | × 500 steps | Total |
|--------|:-----------------------:|:-----------:|:-----:|
| Finite differences | 4,065 | × 500 | ~2,000,000 |
| Autograd | 1 forward + 1 backward | × 500 | 1,000 |

!!! tip

    Autograd is **~2000× more efficient** than the brute-force approach. This is why automatic differentiation was a breakthrough.

## The Gradient Reset (Line 182)

```python title="microgpt.py — Line 182"
p.grad = 0
```

!!! warning "Critical"

    After using each gradient for the parameter update, it's reset to zero. Why? Because `backward()` uses `+=` to accumulate gradients. Without resetting, gradients from previous steps would **leak** into the current step, corrupting the computation.

??? note "Terminology"

    | Term | Meaning |
    |------|---------|
    | **Backpropagation** | Running `backward()` on the loss to compute all gradients |
    | **Gradient** | $\partial\text{loss}/\partial\text{parameter}$ — how much the loss changes per unit parameter change |
    | **Gradient reset** | Setting all gradients to zero before the next step |
