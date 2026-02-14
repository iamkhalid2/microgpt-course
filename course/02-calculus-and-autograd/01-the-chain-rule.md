# The Chain Rule

## The Problem

We know derivatives tell us "which way to nudge." But in a neural network, the loss depends on parameters through a **long chain** of operations:

```
parameter → embedding → linear → relu → linear → softmax → log → loss
```

How do we find the derivative of `loss` with respect to `parameter` when there are 6 operations between them?

## Composition of Functions

When you nest functions inside functions, it's called **composition**:

```
Simple:    f(x) = x²    (one step)

Composed:  h(x) = f(g(x))
           where g(x) = 2x + 1  and  f(z) = z²
           so h(x) = (2x + 1)²  (two steps)
```

For `h(x) = (2x + 1)²`:
1. First compute `g = 2x + 1`
2. Then compute `f = g²`

The question is: what is the derivative of `h` with respect to `x`?

## The Chain Rule

The chain rule says:

> **The derivative of a composition is the product of the individual derivatives.**

In notation:

```
dh/dx = df/dg × dg/dx
```

Let's verify with numbers. For `h(x) = (2x + 1)²` at `x = 3`:

**Step 1:** Individual derivatives
```
g(x) = 2x + 1   →  dg/dx = 2
f(g) = g²        →  df/dg = 2g = 2(2x + 1)
```

**Step 2:** Chain rule
```
dh/dx = df/dg × dg/dx = 2(2×3 + 1) × 2 = 2(7) × 2 = 28
```

**Verification:** Nudge x from 3 to 3.001:
```
h(3)     = (2×3 + 1)²     = 7²     = 49
h(3.001) = (2×3.001 + 1)² = 7.002² = 49.028004
Change: 0.028004 / 0.001 = 28.004 ≈ 28 ✓
```

## The Chain Rule Visually

Think of it like a pipeline. Each stage multiplies the "sensitivity":

```
x ──[×2]──▶ g ──[×2g]──▶ h

If x wiggles by 1:
  g wiggles by 2        (because dg/dx = 2)
  h wiggles by 2 × 2g   (because df/dg = 2g, and g already wiggled by 2)
```

Or think of it like gears:

```
[Small gear] ──▶ [Medium gear] ──▶ [Big gear]
    x                 g                  h

If the small gear turns 1 revolution:
  The medium gear turns 2 revolutions (the gear ratio dg/dx = 2)
  The big gear turns 2g × 2 revolutions (the gear ratio df/dg = 2g)
```

## Longer Chains

The chain rule extends to any number of steps:

```
f(g(h(x)))

df/dx = df/dg × dg/dh × dh/dx
```

Just multiply all the individual derivatives along the chain.

In a neural network, this chain might be 10 or 100 steps long. But the principle is always the same: **multiply the local derivatives along the path**.

## The Key Insight for Autograd

This is why `microgpt.py`'s `Value` class stores **local gradients** at each operation:

```
z = x + y   →  local gradient of z w.r.t. x is 1
z = x * y   →  local gradient of z w.r.t. x is y
z = x²      →  local gradient of z w.r.t. x is 2x
```

Each operation only needs to know its *own* local derivative. The chain rule takes care of composing them into the full derivative.

## A Three-Node Example

Let's trace through a tiny computation graph:

```
a = 2.0
b = 3.0
c = a × b    (c = 6.0)
d = c + 1    (d = 7.0)
loss = d²    (loss = 49.0)
```

We want: `d(loss)/da` — how does the loss change if we tweak `a`?

```
d(loss)/da = d(loss)/dd × dd/dc × dc/da
           = 2d          × 1     × b
           = 2(7)        × 1     × 3
           = 42
```

Each factor is a **local gradient** — the derivative of one node with respect to its immediate input. The chain rule multiplies them together.

## Terminology

| Term | Meaning |
|------|---------|
| **Chain rule** | d(f∘g)/dx = df/dg × dg/dx — multiply local derivatives |
| **Local gradient** | The derivative of one operation w.r.t. its immediate input |
| **Composition** | Nesting functions: f(g(x)) |
| **Sensitivity** | How much the output changes when an input wiggles |

## Next

Now we have the math. In the [next lesson](./02-the-value-class.md), we'll see how `microgpt.py` implements this in code with the `Value` class.
