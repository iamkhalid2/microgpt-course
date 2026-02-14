# The Backward Pass

## The Problem

After the forward pass, we have:
- A final output: the **loss** (a single number measuring how wrong the model was)
- A computation graph: every operation that led to the loss, recorded as `Value` nodes

Now we need to answer: **for every parameter in the model, how much did it contribute to the loss?** That is, we need the derivative of `loss` with respect to every parameter.

## The Backward Pass Algorithm (Lines 59–72)

```python
# Lines 59-72 of microgpt.py
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

This is only 14 lines of code, but it's doing something profound. Let's decompose it.

## Step 1: Topological Sort (Lines 60–68)

```python
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._children:
            build_topo(child)
        topo.append(v)
build_topo(self)
```

### What is a topological sort?

A **topological sort** is an ordering of nodes such that every node comes **after** all its children.

For our example graph:

```
a(2.0) ─────┐
             ├──[×]──▶ c(-6.0) ──┐
b(-3.0) ────┘                     ├──[+]──▶ e(4.0) ──[relu]──▶ f(4.0)
                                  │
d(10.0) ─────────────────────────┘
```

A valid topological order: `[a, b, c, d, e, f]`

The algorithm uses **depth-first search**: visit all children before adding yourself. The `visited` set prevents visiting the same node twice (since a node can be used in multiple places).

### Why do we need this?

Because the backward pass processes nodes from the output back to the inputs. By reversing the topological order (`reversed(topo)` = `[f, e, d, c, b, a]`), we guarantee that every node's gradient is fully computed before we try to propagate it to its children.

## Step 2: Seed the Gradient (Line 69)

```python
self.grad = 1
```

`self` is the loss node — the root of the graph. We set its gradient to 1 because:

```
d(loss)/d(loss) = 1
```

The derivative of anything with respect to itself is always 1. This is our starting point.

## Step 3: Propagate Gradients Backward (Lines 70–72)

```python
for v in reversed(topo):
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad += local_grad * v.grad
```

For each node `v` (starting from the loss, going backwards):
- For each child of `v`:
  - `child.grad += local_grad * v.grad`

This is the **chain rule in action**: the gradient of a child = (local gradient) × (parent's gradient). We use `+=` because a node might have multiple parents (its gradient accumulates contributions from all of them).

## Full Walkthrough

Let's trace the backward pass for `f = relu(a*b + d)`:

```
Forward pass results:
  a.data = 2.0,  b.data = -3.0,  c.data = -6.0,  d.data = 10.0
  e.data = 4.0,  f.data = 4.0
```

**Backward pass:**

```
Step 1: f.grad = 1  (seed)

Step 2: Process f = relu(e)
        Local grad of relu at e=4.0 is 1.0 (positive input)
        e.grad += 1.0 × f.grad = 1.0 × 1 = 1.0

Step 3: Process e = c + d
        Local grad of + w.r.t. c is 1
        Local grad of + w.r.t. d is 1
        c.grad += 1 × e.grad = 1 × 1.0 = 1.0
        d.grad += 1 × e.grad = 1 × 1.0 = 1.0

Step 4: Process c = a * b
        Local grad of × w.r.t. a is b.data = -3.0
        Local grad of × w.r.t. b is a.data = 2.0
        a.grad += (-3.0) × c.grad = -3.0 × 1.0 = -3.0
        b.grad += 2.0 × c.grad = 2.0 × 1.0 = 2.0
```

**Result:**
```
a.grad = -3.0   →  "If a increases by 1, f decreases by 3"
b.grad =  2.0   →  "If b increases by 1, f increases by 2"
c.grad =  1.0
d.grad =  1.0
e.grad =  1.0
f.grad =  1.0
```

Let's verify `a.grad = -3.0`:
```
f(a=2.0) = relu(2.0 × (-3.0) + 10.0) = relu(4.0) = 4.0
f(a=2.001) = relu(2.001 × (-3.0) + 10.0) = relu(3.997) = 3.997
Change: (3.997 - 4.0) / 0.001 = -3.0 ✓
```

## The "+=" is Crucial

Notice `child.grad +=` (not `=`). This is because a value might be used in **multiple places**:

```python
a = Value(3.0)
b = a + a       # a is used twice!
```

Here `a` is a child of `b` through two paths. The gradient contributions from both paths must be **summed**:

```
b = a + a
db/da (first use) = 1
db/da (second use) = 1
Total: a.grad = 1 + 1 = 2
```

And indeed, if a = 3 then b = 6, and if a = 4 then b = 8. The rate of change is 2. ✓

## The Algorithm in One Picture

```
                 FORWARD (compute values)
                 ═══════════════════════▶

  a(2.0) ─────┐
    grad=-3.0  ├──[×]──▶ c(-6.0)──┐
  b(-3.0)─────┘   grad=1.0        ├──[+]──▶ e(4.0)──[relu]──▶ f(4.0)
    grad=2.0                       │  grad=1.0          grad=1.0
                                   │
  d(10.0) ────────────────────────┘
    grad=1.0

                 ◀═══════════════════════
                 BACKWARD (compute gradients)
```

## What This Means for Training

After calling `loss.backward()`:
- Every parameter (`Value` in the model) has its `.grad` set
- This gradient tells us: "nudge this parameter in the opposite direction of `.grad` to reduce the loss"
- The optimizer then uses these gradients to update all parameters

**All of this happens automatically.** The programmer only needs to:
1. Build the forward pass (which records the graph)
2. Call `.backward()` on the loss

## Terminology

| Term | Meaning |
|------|---------|
| **Backward pass** | Walking the graph in reverse to compute gradients |
| **Topological sort** | Ordering nodes so children come before parents |
| **Gradient accumulation** | Summing gradient contributions from multiple paths (`+=`) |
| **Seed gradient** | Setting the loss's gradient to 1 to start the backward pass |
| **Backpropagation** | Another name for the backward pass (propagating gradients back) |

## Next

Let's step back and see the [full computation graph](./05-building-a-computation-graph.md) that gets built during a real model forward pass.
