# The Backward Pass

## The Problem

After the forward pass, we have:

- A final output: the **loss** (a single number measuring how wrong the model was)
- A computation graph: every operation that led to the loss, recorded as `Value` nodes

Now we need to answer: **for every parameter in the model, how much did it contribute to the loss?** That is, we need $\frac{d(\text{loss})}{d(\text{parameter})}$ for every parameter.

## The Backward Pass Algorithm (Lines 59–72)

```python title="microgpt.py — Lines 59-72"
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

```python title="microgpt.py — Lines 60-68"
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

!!! info "What is a topological sort?"

    A **topological sort** is an ordering of nodes such that every node comes **after** all its children.

    The algorithm uses **depth-first search**: visit all children before adding yourself. The `visited` set prevents visiting the same node twice (since a node can be used in multiple places).

### Why do we need this?

Because the backward pass processes nodes from the output back to the inputs. By reversing the topological order (`reversed(topo)`), we guarantee that every node's gradient is fully computed before we try to propagate it to its children.

## Step 2: Seed the Gradient (Line 69)

```python title="microgpt.py — Line 69"
self.grad = 1
```

`self` is the loss node — the root of the graph. We set its gradient to 1 because:

$$\frac{d(\text{loss})}{d(\text{loss})} = 1$$

The derivative of anything with respect to itself is always 1. This is our starting point.

## Step 3: Propagate Gradients Backward (Lines 70–72)

```python title="microgpt.py — Lines 70-72"
for v in reversed(topo):
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad += local_grad * v.grad
```

For each node $v$ (starting from the loss, going backwards):

$$\text{child.grad} \mathrel{+}= \text{local\_grad} \times v\text{.grad}$$

This is the **chain rule in action**: the gradient of a child = (local gradient) × (parent's gradient). We use `+=` because a node might have multiple parents.

## Full Walkthrough

Let's trace the backward pass for $f = \text{relu}(a \times b + d)$:

=== "Step 1: Seed"

    $$f\text{.grad} = 1 \quad \text{(starting point)}$$

=== "Step 2: Process relu"

    $f = \text{relu}(e)$, local grad at $e = 4.0$ is $1.0$ (positive input)

    $$e\text{.grad} \mathrel{+}= 1.0 \times f\text{.grad} = 1.0 \times 1 = 1.0$$

=== "Step 3: Process addition"

    $e = c + d$, local grads are both $1$

    $$c\text{.grad} \mathrel{+}= 1 \times e\text{.grad} = 1 \times 1.0 = 1.0$$

    $$d\text{.grad} \mathrel{+}= 1 \times e\text{.grad} = 1 \times 1.0 = 1.0$$

=== "Step 4: Process multiplication"

    $c = a \times b$, local grad w.r.t. $a$ is $b = -3.0$, w.r.t. $b$ is $a = 2.0$

    $$a\text{.grad} \mathrel{+}= (-3.0) \times c\text{.grad} = -3.0 \times 1.0 = -3.0$$

    $$b\text{.grad} \mathrel{+}= 2.0 \times c\text{.grad} = 2.0 \times 1.0 = 2.0$$

**Result:**

| Node | `.grad` | Meaning |
|:----:|:-------:|---------|
| $a$ | $-3.0$ | "If $a$ increases by 1, $f$ decreases by 3" |
| $b$ | $2.0$ | "If $b$ increases by 1, $f$ increases by 2" |
| $c$ | $1.0$ | |
| $d$ | $1.0$ | |
| $e$ | $1.0$ | |
| $f$ | $1.0$ | seed |

!!! success "Verification"

    $$f(a=2.0) = \text{relu}(2.0 \times (-3.0) + 10.0) = \text{relu}(4.0) = 4.0$$

    $$f(a=2.001) = \text{relu}(2.001 \times (-3.0) + 10.0) = \text{relu}(3.997) = 3.997$$

    $$\frac{\Delta f}{\Delta a} = \frac{3.997 - 4.0}{0.001} = -3.0 \checkmark$$

## The "+=" is Crucial

!!! warning

    Notice `child.grad +=` (not `=`). This is because a value might be used in **multiple places**:

    ```python
    a = Value(3.0)
    b = a + a       # a is used twice!
    ```

    Here $a$ is a child of $b$ through two paths. The gradient contributions from both paths must be **summed**:

    $$\frac{db}{da} = 1 + 1 = 2$$

    And indeed, if $a = 3$ then $b = 6$, and if $a = 4$ then $b = 8$. Rate of change = 2. :white_check_mark:

## What This Means for Training

After calling `loss.backward()`:

- Every parameter (`Value` in the model) has its `.grad` set
- This gradient tells us: "nudge this parameter in the **opposite** direction of `.grad` to reduce the loss"
- The optimizer then uses these gradients to update all parameters

!!! important

    **All of this happens automatically.** The programmer only needs to:

    1. Build the forward pass (which records the graph)
    2. Call `.backward()` on the loss

??? note "Terminology"

    | Term | Meaning |
    |------|---------|
    | **Backward pass** | Walking the graph in reverse to compute gradients |
    | **Topological sort** | Ordering nodes so children come before parents |
    | **Gradient accumulation** | Summing gradient contributions from multiple paths (`+=`) |
    | **Seed gradient** | Setting the loss's gradient to 1 to start the backward pass |
    | **Backpropagation** | Another name for the backward pass |
