# The Value Class

## The Problem

We know the chain rule lets us compute derivatives through a chain of operations. But doing this by hand for thousands of parameters and hundreds of operations is impossible.

We need the computer to do it **automatically**. That's called **automatic differentiation** (autograd for short).

The trick: every time we do a math operation, we **record** what we did and how to differentiate it. Then at the end, we replay the recording backwards to compute all derivatives at once.

## The Value Class (Lines 30–37)

Here is the core data structure:

```python
# Lines 30-37 of microgpt.py
class Value:
    """Stores a single scalar value and its gradient, as a node in a computation graph."""

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node
        self.grad = 0                   # derivative of the loss w.r.t. this node
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children
```

Every number in the model is wrapped in a `Value` object. Each `Value` stores four things:

| Attribute | What it stores | Example |
|-----------|---------------|---------|
| `data` | The actual number | 3.14 |
| `grad` | How much the loss changes if this number changes | Filled in during backward pass |
| `_children` | Which `Value`s were combined to produce this one | (a, b) if this value = a + b |
| `_local_grads` | The derivative of this operation w.r.t. each child | (1, 1) for addition |

## Addition (Line 39–41)

```python
# Lines 39-41 of microgpt.py
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))
```

When you write `c = a + b` where `a` and `b` are `Value` objects:

1. `self.data + other.data` → compute the result (forward pass)
2. `(self, other)` → remember the children (a and b)
3. `(1, 1)` → store the local gradients

Why `(1, 1)` for addition? Because:
```
c = a + b
dc/da = 1   (changing a by 1 changes c by 1)
dc/db = 1   (changing b by 1 changes c by 1)
```

### Example

```python
a = Value(2.0)
b = Value(3.0)
c = a + b  # c.data = 5.0, c._children = (a, b), c._local_grads = (1, 1)
```

```
          c (5.0)
         / \
  grad=1/   \grad=1    ← local gradients
       /     \
    a(2.0)  b(3.0)
```

## Multiplication (Lines 43–45)

```python
# Lines 43-45 of microgpt.py
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

For `c = a * b`:
```
dc/da = b    (the derivative of a*b with respect to a is b)
dc/db = a    (the derivative of a*b with respect to b is a)
```

So the local gradients are `(other.data, self.data)` — each child's gradient is the *other* child's value.

### Example

```python
a = Value(2.0)
b = Value(3.0)
c = a * b  # c.data = 6.0, c._local_grads = (3.0, 2.0)
```

```
          c (6.0)
         / \
  grad=3/   \grad=2    ← local gradients (swapped!)
       /     \
    a(2.0)  b(3.0)
```

### Why are they swapped?

Intuitively: if you're multiplying `2 × 3` and you increase the 2 to 3, you get `3 × 3 = 9`. The result changed by 3 (which is the *other* number). The sensitivity of the product to one input equals the other input.

## Power (Line 47)

```python
# Line 47 of microgpt.py
def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
```

For `c = a^n`:
```
dc/da = n × a^(n-1)
```

This is the **power rule** from calculus. Note that `other` here is a plain number, not a `Value` — we only compute gradients w.r.t. the base, not the exponent.

### Example: `a² at a=3`

```
dc/da = 2 × 3^(2-1) = 2 × 3 = 6
```

## Logarithm (Line 48)

```python
# Line 48 of microgpt.py
def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
```

For `c = ln(a)`:
```
dc/da = 1/a
```

This is used in the loss function: `-log(probability)`. If the probability is 0.5, the gradient is `1/0.5 = 2`.

## Exponential (Line 49)

```python
# Line 49 of microgpt.py
def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
```

For `c = e^a`:
```
dc/da = e^a
```

The exponential function is its own derivative — one of the most beautiful facts in math. This is used in the softmax function.

## ReLU (Line 50)

```python
# Line 50 of microgpt.py
def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

ReLU (Rectified Linear Unit) is the simplest "activation function":
```
relu(x) = max(0, x) = x if x > 0, else 0
```

Its derivative:
```
d(relu)/dx = 1 if x > 0, else 0
```

Visually:
```
    output                 derivative
      │  /                    │
      │ /                  1  │──────
      │/                      │
──────┼────── x           ────┼────── x
      │                    0  │
```

If the input is positive, the gradient flows through unchanged. If negative, the gradient is zero — the operation is "dead."

## Convenience Operations (Lines 51–57)

```python
# Lines 51-57 of microgpt.py
def __neg__(self): return self * -1
def __radd__(self, other): return self + other
def __sub__(self, other): return self + (-other)
def __rsub__(self, other): return other + (-self)
def __rmul__(self, other): return self * other
def __truediv__(self, other): return self * other**-1
def __rtruediv__(self, other): return other * self**-1
```

These define subtraction, division, and negation in terms of the primitives we already have:

| Operation | Implemented as |
|-----------|---------------|
| `-a` | `a * -1` |
| `a - b` | `a + (-b)` |
| `a / b` | `a * b^(-1)` |

No new gradient logic needed! They just reuse `__add__`, `__mul__`, and `__pow__`.

The `__radd__` and `__rmul__` variants handle cases like `3 + value` (when the `Value` is on the right side of the operator).

## The Computation Graph So Far

Every operation creates a new `Value` node, linked to its children:

```python
a = Value(2.0)
b = Value(3.0)
c = a + b        # 5.0
d = c * a        # 10.0
e = d.log()      # 2.302...
```

This builds a graph:

```
a(2.0)──────┐
  │         │
  │    c = a+b (5.0)
  │         │
  │    d = c×a (10.0)
  │         │
  │    e = ln(d) (2.302)
  │
b(3.0)──────┘
```

The graph records the entire computation. Now we need to walk it backwards to compute gradients. That's the **backward pass** — covered after we understand the [forward pass](./03-forward-pass.md).

## Terminology

| Term | Meaning |
|------|---------|
| **Value** | A wrapper around a number that tracks how it was computed |
| **Computation graph** | The tree of `Value` nodes showing all operations |
| **Local gradient** | The derivative of one operation w.r.t. its inputs |
| **Autograd** | Automatic differentiation — computing all gradients automatically |
| **ReLU** | max(0, x) — an activation function that zeros out negatives |

## Next

Before we see the backward pass, let's understand the [forward pass](./03-forward-pass.md) — what it means to compute "forward" through the graph.
