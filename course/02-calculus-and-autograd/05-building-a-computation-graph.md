# Building a Computation Graph

## Putting It All Together

We've seen the individual pieces:
- `Value` wraps numbers and records operations
- The forward pass builds the graph
- The backward pass walks it in reverse to compute gradients

Now let's see how a **realistic example** looks — one that mirrors what actually happens inside `microgpt.py`.

## A Mini Neural Network

Let's build the simplest possible "network" — one that takes a number and tries to predict another number:

```python
# Two parameters (the model's "knowledge")
w = Value(0.5)    # weight
b = Value(0.1)    # bias

# Input and target
x = 2.0           # input (plain number, not a Value)
target = 3.0      # we want the model to output 3.0

# Forward pass: the prediction
prediction = w * x + b    # 0.5 × 2.0 + 0.1 = 1.1

# Loss: how wrong are we?
error = prediction - target    # 1.1 - 3.0 = -1.9
loss = error ** 2              # (-1.9)² = 3.61
```

### The graph that gets built:

```
                    Forward direction ──▶

w(0.5) ──┐
          ├── [×] ──▶ wx(1.0) ──┐
x(2.0) ──┘                      ├── [+] ──▶ pred(1.1) ──┐
                                 │                        ├── [-] ──▶ err(-1.9) ──┐
b(0.1) ─────────────────────────┘                        │                        ├── [²] ──▶ loss(3.61)
                                                          │                        │
                                              target(3.0)─┘                        │
```

### Now backward:

```python
loss.backward()
```

```
loss.grad = 1.0                     (seed)

err.grad = 2 × (-1.9) × 1.0        (power rule: d(x²)/dx = 2x)
         = -3.8

pred.grad = 1 × err.grad = -3.8    (subtraction passes gradient through)
target_node.grad = -1 × err.grad = 3.8

b.grad = 1 × pred.grad = -3.8      (addition: local grad = 1)
wx.grad = 1 × pred.grad = -3.8     (addition: local grad = 1)

w.grad = x × wx.grad = 2.0 × (-3.8) = -7.6   (multiplication: local grad = other input)
```

### What do the gradients tell us?

```
w.grad = -7.6   →  Increasing w would DECREASE the loss
                    (w is too small, we should increase it)

b.grad = -3.8   →  Increasing b would DECREASE the loss
                    (b is too small, we should increase it)
```

This makes sense! Our prediction was 1.1 but the target was 3.0 — we're too low. Both `w` and `b` need to increase.

### Update:

```python
learning_rate = 0.01
w.data -= learning_rate * w.grad   # 0.5 - 0.01×(-7.6) = 0.576
b.data -= learning_rate * b.grad   # 0.1 - 0.01×(-3.8) = 0.138
```

New prediction: `0.576 × 2.0 + 0.138 = 1.29` (closer to 3.0!)

Repeat this hundreds of times, and `w` and `b` will converge to values that make the prediction close to 3.0.

## Scale: What the Real Graph Looks Like

In the mini example above, the graph had ~8 nodes. In `microgpt.py`, a single forward pass through the `gpt()` function creates **thousands** of nodes:

| Component | Approximate # of Value operations |
|-----------|----------------------------------|
| Embedding lookup | 16 additions (n_embd = 16) |
| RMSNorm | ~50 multiply/add/power ops |
| One attention head | ~200 multiply/add ops |
| Four attention heads | ~800 ops |
| MLP (expand + activate + compress) | ~1500 ops |
| Output linear | ~400 ops |
| Softmax | ~80 ops |
| Loss (log) | ~2 ops |
| **Total per token** | **~3,000+ nodes** |
| **Per 8-token sequence** | **~24,000+ nodes** |

All of these nodes are `Value` objects sitting in memory, linked together. When you call `loss.backward()`, the algorithm visits each node exactly once (thanks to topological sort) and computes the gradient. One pass, all gradients, every parameter.

## Why This Is Beautiful

In a model with 10,000 parameters:
- **Without autograd:** You'd need ~20,000 forward passes (nudge each parameter up and down) to estimate gradients
- **With autograd:** You need **exactly 2 passes** — one forward, one backward

This is why Karpathy says "everything else is just efficiency." The autograd engine is the core invention that makes training neural networks practical.

## Checkpoint ✓

At this point, you understand:
- ✅ Derivatives: "which direction to nudge" (Lesson 0)
- ✅ Chain rule: composing derivatives through a chain (Lesson 1)
- ✅ Value class: recording operations and storing local gradients (Lesson 2)
- ✅ Forward pass: computing the output and building the graph (Lesson 3)
- ✅ Backward pass: walking the graph in reverse to compute all gradients (Lesson 4)
- ✅ The full picture: how these pieces work together (this lesson)

What we **don't** know yet: what does the `gpt()` function actually compute? What is attention? What is a linear layer? For that, we need to understand the **architecture**.

## Next

On to [Module 3: The Architecture](../03-the-architecture/00-parameters-are-knowledge.md) — where we build the neural network that does the actual predicting.
