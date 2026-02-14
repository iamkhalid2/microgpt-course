# The 200-Line Map

## The Entire File at a Glance

Before we dive into the details, let's look at the whole file from 30,000 feet. Every line of `microgpt.py` falls into one of **six blocks**. Here's the map:

```
┌──────────────────────────────────────────────────────────────────┐
│  microgpt.py — 200 lines                                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Lines   1-12   │ SETUP           │ Imports & random seed        │
│  Lines  14-27   │ DATA            │ Load dataset, build tokenizer│
│  Lines  29-72   │ AUTOGRAD ENGINE │ The Value class              │
│  Lines  74-90   │ PARAMETERS      │ Initialize model weights     │
│  Lines  92-144  │ ARCHITECTURE    │ The GPT model function       │
│  Lines 146-184  │ TRAINING        │ Optimizer + training loop    │
│  Lines 186-200  │ INFERENCE       │ Generate new text            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

Let's walk through each block.

---

## Block 1: Setup (Lines 1–12)

```python
import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos
```

Three standard Python libraries, zero external dependencies. The `random.seed(42)` ensures that every time you run the file, you get the same "random" numbers — making experiments reproducible.

---

## Block 2: Data & Tokenization (Lines 14–27)

```python
# Download a dataset of names
docs = [...]           # list of names like ["emma", "olivia", ...]
uchars = sorted(set(...))  # unique characters: ['a', 'b', ..., 'z']
BOS = len(uchars)      # a special "start/end of name" token
vocab_size = len(uchars) + 1
```

**Problem solved:** How do we turn text into numbers a computer can work with?
**Solution:** Assign each unique character an ID (a=0, b=1, ...) plus one special token.

📂 *Covered in detail in [Module 1](../01-data-and-tokenization/00-the-dataset.md)*

---

## Block 3: The Autograd Engine (Lines 29–72)

```python
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        ...
    def backward(self):
        # Automatically compute gradients via the chain rule
        ...
```

**Problem solved:** How do we figure out which parameters are responsible for errors?
**Solution:** Wrap every number in a `Value` object that remembers how it was computed. Then walk backwards through the computation to assign blame.

This is the heart of the file. The `Value` class is a tiny **automatic differentiation engine** — the same idea behind PyTorch's `autograd`.

📂 *Covered in detail in [Module 2](../02-calculus-and-autograd/00-why-we-need-derivatives.md)*

---

## Block 4: Parameters (Lines 74–90)

```python
n_embd = 16       # embedding dimension
n_head = 4        # number of attention heads
n_layer = 1       # number of layers
block_size = 8    # maximum sequence length

state_dict = {
    'wte': matrix(...),    # token embeddings
    'wpe': matrix(...),    # position embeddings
    'lm_head': matrix(...), # output layer
    # + attention and MLP weights for each layer
}
```

**Problem solved:** Where does the model store what it has learned?
**Solution:** In matrices (grids of numbers) that start random and get tuned during training.

📂 *Covered in detail in [Module 3, Lesson 0](../03-the-architecture/00-parameters-are-knowledge.md)*

---

## Block 5: The Architecture (Lines 92–144)

```python
def gpt(token_id, pos_id, keys, values):
    # 1. Look up embeddings
    # 2. For each layer:
    #    a. Multi-head attention (look at context)
    #    b. MLP (process information)
    # 3. Output logits (raw scores for each possible next character)
    return logits
```

**Problem solved:** Given the current character and position, how do we compute a prediction?
**Solution:** A pipeline of transformations: embed → normalize → attend → think → predict.

This is the **Transformer architecture** — the "T" in "GPT".

📂 *Covered in detail in [Module 3](../03-the-architecture/01-embeddings.md)*

---

## Block 6: Training (Lines 146–184)

```python
for step in range(500):
    # 1. Pick a name from the dataset
    # 2. Forward: predict each next character
    # 3. Measure error (loss)
    # 4. Backward: compute gradients
    # 5. Update parameters with Adam optimizer
```

**Problem solved:** How do we make the model better?
**Solution:** Show it examples, measure its mistakes, and nudge its parameters in the right direction. Repeat 500 times.

📂 *Covered in detail in [Module 4](../04-training/00-what-is-training.md)*

---

## Block 7: Inference (Lines 186–200)

```python
for sample_idx in range(20):
    # Start with BOS token
    # Repeatedly: predict next character, pick one, add to output
    # Stop when BOS is predicted again (end of name)
    print(f"sample {sample_idx+1}: {''.join(sample)}")
```

**Problem solved:** How do we use the trained model to create *new* names?
**Solution:** Feed it the start signal, let it predict one character at a time, and collect the output.

📂 *Covered in detail in [Module 5](../05-inference/00-generating-text.md)*

---

## The Dependency Chain

The blocks build on each other in a strict order:

```
Data & Tokenizer
       │
       ▼
   Autograd Engine
       │
       ▼
   Parameters
       │
       ▼
   Architecture
       │
       ▼
    Training  ──────▶  Inference
```

You can't understand the architecture without understanding autograd.
You can't understand training without understanding the architecture.
And you can't generate text without a trained model.

This course follows this exact dependency chain.

## Next

In the [next lesson](./02-the-learning-machine-analogy.md), we'll build a mental model for how the entire learning process works — before any math or code.
