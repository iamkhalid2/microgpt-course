# The Dataset

## The Problem

We want to build a model that generates human-like names. But before we can do any math, we need **examples** for the model to learn from.

!!! info "Think of it this way"

    If you wanted to learn what names look like in a foreign language, you'd need a list of names in that language to study. The model needs the same thing.

## The Data (Lines 14–21)

Here's how `microgpt.py` gets its training data:

```python title="microgpt.py — Lines 14-21"
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

Let's break this down line by line.

### Lines 15-18: Download if missing

```python title="microgpt.py — Lines 15-18"
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
```

Checks if the file `input.txt` already exists. If not, it downloads a list of ~32,000 human names from the internet and saves it locally. Simple.

### Line 19: Load and clean

```python title="microgpt.py — Line 19"
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
```

This is a dense one-liner. Let's unpack it step by step:

=== "Step 1: Read"

    ```python
    open('input.txt').read()  # → reads the ENTIRE file as one big string
    ```

=== "Step 2: Strip"

    ```python
    .strip()  # → removes whitespace from the start and end
    ```

=== "Step 3: Split"

    ```python
    .split('\n')  # → splits by newlines → list of lines
    ```

=== "Step 4: Filter"

    ```python
    [l.strip() for l in ... if l.strip()]  # → strip each line, skip empties
    ```

The result: `docs` is a Python list of strings:

```python
docs = ["emma", "olivia", "ava", "isabella", "sophia", ...]
```

Each string is one "document." In our case, a document is a single name.

### Line 20: Shuffle

```python title="microgpt.py — Line 20"
random.shuffle(docs)
```

!!! warning "Why shuffle?"

    If the model saw all names starting with "a" first, then all "b" names, etc., it might learn the wrong patterns. Shuffling ensures the model sees a diverse mix at each training step.

### Line 21: Print the count

```python title="microgpt.py — Line 21"
print(f"num docs: {len(docs)}")  # Output: num docs: 32033
```

About 32,000 names. That's our entire dataset.

## Why This Matters

!!! important

    The dataset is the **source of truth**. The model will never know anything that isn't somehow present in this data. If we fed it city names instead, it would generate city-like names. If we fed it Python code, it would generate code-like text.

    **The model learns patterns from data. No data = no learning.**

??? note "Terminology"

    | Term | Meaning |
    |------|---------|
    | **Document** | A single training example. Here, one name (e.g., "emma") |
    | **Dataset** | The full collection of documents |
    | **Corpus** | Another word for dataset (you'll see this in papers) |
    | **Shuffle** | Randomize the order of documents before training |
