# The Dataset

## The Problem

We want to build a model that generates human-like names. But before we can do any math, we need **examples** for the model to learn from.

Think of it like this: if you wanted to learn what names look like in a foreign language, you'd need a list of names in that language to study. The model needs the same thing.

## The Data (Lines 14–21)

Here's how `microgpt.py` gets its training data:

```python
# Lines 14-21 of microgpt.py
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

Let's break this down line by line.

### Line 15-18: Download if missing

```python
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
```

Checks if the file `input.txt` already exists. If not, it downloads a list of ~32,000 human names from the internet and saves it locally. Simple.

### Line 19: Load and clean

```python
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
```

This is a dense one-liner. Let's unpack it step by step:

```
open('input.txt').read()     → reads the ENTIRE file as one big string
.strip()                     → removes whitespace from the start and end
.split('\n')                 → splits by newlines → list of lines
for l in ...                 → loop over each line
l.strip()                    → remove whitespace from each line
if l.strip()                 → skip empty lines
```

The result: `docs` is a Python list of strings:
```python
docs = ["emma", "olivia", "ava", "isabella", "sophia", ...]
```

Each string is one "document." In our case, a document is a single name.

### Line 20: Shuffle

```python
random.shuffle(docs)
```

Randomize the order. Why? If the model saw all names starting with "a" first, then all "b" names, etc., it might learn the wrong patterns. Shuffling ensures the model sees a diverse mix at each training step.

### Line 21: Print the count

```python
print(f"num docs: {len(docs)}")  # Output: num docs: 32033
```

About 32,000 names. That's our entire dataset.

## Why This Matters

The dataset is the **source of truth**. The model will never know anything that isn't somehow present in this data. If we fed it city names instead, it would generate city-like names. If we fed it Python code, it would generate code-like text.

**The model learns patterns from data. No data = no learning.**

## Terminology

| Term | Meaning |
|------|---------|
| **Document** | A single training example. Here, one name (e.g., "emma") |
| **Dataset** | The full collection of documents |
| **Corpus** | Another word for dataset (you'll see this in papers) |
| **Shuffle** | Randomize the order of documents before training |

## Next

We have a list of strings, but the model needs **numbers**. In the [next lesson](./01-characters-as-numbers.md), we'll see how to convert characters into numeric IDs.
