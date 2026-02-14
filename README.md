# MicroGPT: A First-Principles Course

## What Is This?

A comprehensive, self-contained course that reverse-engineers **every single line** of Andrej Karpathy's [`microgpt.py`](./microgpt.py) — a complete GPT language model implemented in just 200 lines of pure Python, with no external dependencies.

By the end of this course, you will understand exactly how a language model works: from raw text to trained neural network to generated output. No magic, no hand-waving, no "just trust me."

## Based On

This course is built around [microgpt.py](https://github.com/karpathy/microgpt), a minimal but fully functional character-level GPT model by **Andrej Karpathy**. The file implements:

- A custom autograd engine (automatic differentiation)
- A Transformer architecture (attention, MLP, residual connections)
- A training loop with the Adam optimizer
- Text generation with temperature-controlled sampling

All in **200 lines of Python** using only the standard library (`os`, `math`, `random`, `urllib`).

## Design Philosophy

This course was designed with several principles in mind:

### 1. Problem-First, Not Definition-First

Every lesson starts with a **concrete problem** — "we need to turn characters into numbers" or "we need to measure how wrong the model is" — and then builds toward the solution. Concepts are introduced only when they're needed, never before.

### 2. Incremental Building

Each file builds on the previous one. There are no forward references to unexplained concepts. If you read the files in order, every piece of notation, every formula, and every line of code will make sense when you encounter it.

### 3. Minimal Prerequisites

The course assumes:

- **Math:** 10th-grade level (basic algebra, exponents). A [math refresher](./course/appendix/math-refresher.md) covers everything else.
- **Programming:** Basic Python (variables, loops, functions, lists).
- **Machine Learning:** Zero prior knowledge required.

### 4. Every Line Explained

The course doesn't just explain "how attention works" in the abstract. It quotes **exact lines** from `microgpt.py` with line numbers and traces through them with concrete numerical examples. You'll see what every variable holds at every step.

### 5. Code as the Source of Truth

The course structure mirrors the code structure. Each module corresponds to a section of `microgpt.py`, and every lesson maps to specific line ranges. The [index](./course/INDEX.md) maps every lesson to its corresponding lines.

## Structure

The course is organized into 6 modules plus an appendix, containing **33 markdown files** total:

| Module | Topic | Lines |
|--------|-------|-------|
| **00** | The Big Picture | Overview |
| **01** | Data & Tokenization | 14–27 |
| **02** | Calculus & Autograd | 29–72 |
| **03** | The Architecture | 74–144 |
| **04** | Training | 146–184 |
| **05** | Inference & Generation | 186–200 |
| **Appendix** | Glossary & Math Refresher | — |

## How to Read

1. Start at [`course/00-the-big-picture/00-what-is-a-language-model.md`](./course/00-the-big-picture/00-what-is-a-language-model.md)
2. Follow the "Next" link at the bottom of each lesson
3. Refer to the [glossary](./course/appendix/glossary.md) if you encounter an unfamiliar term
4. Refer to the [math refresher](./course/appendix/math-refresher.md) if any formula feels unclear
5. See the [full index](./course/INDEX.md) for a bird's-eye view of all lessons

## Who Is This For?

- **Curious developers** who want to understand what's inside an LLM
- **Students** looking for a ground-up explanation of Transformers
- **Engineers** who use ML frameworks but want to understand what happens under the hood
- **Anyone** who has asked "but how does it *actually* work?"

---

*This course was created using **Claude Opus 4.6** (Anthropic).*
