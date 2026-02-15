# MicroGPT: A First-Principles Course

[![GitHub Pages](https://img.shields.io/badge/📖_Read_the_Course-iamkhalid2.github.io/microgpt--course-7c4dff)](https://iamkhalid2.github.io/microgpt-course/)

## What Is This?

A comprehensive, self-contained course that reverse-engineers **every single line** of Andrej Karpathy's [`microgpt.py`](./microgpt.py) — a complete GPT language model in just 200 lines of pure Python, with no external dependencies.

By the end of this course, you will understand exactly how a language model works: from raw text to trained neural network to generated output. No magic, no hand-waving.

**👉 [Read the course →](https://iamkhalid2.github.io/microgpt-course/)**

## Based On

[microgpt.py](https://github.com/karpathy/microgpt) by **Andrej Karpathy** — a minimal but fully functional character-level GPT model implementing:

- A custom autograd engine (automatic differentiation)
- A Transformer architecture (attention, MLP, residual connections)
- A training loop with the Adam optimizer
- Text generation with temperature-controlled sampling

All in **200 lines** using only the standard library.

## Course Structure

| Module | Topic | Lines in microgpt.py |
|--------|-------|:--------------------:|
| **00** | The Big Picture | Overview |
| **01** | Data & Tokenization | 14–27 |
| **02** | Calculus & Autograd | 29–72 |
| **03** | The Architecture | 74–144 |
| **04** | Training | 146–184 |
| **05** | Inference & Generation | 186–200 |
| **Appendix** | Glossary & Math Refresher | — |

**33 lessons** across 6 modules, featuring Mermaid diagrams, KaTeX math, interactive tabs, and styled admonitions — built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Prerequisites

- **Math:** 10th-grade level (algebra, exponents). A [math refresher](https://iamkhalid2.github.io/microgpt-course/appendix/math-refresher/) is included.
- **Programming:** Basic Python (variables, loops, functions, lists).
- **Machine Learning:** Zero prior knowledge required.

## Local Development

```bash
pip install mkdocs-material pymdown-extensions mkdocs-minify-plugin
mkdocs serve
```

Then open `http://127.0.0.1:8000/`.

## Who Is This For?

- **Curious developers** who want to understand what's inside an LLM
- **Students** looking for a ground-up explanation of Transformers
- **Engineers** who use ML frameworks but want to peek under the hood
- **Anyone** who has asked "but how does it *actually* work?"

---

*Course content authored with the assistance of **Claude** (Anthropic).*
