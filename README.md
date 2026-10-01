# Invent a GPT: a first-principles course

An interactive course that teaches [`microgpt.py`](./microgpt.py), Andrej Karpathy's complete GPT in about 200 lines of dependency-free Python, to someone who knows **no maths and only a little Python**. The goal is the one from Burn Math Class: you finish thinking *"I could have come up with that."*

You don't read about a GPT, you **reinvent** one. Each chapter starts with a problem you can feel (letters are just shapes to a computer; a counting table can't improve; the chain rule by hand is hopeless…), lets you guess, play with a widget, build the fix, break it, and then **bank** it: the matching lines of the real `microgpt.py` light up in a fog-of-war copy of the file. By the end, all 160 lines of code are yours, and the last chapter hands you a blank editor to rebuild it from nothing.

Everything runs in your browser (Python included, via Pyodide). Nothing is uploaded.

## The course

| Act | Chapters | You invent |
|-----|----------|------------|
| Prologue | 0 | the destination: watch the real file train |
| I | 1–3 | tokens, probabilities and a counting model, the loss |
| II | 4–6 | dials, slopes, gradient descent, the chain rule |
| III | 7–9 | an autograd engine, softmax, embeddings and linear layers |
| IV | 10–14 | attention, heads, the MLP, skip lanes, RMSNorm, the whole `gpt()` |
| V | 15–19 | Adam, the training loop, sampling, reading the full file, a blank-page rebuild |

- **Two routes for every exercise.** Plain English recipes (pick the right step; the real Python sits beside it) or the Python editor. Both count; both run the same hidden check.
- **Everything is measured.** Every number in the text came from running real code and is pinned by a test where feasible. The in-browser model is checked against Python reference outputs.
- **Real training, live.** Chapter 16 trains the actual `microgpt.py` in your browser; Chapter 19 lets you train it on dinosaurs, cities or your own list.
- Glossary, light/dark themes, reader mode, mobile layout.

## Run it

```bash
cd web
npm install
npm run dev      # http://localhost:5173
npm run build    # static site in web/dist
npm test         # unit tests
```

See [`web/README.md`](./web/README.md) for the architecture, the tests and how the recorded data is regenerated. A GitHub Actions workflow (`.github/workflows/deploy.yml`) builds `web/` and publishes it to GitHub Pages.

## Repo layout

| Path | What it is |
|------|------------|
| `microgpt.py`, `input.txt` | Karpathy's file and its dataset (32,033 names). The site serves these exact files. |
| `web/` | The interactive course (Vite + Svelte 5). |
| `docs/`, `mkdocs.yml` | The original text-only version of the course (MkDocs). Kept for reference; no longer built. |

## Credits

[`microgpt.py`](https://github.com/karpathy/microgpt) is by **Andrej Karpathy**. The course was written with the assistance of **Claude** (Anthropic).
