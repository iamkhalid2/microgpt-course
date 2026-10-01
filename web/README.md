# Invent a GPT: the interactive course

A custom site that teaches [`../microgpt.py`](../microgpt.py) (Andrej Karpathy's 200-line GPT) from first principles, to someone who knows no maths and little Python. Every idea is introduced as the fix for a problem you've just felt, built with interactive widgets and exercises, and banked into a "fog-of-war" copy of the real file that lights up line by line.

Stack: Vite + Svelte 5, CodeMirror 6, and Python in the browser (Pyodide in a Web Worker, loaded from a CDN). Nothing is sent anywhere; progress lives in `localStorage`.

## Run it

    npm install
    npm run dev        # http://localhost:5173
    npm run build      # static site in dist/ (hash routing, relative base: works on GitHub Pages)
    npm test           # unit tests for the JS logic libraries

`npm run dev` and `npm run build` first copy `../input.txt` and `../microgpt.py` into `public/data/` (git-ignored), so the site always serves the exact files in the repo root.

## What's in it

Prologue and chapters 1–19, in five acts:

| Act | Chapters | You invent |
|-----|----------|------------|
| I | 1–3 | tokens, probabilities and a counting model, the loss |
| II | 4–6 | dials, slopes, gradient descent, the chain rule |
| III | 7–9 | an autograd engine, softmax, embeddings and linear layers |
| IV | 10–14 | attention, heads, the MLP, skip lanes, RMSNorm, the whole `gpt()` |
| V | 15–19 | Adam, the training loop, sampling, reading the full file, a blank-page rebuild |

Every exercise has **two routes** that count the same and run the same hidden check: a *Plain English* recipe (choose the right step, with the real Python beside it) or the *Python editor*. Recipes live in `src/chapters/plain.js`.

The final chapter includes a capstone (rebuild microgpt from a blank editor, with milestone tests and a three-level hint ladder) and a "mutation lab" which trains the real `microgpt.py` in the browser on other data (dinosaurs, cities, your own list) or with other sizes.

Also: a glossary (`#/glossary`), a reader mode, light and dark themes, and a mobile layout.

## Honesty rules

Every number quoted in the text was measured by running real code, and where feasible is pinned by a test (`tests/*.test.js`). The JS port of the model (`src/lib/gpt.js`) is verified against Python reference logits. The weights in `public/data/weights.json` come from a real 3,000-step run of `../microgpt.py`; `replay.json` and `long_run.json` are real training recordings. To regenerate them:

    python3 tools/record_replay.py      # public/data/replay.json   (~1–2 min)
    python3 tools/record_long_run.py    # public/data/long_run.json (~10 min)
    python3 tools/train_export.py       # public/data/weights.json  (~6 min)
    python3 tools/facts.py              # Python ground truth used by the unit tests

## Browser tests

They drive your installed Google Chrome with `playwright-core` and need the dev server running (`npm run dev`):

    node tests/e2e/ch0.mjs      # also ch1 … ch19, plain, mobile, glossary
    node tests/e2e/widgets.mjs  # screenshots of the widgets, add --dark for dark mode
    node tests/e2e/all.mjs      # run everything in order

Screenshots go to `$SHOTS` (default `/tmp/igpt-shots`).

## Deploying

`.github/workflows/deploy.yml` runs `npm ci`, `npm test` and `npm run build` in `web/`, and publishes `web/dist` to GitHub Pages. In the repository settings, set Pages → Source to "GitHub Actions". The old MkDocs course is still in `../docs`, untouched, but no longer built.
