// Pure counting / probability code for chapters 1-3. No DOM, no network: importable from Node tests.
// Conventions follow microgpt.py exactly: characters are sorted, BOS = len(uchars),
// and every name is wrapped as [BOS, c1 ... cn, BOS].
import { sampleIndex } from './rng.js';

export function parseDocs(text) {
  return text.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
}

export function buildVocab(docs) {
  const uchars = [...new Set(docs.join(''))].sort(); // code-point order, like Python's sorted()
  const BOS = uchars.length;
  return { uchars, BOS, V: uchars.length + 1 };
}

export function encode(doc, vocab) {
  const ids = [vocab.BOS];
  for (const ch of doc) {
    const i = vocab.uchars.indexOf(ch);
    if (i < 0) return null; // a character the model has never seen
    ids.push(i);
  }
  ids.push(vocab.BOS);
  return ids;
}

export const tokenLabel = (id, vocab) => (id === vocab.BOS ? '⏎' : vocab.uchars[id]);

// Every (context -> next) event, as [prev, next] pairs.
export function* pairs(docs, vocab) {
  for (const d of docs) {
    const t = encode(d, vocab);
    for (let i = 0; i < t.length - 1; i++) yield [t[i], t[i + 1]];
  }
}

export function unigramCounts(docs, vocab) {
  const c = new Array(vocab.V).fill(0);
  for (const [, nxt] of pairs(docs, vocab)) c[nxt]++;
  return c;
}

export function bigramCounts(docs, vocab) {
  const c = Array.from({ length: vocab.V }, () => new Array(vocab.V).fill(0));
  for (const [p, n] of pairs(docs, vocab)) c[p][n]++;
  return c;
}

export const sum = (a) => a.reduce((s, x) => s + x, 0);

// counts -> probabilities, with optional "pretend we saw everything alpha extra times" smoothing.
export function normalize(counts, alpha = 0) {
  const total = sum(counts) + alpha * counts.length;
  return counts.map((c) => (c + alpha) / total);
}

// ---- Losses (average surprise in nats over every predicted token) ---------------------------
export const lossUniform = (V) => Math.log(V);

export function lossWith(docs, vocab, probOf) {
  let total = 0, n = 0;
  for (const [p, nxt] of pairs(docs, vocab)) {
    const pr = probOf(p, nxt);
    total += pr > 0 ? -Math.log(pr) : Infinity;
    n++;
  }
  return total / n;
}

export function lossUnigram(docs, vocab, trainDocs = docs, alpha = 0) {
  const probs = normalize(unigramCounts(trainDocs, vocab), alpha);
  return lossWith(docs, vocab, (_p, n) => probs[n]);
}

export function lossBigram(docs, vocab, trainDocs = docs, alpha = 0) {
  const rows = bigramCounts(trainDocs, vocab).map((r) => normalize(r, alpha));
  return lossWith(docs, vocab, (p, n) => rows[p][n]);
}

// ---- Generation -----------------------------------------------------------------------------
export function sampleUnigram(counts, vocab, rng, maxLen = 16) {
  const out = [];
  for (let i = 0; i < maxLen; i++) {
    const t = sampleIndex(counts, rng);
    if (t === vocab.BOS) break;
    out.push(vocab.uchars[t]);
  }
  return out.join('');
}

export function sampleBigram(bigram, vocab, rng, maxLen = 16) {
  const out = [];
  let cur = vocab.BOS;
  for (let i = 0; i < maxLen; i++) {
    const t = sampleIndex(bigram[cur], rng);
    if (t === vocab.BOS) break;
    out.push(vocab.uchars[t]);
    cur = t;
  }
  return out.join('');
}

export function sampleBabble(vocab, rng, maxLen = 10) {
  // uniformly random letters; after each letter a 1-in-(maxLen/2) chance of stopping
  const out = [];
  const len = 2 + rng.int(maxLen - 2);
  for (let i = 0; i < len; i++) out.push(vocab.uchars[rng.int(vocab.uchars.length)]);
  return out.join('');
}

// ---- The explosion problem: longer context => table that the data can never fill --------------
// For each context length n, how many distinct contexts exist in `trainDocs`, versus how many
// are possible (V^n), and how often a held-out position has a context the table has never seen.
export function contextCoverage(trainDocs, testDocs, vocab, maxN = 6) {
  const rows = [];
  const ctxKey = (t, i, n) => t.slice(Math.max(0, i - n + 1), i + 1).join(',');
  for (let n = 1; n <= maxN; n++) {
    const seen = new Set();
    let events = 0;
    for (const d of trainDocs) {
      const t = encode(d, vocab);
      for (let i = 0; i < t.length - 1; i++) { seen.add(ctxKey(t, i, n)); events++; }
    }
    let unseen = 0, total = 0;
    for (const d of testDocs) {
      const t = encode(d, vocab);
      for (let i = 0; i < t.length - 1; i++) { total++; if (!seen.has(ctxKey(t, i, n))) unseen++; }
    }
    rows.push({ n, possible: Math.pow(vocab.V, n), seen: seen.size, events, unseenRate: unseen / total });
  }
  return rows;
}

// Deterministic split used everywhere: every 10th name is held out.
export function splitDocs(docs) {
  const train = [], test = [];
  docs.forEach((d, i) => (i % 10 === 0 ? test : train).push(d));
  return { train, test };
}

// ---- Chapter 4: a model with just two dials -------------------------------------------------------
// Predict only "is the next letter a vowel?". Dial a = P(vowel | last letter was a vowel),
// dial b = P(vowel | last letter was a consonant). Only letter-to-letter steps inside names count.
export const VOWELS = 'aeiou';

export function vowelTransitions(docs) {
  let vv = 0, vc = 0, cv = 0, cc = 0;
  for (const d of docs) {
    for (let i = 0; i < d.length - 1; i++) {
      const x = VOWELS.includes(d[i]), y = VOWELS.includes(d[i + 1]);
      if (x && y) vv++; else if (x) vc++; else if (y) cv++; else cc++;
    }
  }
  return { vv, vc, cv, cc, N: vv + vc + cv + cc };
}

export function lossVC(c, a, b) {
  if (!(a > 0 && a < 1 && b > 0 && b < 1)) return Infinity;
  return -(c.vv * Math.log(a) + c.vc * Math.log(1 - a) + c.cv * Math.log(b) + c.cc * Math.log(1 - b)) / c.N;
}

// The counted answer: the dial settings that give the lowest possible loss.
export const bestVC = (c) => ({ a: c.vv / (c.vv + c.vc), b: c.cv / (c.cv + c.cc) });

// ---- Chapter 5: slopes ---------------------------------------------------------------------------
// Exact slopes of the two-dial loss (what calculus gives), and the "wiggle" estimate that needs nothing but the loss itself.
export function slopesVC(c, a, b) {
  return { da: -(c.vv / a - c.vc / (1 - a)) / c.N, db: -(c.cv / b - c.cc / (1 - b)) / c.N };
}
export function wiggleVC(c, a, b, h = 1e-5) {
  const L = lossVC(c, a, b);
  return { da: (lossVC(c, a + h, b) - L) / h, db: (lossVC(c, a, b + h) - L) / h };
}
// Plain gradient descent on the two dials. Returns every position visited.
export function descendVC(c, { a = 0.5, b = 0.5, lr = 0.2, steps = 60 } = {}) {
  const path = [{ a, b, loss: lossVC(c, a, b) }];
  for (let i = 0; i < steps; i++) {
    const g = wiggleVC(c, a, b);
    a -= lr * g.da; b -= lr * g.db;
    const l = lossVC(c, a, b);
    path.push({ a, b, loss: l });
    if (!Number.isFinite(l)) break;
  }
  return path;
}

// ---- Chapter 8: softmax and the trainable bigram table --------------------------------------------
export function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}
// How the loss -log p[target] changes if you nudge each score: "what you predicted minus what happened".
export function errorSignal(z, target) {
  return softmax(z).map((p, i) => p - (i === target ? 1 : 0));
}

// A fixed sample of names used by the "switch a head off" experiment (Chapter 11), so every number quoted is reproducible.
export const ablationSample = (docs) => docs.filter((_, i) => i % 16 === 3).slice(0, 600);
