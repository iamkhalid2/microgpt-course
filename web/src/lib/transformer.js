// A line-for-line port of gpt() from microgpt.py, on plain numbers, that also RETURNS every intermediate value
// (so the page can show attention weights, MLP activations, and so on). Tested against the original Python.
import { sampleIndex } from './rng.js';
import { softmax } from './stats.js';

export const linear = (x, w) => w.map((row) => { let s = 0; for (let i = 0; i < x.length; i++) s += row[i] * x[i]; return s; });
export function rmsnorm(x) {
  let ms = 0;
  for (const v of x) ms += v * v;
  ms /= x.length;
  const scale = Math.pow(ms + 1e-5, -0.5);
  return x.map((v) => v * scale);
}
const add = (a, b) => a.map((v, i) => v + b[i]);

export function makeModel(weights, config) {
  const { n_embd, n_head, n_layer, block_size } = config;
  return { w: weights, n_embd, n_head, n_layer, block_size, head_dim: n_embd / n_head, vocab: config.vocab_size, uchars: config.uchars, BOS: config.uchars.length };
}

// Forward one whole sequence of token ids, position by position (keys/values accumulate, as in microgpt's KV cache).
export function forward(model, ids, opts = {}) {
  const ablate = opts.ablate ?? new Set();       // heads (by number) whose output is switched off, for experiments
  const { w, n_layer, n_head, head_dim } = model;
  const keys = Array.from({ length: n_layer }, () => []);
  const values = Array.from({ length: n_layer }, () => []);
  const steps = [];
  for (let pos = 0; pos < ids.length; pos++) {
    const tokEmb = w.wte[ids[pos]], posEmb = w.wpe[pos];
    const x0 = add(tokEmb, posEmb);
    let x = rmsnorm(x0);
    const rec = { id: ids[pos], pos, tokEmb, posEmb, x0, x1: x, layers: [] };
    for (let li = 0; li < n_layer; li++) {
      const L = {};
      L.xres = x;
      L.xn = rmsnorm(x);
      const q = linear(L.xn, w[`layer${li}.attn_wq`]), k = linear(L.xn, w[`layer${li}.attn_wk`]), v = linear(L.xn, w[`layer${li}.attn_wv`]);
      keys[li].push(k); values[li].push(v);
      L.q = q; L.k = k; L.v = v;
      L.heads = [];
      const xAttn = [];
      for (let h = 0; h < n_head; h++) {
        const hs = h * head_dim;
        const qh = q.slice(hs, hs + head_dim);
        const logits = keys[li].map((kk) => { let s = 0; for (let j = 0; j < head_dim; j++) s += qh[j] * kk[hs + j]; return s / Math.sqrt(head_dim); });
        const weights = softmax(logits);
        const out = new Array(head_dim).fill(0);
        if (!ablate.has(h)) for (let t = 0; t < weights.length; t++) for (let j = 0; j < head_dim; j++) out[j] += weights[t] * values[li][t][hs + j];
        L.heads.push({ logits, weights, out });
        xAttn.push(...out);
      }
      L.xAttn = xAttn;
      L.attnOut = linear(xAttn, w[`layer${li}.attn_wo`]);
      x = add(L.attnOut, L.xres);
      L.xAfterAttn = x;
      L.xres2 = x;
      L.xn2 = rmsnorm(x);
      L.h = linear(L.xn2, w[`layer${li}.mlp_fc1`]);
      L.act = L.h.map((v) => (v > 0 ? v : 0) ** 2);      // relu then squared, as in microgpt
      L.mlpOut = opts.noMlp ? new Array(model.n_embd).fill(0) : linear(L.act, w[`layer${li}.mlp_fc2`]);
      x = add(L.mlpOut, L.xres2);
      L.xOut = x;
      rec.layers.push(L);
    }
    rec.logits = linear(x, w.lm_head);
    rec.probs = softmax(rec.logits);
    steps.push(rec);
  }
  return steps;
}

// Next-symbol distribution after a prefix string (the model starts from BOS).
export function nextProbs(model, prefix, temperature = 1) {
  const ids = [model.BOS, ...[...prefix].map((c) => model.uchars.indexOf(c))];
  if (ids.some((i) => i < 0)) return null;
  const steps = forward(model, ids.slice(-model.block_size));
  const last = steps[steps.length - 1];
  return { steps, logits: last.logits, probs: temperature === 1 ? last.probs : softmax(last.logits.map((l) => l / temperature)) };
}

export function generate(model, rng, temperature = 0.5) {
  let name = '';
  const ids = [model.BOS];
  for (let pos = 0; pos < model.block_size; pos++) {
    const steps = forward(model, ids);
    const logits = steps[steps.length - 1].logits;
    const probs = softmax(logits.map((l) => l / temperature));
    const t = sampleIndex(probs, rng);
    if (t === model.BOS) break;
    name += model.uchars[t];
    ids.push(t);
  }
  return name;
}

// Number of dials, derived from the architecture (the arithmetic of Chapter 14).
export function countParams({ vocab_size, n_embd, n_layer, block_size }) {
  const emb = vocab_size * n_embd + block_size * n_embd + vocab_size * n_embd;
  const perLayer = 4 * n_embd * n_embd + 4 * n_embd * n_embd + 4 * n_embd * n_embd;
  return emb + n_layer * perLayer;
}


// Average loss of the model on some names (same definition as microgpt's training loss: -log p(next symbol), averaged over positions).
export function modelLoss(model, names, opts = {}) {
  let total = 0, n = 0;
  for (const name of names) {
    const ids = [model.BOS, ...[...name].map((c) => model.uchars.indexOf(c)), model.BOS].slice(0, model.block_size + 1);
    const steps = forward(model, ids.slice(0, -1), opts);
    for (let i = 0; i < steps.length; i++) { total += -Math.log(steps[i].probs[ids[i + 1]]); n++; }
  }
  return total / n;
}
