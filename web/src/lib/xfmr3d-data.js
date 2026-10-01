// The numbers behind the hero's 3D transformer. Everything here is real: it runs the exported, trained microgpt
// (see transformer.js) on a short name and flattens what comes out of every stage into plain arrays the renderer can tween.

import { forward } from './transformer.js';

// The stages of one pass through the real model, bottom to top. `chapter` is where the course builds that part;
// `lines` are the lines of microgpt.py that do it.
export const LAYERS = [
  { id: 'tokens', short: 'letters in', name: 'Letters in', chapter: 1, lines: '23–27' },
  { id: 'embed', short: 'embeddings', name: 'Embeddings', chapter: 9, lines: '109–112' },
  { id: 'attn', short: 'attention', name: 'Attention, four heads', chapter: 10, lines: '118–133' },
  { id: 'res1', short: 'add back', name: 'Add it back (skip lane)', chapter: 13, lines: '134' },
  { id: 'mlp', short: 'MLP', name: 'The MLP thinks', chapter: 12, lines: '137–140' },
  { id: 'res2', short: 'add back', name: 'Add it back again', chapter: 13, lines: '141' },
  { id: 'out', short: 'next letter', name: 'Odds for the next letter', chapter: 14, lines: '143' },
];

export const NAMES = ['emma', 'mariah', 'olivia', 'kayla', 'ava', 'sophia', 'elena'];
export const TWEENED = ['emb', 'res1', 'hid', 'res2', 'attn', 'probs', 'scales'];

const N = 16;          // numbers per position (n_embd)
const HID = 64;        // neurons in the MLP
const rmsOf = (a) => { let s = 0; for (const v of a) s += v * v; return Math.sqrt(s / Math.max(1, a.length)); };

// Run the real model over "⏎" + the name without its last letter, and keep what each stage produced at every position.
export function buildSnapshot(model, name) {
  const prefix = name.slice(0, -1);
  const ids = [model.BOS, ...[...prefix].map((c) => model.uchars.indexOf(c))].slice(0, model.block_size);
  if (ids.some((i) => i < 0)) throw new Error(`"${name}" has a letter the model does not know`);
  const steps = forward(model, ids);
  const P = steps.length;
  const H = model.n_head;
  const emb = new Float32Array(P * N), res1 = new Float32Array(P * N), res2 = new Float32Array(P * N);
  const hid = new Float32Array(P * HID), attn = new Float32Array(H * P * P);
  steps.forEach((s, i) => {
    const L = s.layers[0];
    emb.set(s.x1, i * N); res1.set(L.xAfterAttn, i * N); res2.set(L.xOut, i * N); hid.set(L.act, i * HID);
    L.heads.forEach((hd, h) => hd.weights.forEach((w, j) => { attn[(h * P + i) * P + j] = w; }));
  });
  const last = steps[P - 1];
  const probs = Float32Array.from(last.probs);
  const label = (id) => (id === model.BOS ? '⏎' : model.uchars[id]);
  const tokens = ids.map(label);

  // Facts for the captions, all measured from this very run.
  let look = null;
  last.layers[0].heads.forEach((hd, h) => hd.weights.forEach((w, j) => { if (!look || w > look.w) look = { head: h, from: j, label: tokens[j], w }; }));
  const awake = last.layers[0].act.reduce((n, v) => n + (v > 0 ? 1 : 0), 0);
  const top = [...probs].map((p, id) => ({ id, p, label: label(id) })).sort((a, b) => b.p - a.p).slice(0, 3);
  const evenAttention = look.w < 1 / P + 0.12;

  // Colour scales: a number this big (2.5 × the sheet's typical size) is drawn at full strength.
  const scales = Float32Array.from([2.5 * rmsOf(emb), 2.5 * rmsOf(res1), Math.max(1e-6, Math.max(...hid)), 2.5 * rmsOf(res2)]);
  return { name, prefix, P, H, tokens, emb, res1, hid, res2, attn, probs, scales, look, awake, top, evenAttention, vocab: probs.length };
}

// Blend two snapshots with the same shape: the picture glides from one training step to the next.
export function blend(a, b, t, out = {}) {
  for (const k of TWEENED) {
    const x = a[k], y = b[k];
    const o = out[k] ?? (out[k] = new Float32Array(y.length));
    for (let i = 0; i < y.length; i++) o[i] = x[i] + (y[i] - x[i]) * t;
  }
  return out;
}

export const sameShape = (a, b) => a && b && a.P === b.P && a.H === b.H;

// What the caption says for a stage, using the real numbers from the current run.
export function captionFor(id, s) {
  const pct = (p) => `${Math.round(p * 100)}%`;
  switch (id) {
    case 'tokens':
      return `The model never sees letters, only numbers. Here it has read “${s.prefix}” (⏎ marks the start) and must guess what comes next.`;
    case 'embed':
      return 'Each letter becomes 16 numbers (one stripe each), blended with 16 more for its position. Orange is positive, blue is negative.';
    case 'attn':
      return s.evenAttention
        ? 'Each letter looks back at earlier ones, with four heads at once. Untrained, every head spreads its attention evenly, so nothing stands out.'
        : `Each letter looks back at earlier ones, with four heads at once. Thicker arcs mean more attention: head ${s.look.head + 1} sends ${pct(s.look.w)} of the last letter’s attention to “${s.look.label}”.`;
    case 'res1':
      return 'What attention found is added to what was already there, along the dashed skip lane, so nothing is lost on the way up.';
    case 'mlp':
      return `Each letter then thinks on its own: 16 numbers expand to 64 neurons and only the positive ones fire. At the last letter, ${s.awake} of 64 are awake.`;
    case 'res2':
      return 'The thinking is added back too. Attention plus the MLP is one whole transformer block. This model has just one.';
    case 'out': {
      const [a, b, c] = s.top;
      return a.p < 0.07
        ? `16 numbers become 27 scores, then probabilities. At this step the model is still guessing: every symbol gets about ${pct(1 / s.vocab)}.`
        : `16 numbers become 27 scores, then probabilities. After “${s.prefix}” it favours “${a.label}” (${pct(a.p)}), then “${b.label}” (${pct(b.p)}) and “${c.label}” (${pct(c.p)}).`;
    }
  }
  return '';
}

// ───────────── tiny 3D maths: a camera that yaws and pitches around the middle of the stack ─────────────

export function makeCamera({ yaw, pitch, cx, cy, k, centerY, dist = 26 }) {
  const cyaw = Math.cos(yaw), syaw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  // x,y,z in world units -> [screen x, screen y, depth (bigger = nearer the viewer)]
  const project = (x, y, z, out = [0, 0, 0]) => {
    const yy = y - centerY;
    const x1 = x * cyaw + z * syaw, z1 = -x * syaw + z * cyaw;
    const y2 = yy * cp - z1 * sp, z2 = yy * sp + z1 * cp;
    const s = (k * dist) / (dist - z2);
    out[0] = cx + x1 * s; out[1] = cy - y2 * s; out[2] = z2;
    return out;
  };
  // The size of one world unit on screen at the middle of the scene (for line widths and text).
  return { project, unit: k };
}

export function pointInPolygon(px, py, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
