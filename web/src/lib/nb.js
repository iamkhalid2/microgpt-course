// A bigram model whose 27x27 table is made of trainable dials (logits) instead of counts.
// Full-batch gradient descent on the exact average loss, using the pair counts (so each step "sees" every pair).
import { makeRng } from './rng.js';
import { gauss } from './optim.js';
import { softmax, sum } from './stats.js';

export function makeNB(counts, seed = 5) {
  const V = counts.length, rng = makeRng(seed);
  const W = Array.from({ length: V }, () => Array.from({ length: V }, () => 0.02 * gauss(rng)));
  const rowN = counts.map((r) => sum(r));
  return { V, W, counts, rowN, N: sum(rowN), steps: 0 };
}
export function stepNB(nb, lr) {
  for (let r = 0; r < nb.V; r++) {
    if (!nb.rowN[r]) continue;
    const p = softmax(nb.W[r]), w = nb.rowN[r] / nb.N;
    for (let c = 0; c < nb.V; c++) nb.W[r][c] -= lr * w * (p[c] - nb.counts[r][c] / nb.rowN[r]);
  }
  nb.steps++;
}
export function lossNB(nb) {
  let t = 0;
  for (let r = 0; r < nb.V; r++) {
    if (!nb.rowN[r]) continue;
    const p = softmax(nb.W[r]);
    for (let c = 0; c < nb.V; c++) if (nb.counts[r][c]) t -= nb.counts[r][c] * Math.log(p[c]);
  }
  return t / nb.N;
}
export const probsNB = (nb, r) => softmax(nb.W[r]);
