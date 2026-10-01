// Random hill-climbing, the "try a random nudge, keep it if it helps" strategy of Chapter 4.
import { makeRng } from './rng.js';

export function gauss(rng) {
  let u = 0, v = 0;
  while (!u) u = rng();
  while (!v) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// A bowl-shaped problem with d dials: loss = average squared distance from a hidden best setting.
// Start with every dial at 0. Each attempt nudges ALL dials randomly and keeps the nudge only if loss drops.
export function bowlRun(d, seed, { eps = 0.01, cap = 20000, sigma = 0.3 / Math.sqrt(d), keepEvery = 1 } = {}) {
  const r = makeRng(seed);
  const target = Array.from({ length: d }, () => 2 * r() - 1);
  const L = (x) => x.reduce((s, xi, i) => s + (xi - target[i]) ** 2, 0) / d;
  let x = new Array(d).fill(0);
  let best = L(x);
  const history = [best];
  for (let n = 1; n <= cap; n++) {
    const y = x.map((xi) => xi + sigma * gauss(r));
    const l = L(y);
    if (l < best) { x = y; best = l; }
    if (n % keepEvery === 0) history.push(best);
    if (best < eps) return { attempts: n, history };
  }
  return { attempts: cap, history };
}

export function medianAttempts(d, seeds = [1, 2, 3, 4, 5]) {
  const a = seeds.map((s) => bowlRun(d, s).attempts).sort((p, q) => p - q);
  return a[Math.floor(a.length / 2)];
}
