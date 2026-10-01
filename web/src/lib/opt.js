// Optimisers racing down a long narrow valley f(x, y) = 0.5 * (x^2 + B * y^2), where the y-dial is B times steeper.
// 'gd' = plain gradient descent, 'mom' = momentum, 'adam' = Adam (beta1 0.9, beta2 0.95, as in microgpt.py).
export const lossAt = (B, x, y) => 0.5 * (x * x + B * y * y);

export function runOpt(kind, lr, B, steps = 3000) {
  let x = -3, y = 1.2, m = [0, 0], v = [0, 0], mom = [0, 0], hit = -1;
  const hist = [lossAt(B, x, y)];
  for (let t = 1; t <= steps; t++) {
    const g = [x, B * y];
    if (kind === 'gd') { x -= lr * g[0]; y -= lr * g[1]; }
    else if (kind === 'mom') { mom = mom.map((mi, i) => 0.9 * mi + g[i]); x -= lr * mom[0]; y -= lr * mom[1]; }
    else {
      m = m.map((mi, i) => 0.9 * mi + 0.1 * g[i]);
      v = v.map((vi, i) => 0.95 * vi + 0.05 * g[i] * g[i]);
      const mh = m.map((mi) => mi / (1 - 0.9 ** t)), vh = v.map((vi) => vi / (1 - 0.95 ** t));
      x -= (lr * mh[0]) / (Math.sqrt(vh[0]) + 1e-8); y -= (lr * mh[1]) / (Math.sqrt(vh[1]) + 1e-8);
    }
    const l = lossAt(B, x, y);
    hist.push(l);
    if (hit < 0 && l < 1e-3) hit = t;
    if (!Number.isFinite(l) || l > 1e12) return { hit: -2, hist, exploded: true };
  }
  return { hit, hist, exploded: false };
}

// Size of each step an optimiser takes on a dial whose slope is always g (to show Adam's per-dial scaling).
export function stepSizes(kind, g, lr = 0.01, n = 12) {
  let m = 0, v = 0;
  const out = [];
  for (let t = 1; t <= n; t++) {
    if (kind === 'plain') out.push(lr * g);
    else { m = 0.9 * m + 0.1 * g; v = 0.95 * v + 0.05 * g * g; out.push((lr * (m / (1 - 0.9 ** t))) / (Math.sqrt(v / (1 - 0.95 ** t)) + 1e-8)); }
  }
  return out;
}
export const cosineLr = (lr, step, total) => lr * 0.5 * (1 + Math.cos((Math.PI * step) / total));
