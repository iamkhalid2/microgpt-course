// Principal components by power iteration: squash many-number points down to a few axes, keeping as much spread as possible.
export function pca(X, k = 2) {
  const n = X.length, m = X[0].length;
  const mean = new Array(m).fill(0);
  X.forEach((r) => r.forEach((v, j) => (mean[j] += v / n)));
  const C = X.map((r) => r.map((v, j) => v - mean[j]));
  let M = Array.from({ length: m }, () => new Array(m).fill(0));
  C.forEach((r) => { for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) M[i][j] += (r[i] * r[j]) / n; });
  const comps = [];
  for (let c = 0; c < k; c++) {
    let v = Array.from({ length: m }, (_, i) => Math.sin(i + 1 + c));
    for (let it = 0; it < 400; it++) {
      const w = M.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));
      const nn = Math.hypot(...w) || 1;
      v = w.map((x) => x / nn);
    }
    const Mv = M.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));
    const lam = Mv.reduce((s, x, i) => s + x * v[i], 0);
    comps.push({ v, lam });
    M = M.map((r, i) => r.map((x, j) => x - lam * v[i] * v[j]));
  }
  const project = (Y) => Y.map((r) => comps.map((c) => r.reduce((s, x, j) => s + (x - mean[j]) * c.v[j], 0)));
  return { mean, comps, project };
}
