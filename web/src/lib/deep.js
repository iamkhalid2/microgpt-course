// A stack of MLP blocks with random weights, to show why deep networks need RMSNorm and a residual "skip lane".
// Forward: how big the signal is at each layer. Backward: how big the slope (gradient) is at each layer.
// Hand-written gradients, including the gradient of RMSNorm. Deterministic.
import { makeRng } from './rng.js';
import { gauss } from './optim.js';

const D = 16, H = 64;
const mat = (g, out, inn, std) => Array.from({ length: out }, () => Array.from({ length: inn }, () => std * g()));
const mv = (W, x) => W.map((r) => r.reduce((s, w, i) => s + w * x[i], 0));
const mtv = (W, g, n) => { const o = new Array(n).fill(0); for (let i = 0; i < W.length; i++) for (let j = 0; j < n; j++) o[j] += W[i][j] * g[i]; return o; };
export const rms = (x) => Math.sqrt(x.reduce((s, v) => s + v * v, 0) / x.length);

export function rmsnormParts(x) {
  const ms = x.reduce((s, v) => s + v * v, 0) / x.length;
  const sc = Math.pow(ms + 1e-5, -0.5);
  return { y: x.map((v) => v * sc), sc };
}
// gradient through rmsnorm: dx_j = s*dy_j - (s^3 * x_j / n) * sum_i(dy_i * x_i)
const rmsnormBack = (x, dy, sc) => { const dot = dy.reduce((s, v, i) => s + v * x[i], 0); return x.map((xj, j) => sc * dy[j] - ((sc ** 3) * xj / x.length) * dot); };

export function runDeep({ L = 16, std = 0.02, norm = false, skip = false, seed = 1 } = {}) {
  const r = makeRng(seed), g = () => gauss(r);
  const layers = Array.from({ length: L }, () => ({ W1: mat(g, H, D, std), W2: mat(g, D, H, std) }));
  let x = Array.from({ length: D }, g);
  const fwd = [rms(x)], cache = [];
  for (const l of layers) {
    const xin = x;
    const np = norm ? rmsnormParts(xin) : { y: xin, sc: 1 };
    const pre = mv(l.W1, np.y), act = pre.map((v) => (v > 0 ? v * v : 0)), out = mv(l.W2, act);
    x = skip ? xin.map((v, i) => v + out[i]) : out;
    fwd.push(rms(x));
    cache.push({ xin, np, pre, l });
  }
  // backward from an output slope of 1/sqrt(D) in every place; record the size of the slope at each layer's input
  let gr = new Array(D).fill(1 / Math.sqrt(D));
  const bwd = new Array(L + 1);
  bwd[L] = rms(gr);
  for (let k = L - 1; k >= 0; k--) {
    const { xin, np, pre, l } = cache[k];
    const dact = mtv(l.W2, gr, H);
    const dpre = dact.map((d, j) => (pre[j] > 0 ? 2 * pre[j] * d : 0));
    let dxn = mtv(l.W1, dpre, D);
    if (norm) dxn = rmsnormBack(xin, dxn, np.sc);
    gr = skip ? gr.map((v, i) => v + dxn[i]) : dxn;
    bwd[k] = rms(gr);
  }
  return { fwd, bwd };
}
// Judge the whole journey, not just the last layer (an exploded signal can overflow and read as 0 at the very end).
export const peak = (fwd) => Math.max(...fwd.filter(Number.isFinite));
export const verdict = (fwd) => {
  if (fwd.some((v) => !Number.isFinite(v)) || peak(fwd) > 1e6) return 'exploded';
  if (fwd[fwd.length - 1] < 1e-6) return 'vanished';
  return 'healthy';
};
