// A tiny classifier for the XOR-like pattern (label depends on whether x and y have the same sign), to show why a
// "hidden layer" needs a nonlinearity. Full-batch gradient descent, manual gradients, deterministic.
import { makeRng } from './rng.js';
import { gauss } from './optim.js';

export function makeData(seed = 11, n = 240) {
  const rng = makeRng(seed);
  return Array.from({ length: n }, () => { const x = 2 * rng() - 1, y = 2 * rng() - 1; return { x, y, t: x * y > 0 ? 1 : 0 }; });
}

// mode: 'none' (no hidden layer) | 'linear' (hidden layer, no nonlinearity) | 'relu' (hidden layer with ReLU)
export function makeNet(mode, H = 8, seed = 3) {
  const r = makeRng(seed), g = () => gauss(r);
  const hidden = mode === 'none' ? 0 : H;
  return {
    mode, H: hidden, steps: 0,
    W1: Array.from({ length: hidden }, () => [0.5 * g(), 0.5 * g()]), b1: new Array(hidden).fill(0),
    W2: Array.from({ length: hidden }, () => 0.5 * g()), b2: 0,
    w: [0.1 * g(), 0.1 * g()], b: 0,
  };
}
const sig = (z) => 1 / (1 + Math.exp(-z));
export function logit(net, x, y) {
  if (net.H === 0) return net.w[0] * x + net.w[1] * y + net.b;
  let z = net.b2;
  for (let j = 0; j < net.H; j++) {
    const pre = net.W1[j][0] * x + net.W1[j][1] * y + net.b1[j];
    z += (net.mode === 'relu' ? Math.max(0, pre) : pre) * net.W2[j];
  }
  return z;
}
export function trainSteps(net, pts, n, lr = 0.3) {
  const N = pts.length, H = net.H, relu = net.mode === 'relu';
  for (let s = 0; s < n; s++) {
    if (H === 0) {
      let gw0 = 0, gw1 = 0, gb = 0;
      for (const p of pts) { const d = sig(logit(net, p.x, p.y)) - p.t; gw0 += d * p.x; gw1 += d * p.y; gb += d; }
      net.w[0] -= (lr * gw0) / N; net.w[1] -= (lr * gw1) / N; net.b -= (lr * gb) / N;
    } else {
      const gW1 = net.W1.map(() => [0, 0]), gb1 = new Array(H).fill(0), gW2 = new Array(H).fill(0);
      let gb2 = 0;
      for (const p of pts) {
        const pre = net.W1.map((r, j) => r[0] * p.x + r[1] * p.y + net.b1[j]);
        const h = pre.map((v) => (relu ? Math.max(0, v) : v));
        let z = net.b2; for (let j = 0; j < H; j++) z += h[j] * net.W2[j];
        const d = sig(z) - p.t;
        for (let j = 0; j < H; j++) {
          gW2[j] += d * h[j];
          const dh = d * net.W2[j] * (relu ? (pre[j] > 0 ? 1 : 0) : 1);
          gW1[j][0] += dh * p.x; gW1[j][1] += dh * p.y; gb1[j] += dh;
        }
        gb2 += d;
      }
      for (let j = 0; j < H; j++) { net.W1[j][0] -= (lr * gW1[j][0]) / N; net.W1[j][1] -= (lr * gW1[j][1]) / N; net.b1[j] -= (lr * gb1[j]) / N; net.W2[j] -= (lr * gW2[j]) / N; }
      net.b2 -= (lr * gb2) / N;
    }
    net.steps++;
  }
}
export const accuracy = (net, pts) => pts.filter((p) => (logit(net, p.x, p.y) > 0 ? 1 : 0) === p.t).length / pts.length;
export const probAt = (net, x, y) => sig(logit(net, x, y));
