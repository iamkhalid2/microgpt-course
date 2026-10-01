import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runDeep, verdict, rmsnormParts, rms } from '../src/lib/deep.js';

test('rmsnorm output has RMS 1 whatever the scale', () => {
  for (const k of [1, 10, 1000]) assert.ok(Math.abs(rms(rmsnormParts([3, -1, 2, 0.5].map((v) => v * k)).y) - 1) < 1e-3);
  // for absurdly tiny inputs the 1e-5 safety term wins (it stops us dividing by ~0), so the output stays small
  assert.ok(rms(rmsnormParts([3, -1, 2, 0.5].map((v) => v * 1e-6)).y) < 0.5);
});

test('numbers quoted in Chapter 13 (16 layers, seed 1)', () => {
  const plain = runDeep({ std: 0.02 });
  assert.ok(plain.fwd[0] > 0.8 && plain.fwd[0] < 0.95);
  assert.ok(plain.fwd[1] > 1e-3 && plain.fwd[1] < 2e-3);
  assert.ok(plain.fwd[2] < 1e-8);
  assert.equal(verdict(plain.fwd), 'vanished');
  const skip = runDeep({ std: 0.02, skip: true });
  assert.ok(skip.fwd.every((v) => Math.abs(v - skip.fwd[0]) < 0.01));
  assert.ok(skip.bwd.every((v) => Math.abs(v - 0.25) < 0.01), 'slope passes undimmed through a skip lane');
  const big = runDeep({ std: 0.2 });
  assert.ok(big.fwd[8] > 1e20);
  assert.equal(verdict(runDeep({ std: 0.2, skip: true }).fwd), 'exploded');
  assert.equal(verdict(runDeep({ std: 0.2 }).fwd), 'exploded', 'a plain stack with bigger dials explodes (even though it later overflows to 0)');
  const both = runDeep({ std: 0.2, skip: true, norm: true });
  assert.equal(verdict(both.fwd), 'healthy');
  assert.ok(Math.max(...both.fwd) < 10);
});

test('gradient through rmsnorm matches the wiggle test', () => {
  const x = [0.4, -1.2, 2.0, 0.7], dy = [0.3, -0.5, 0.2, 0.9], h = 1e-6;
  const f = (xx) => rmsnormParts(xx).y.reduce((s, v, i) => s + v * dy[i], 0);
  const { sc } = rmsnormParts(x);
  const dot = dy.reduce((s, v, i) => s + v * x[i], 0);
  x.forEach((xj, j) => { const up = x.slice(); up[j] += h; const ana = sc * dy[j] - ((sc ** 3) * xj / x.length) * dot; assert.ok(Math.abs(ana - (f(up) - f(x)) / h) < 1e-4); });
});
