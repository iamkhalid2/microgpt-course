import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pca } from '../src/lib/pca.js';

test('first component of points along a line is that line', () => {
  const X = [[0, 0, 0], [1, 2, 0], [2, 4, 0], [3, 6, 0], [-1, -2, 0]];
  const p = pca(X, 1);
  const v = p.comps[0].v, n = Math.hypot(1, 2);
  assert.ok(Math.abs(Math.abs(v[0]) - 1 / n) < 1e-6 && Math.abs(Math.abs(v[1]) - 2 / n) < 1e-6 && Math.abs(v[2]) < 1e-9);
  const proj = p.project(X).map((r) => r[0]);
  assert.ok(Math.abs(Math.abs(proj[3] - proj[0]) - 3 * Math.hypot(1, 2)) < 1e-6);
});
