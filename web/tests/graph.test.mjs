import { test } from 'node:test';
import assert from 'node:assert/strict';
import { backward, wiggle, PRESETS, nodesOf } from '../src/lib/graph.js';

const close = (a, b, tol = 1e-4) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

test('the worked chain example from Chapter 6: slopes 42 and 28', () => {
  const nodes = nodesOf(PRESETS.chain);
  const r = backward(nodes);
  assert.equal(r.val.loss, 49);
  assert.equal(r.grad.loss, 1);
  assert.equal(r.grad.d, 14);      // d(d^2) = 2d
  assert.equal(r.grad.c, 14);      // adding 1 changes nothing about the slope
  assert.equal(r.grad.a, 42);      // 14 x b
  assert.equal(r.grad.b, 28);      // 14 x a
});

test('every preset: reverse-mode slopes equal the wiggle test', () => {
  for (const [name, p] of Object.entries(PRESETS)) {
    const nodes = nodesOf(p);
    const r = backward(nodes);
    for (const id of p.edit) close(r.grad[id], wiggle(nodes, {}, id), 1e-3);
  }
});

test('an input used twice collects slope from both paths (a*a has slope 2a)', () => {
  const r = backward(nodesOf(PRESETS.twice));
  assert.equal(r.grad.a, 6);
});

import { topoTrace } from '../src/lib/graph.js';
test('topological order puts every card after the cards it was made from', () => {
  for (const p of Object.values(PRESETS)) {
    const nodes = nodesOf(p);
    const root = nodes[nodes.length - 1].id;
    const { order } = topoTrace(nodes, root);
    for (const n of nodes) for (const inp of n.inputs ?? []) assert.ok(order.indexOf(inp) < order.indexOf(n.id), `${inp} before ${n.id}`);
    assert.equal(new Set(order).size, order.length);
  }
});
