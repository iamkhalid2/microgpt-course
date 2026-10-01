import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runOpt, stepSizes, cosineLr } from '../src/lib/opt.js';

test('Chapter 15 race: steps to reach loss < 0.001 (numbers quoted in the chapter)', () => {
  for (const [B, gdSteps, adamSteps] of [[30, 68, 63], [1000, 2335, 85], [100000, -1, 132]]) {
    const limit = 2 / B;
    assert.equal(runOpt('gd', 0.9 * limit, B).hit, gdSteps, `gd B=${B}`);
    assert.equal(runOpt('adam', 0.1, B).hit, adamSteps, `adam B=${B}`);
    assert.equal(runOpt('gd', 1.1 * limit, B).exploded, true, 'just above the stability limit plain descent explodes');
  }
  assert.ok(runOpt('adam', 0.05, 30).hit > 0 && runOpt('adam', 1.0, 30).hit > 0, 'Adam converges for a 20x range of learning rates');
});

test('Adam takes steps of about lr whatever the slope size; plain descent scales with the slope', () => {
  for (const g of [0.001, 0.1, 10, 1000]) {
    stepSizes('adam', g).forEach((s) => assert.ok(Math.abs(s - 0.01) < 5e-4, `g=${g} step=${s}`));
    assert.ok(Math.abs(stepSizes('plain', g)[0] - 0.01 * g) < 1e-12);
  }
});

test('cosine decay: full lr at the start, half at the middle, zero at the end', () => {
  assert.equal(cosineLr(0.01, 0, 500), 0.01);
  assert.ok(Math.abs(cosineLr(0.01, 250, 500) - 0.005) < 1e-12);
  assert.ok(Math.abs(cosineLr(0.01, 500, 500)) < 1e-12);
});
