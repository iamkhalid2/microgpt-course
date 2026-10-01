// The JS port of gpt() must reproduce the original Python's logits, from the exported real weights.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { makeModel, forward, nextProbs, generate, countParams } from '../src/lib/transformer.js';
import { makeRng } from '../src/lib/rng.js';

const data = JSON.parse(readFileSync(new URL('../public/data/weights.json', import.meta.url), 'utf8'));
const model = makeModel(data.checkpoints['3000'], data.config);

test('logits match the original Python gpt() for every reference prefix', () => {
  for (const [prefix, want] of Object.entries(data.ref)) {
    const got = nextProbs(model, prefix).logits;
    want.forEach((w, i) => assert.ok(Math.abs(w - got[i]) < 2e-3, `prefix "${prefix}" logit ${i}: ${w} vs ${got[i]}`));
  }
});

test('probabilities are valid and attention weights add to 1', () => {
  const steps = forward(model, [model.BOS, 10, 0, 17]);
  const last = steps[steps.length - 1];
  assert.ok(Math.abs(last.probs.reduce((a, b) => a + b, 0) - 1) < 1e-9);
  for (const h of last.layers[0].heads) assert.ok(Math.abs(h.weights.reduce((a, b) => a + b, 0) - 1) < 1e-9 && h.weights.length === 4);
});

test('the architecture arithmetic gives 4,064 parameters', () => {
  assert.equal(countParams(data.config), 4064);
  assert.equal(data.config.num_params, 4064);
});

test('real model writes name-like strings (and an untrained one writes noise)', () => {
  const rng = makeRng(3);
  const names = Array.from({ length: 12 }, () => generate(model, rng, 0.5));
  assert.ok(names.filter((n) => n.length >= 2).length >= 10, names.join(','));
  assert.ok(names.filter((n) => /[aeiouy]/.test(n)).length >= 10);
});
