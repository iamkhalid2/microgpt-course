// Pins the numbers quoted in Chapter 17 (300 names per temperature, seed 7, the fully trained model).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { makeModel, generate } from '../src/lib/transformer.js';
import { parseDocs } from '../src/lib/stats.js';
import { makeRng } from '../src/lib/rng.js';

const w = JSON.parse(readFileSync(new URL('../public/data/weights.json', import.meta.url), 'utf8'));
const set = new Set(parseDocs(readFileSync(new URL('../../input.txt', import.meta.url), 'utf8')));
const model = makeModel(w.checkpoints['3000'], w.config);

test('temperature trades reliability for variety (Chapter 17 table)', () => {
  const stat = (T) => { const r = makeRng(7); const out = Array.from({ length: 300 }, () => generate(model, r, T)); return { real: Math.round(out.filter((n) => set.has(n)).length / 3), uniq: Math.round(new Set(out).size / 3) }; };
  const cold = stat(0.1), mid = stat(0.5), hot = stat(2.0);
  assert.ok(cold.real > 55 && cold.uniq < 25, JSON.stringify(cold));
  assert.ok(mid.real > 25 && mid.real < 45 && mid.uniq > 90, JSON.stringify(mid));
  assert.ok(hot.real < 6 && hot.uniq > 90, JSON.stringify(hot));
});
