import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as S from '../src/lib/stats.js';
import { makeNB, stepNB, lossNB } from '../src/lib/nb.js';

const docs = S.parseDocs(readFileSync(new URL('../../input.txt', import.meta.url), 'utf8'));
const vocab = S.buildVocab(docs);
const counts = S.bigramCounts(docs, vocab);

test('softmax: positive, sums to 1, shift-invariant, and survives huge scores', () => {
  const p = S.softmax([1, 2, 3]);
  assert.ok(Math.abs(p[0] - 0.09003057) < 1e-7 && Math.abs(p[2] - 0.66524096) < 1e-7);
  assert.ok(Math.abs(S.sum(S.softmax([5, -3, 0.5, 2])) - 1) < 1e-12);
  assert.ok(S.softmax([1000, 1001]).every(Number.isFinite));
  const q = S.softmax([1, 2, 3]), r = S.softmax([101, 102, 103]);
  q.forEach((x, i) => assert.ok(Math.abs(x - r[i]) < 1e-12));
});

test('error signal = predicted minus actual, and matches the wiggle test', () => {
  const z = [0.3, -1.2, 2.0, 0.5], t = 2, h = 1e-6;
  const g = S.errorSignal(z, t);
  const L = (zz) => -Math.log(S.softmax(zz)[t]);
  z.forEach((_, i) => { const up = z.slice(); up[i] += h; assert.ok(Math.abs(g[i] - (L(up) - L(z)) / h) < 1e-4); });
});

test('a trainable table learns the counting table (quoted in Chapter 8: ~2.4548 vs 2.4540)', () => {
  const nb = makeNB(counts);
  assert.ok(Math.abs(lossNB(nb) - Math.log(27)) < 0.01);           // untrained = pure chance
  for (let i = 0; i < 2000; i++) stepNB(nb, 40);
  const learned = lossNB(nb), counted = S.lossBigram(docs, vocab);
  assert.ok(learned - counted < 0.002 && learned >= counted - 1e-9, `${learned} vs ${counted}`);
  // learned dial differences equal log-count ratios
  const r = vocab.uchars.indexOf('e'), a = vocab.uchars.indexOf('a'), n = vocab.uchars.indexOf('n');
  assert.ok(Math.abs((nb.W[r][a] - nb.W[r][n]) - Math.log(counts[r][a] / counts[r][n])) < 0.05);
});
