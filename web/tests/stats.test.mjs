// The site computes its numbers in JS (so widgets are instant). This test proves those numbers
// equal what independent Python code gets from the same dataset.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import * as S from '../src/lib/stats.js';
import { medianAttempts } from '../src/lib/optim.js';

const here = dirname(fileURLToPath(import.meta.url));
const docs = S.parseDocs(readFileSync(resolve(here, '../../input.txt'), 'utf8'));
const vocab = S.buildVocab(docs);
const py = JSON.parse(execFileSync('python3', [resolve(here, '../tools/facts.py')], { encoding: 'utf8' }));
const close = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-9, `${msg}: ${a} vs ${b}`);

test('dataset + vocabulary', () => {
  assert.equal(docs.length, py.nDocs);
  assert.equal(vocab.uchars.join(''), py.uchars);
  assert.equal(vocab.V, py.V);
});

test('counts', () => {
  assert.deepEqual(S.unigramCounts(docs, vocab), py.unigramCounts);
  assert.deepEqual(S.bigramCounts(docs, vocab)[vocab.BOS], py.bigramRow_BOS);
  assert.equal(S.sum(S.unigramCounts(docs, vocab)), py.totalEvents);
});

test('losses match Python', () => {
  close(S.lossUniform(vocab.V), py.lossUniform, 'uniform');
  close(S.lossUnigram(docs, vocab), py.lossUnigram, 'unigram');
  close(S.lossBigram(docs, vocab), py.lossBigram, 'bigram');
  const { train, test: held } = S.splitDocs(docs);
  close(S.lossUnigram(held, vocab, train), py.lossUnigramHeldOut, 'unigram held-out');
  assert.equal(py.lossBigramHeldOut, null);                       // Python: infinite
  assert.equal(S.lossBigram(held, vocab, train), Infinity);       // JS agrees
  close(S.lossBigram(held, vocab, train, 1), py.lossBigramHeldOutSmooth1, 'bigram held-out, smoothed');
});

test('context coverage grows more sparse with longer context', () => {
  const { train, test: held } = S.splitDocs(docs);
  const rows = S.contextCoverage(train, held, vocab, 5);
  for (let i = 1; i < rows.length; i++) assert.ok(rows[i].unseenRate >= rows[i - 1].unseenRate);
  assert.equal(rows[0].unseenRate, 0);
});

test('two-dial vowel model matches Python', () => {
  const c = S.vowelTransitions(docs);
  assert.deepEqual([c.vv, c.vc, c.cv, c.cc], py.vowelCounts);
  const { a, b } = S.bestVC(c);
  close(a, py.vowelBest[0], 'best a'); close(b, py.vowelBest[1], 'best b');
  close(S.lossVC(c, a, b), py.vowelMinLoss, 'min loss');
  close(S.lossVC(c, 0.5, 0.5), Math.log(2), 'coin-flip loss is ln 2');
  assert.equal(S.lossVC(c, 0, 0.5), Infinity);
});

test('random hill-climbing needs roughly proportionally more attempts per dial (numbers quoted in Chapter 4)', () => {
  const got = [1, 2, 4, 8, 16, 32, 64, 128].map((d) => medianAttempts(d));
  assert.deepEqual(got, [6, 13, 33, 68, 138, 348, 664, 1424]);
});
