import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeData, makeNet, trainSteps, accuracy } from '../src/lib/xor.js';

test('Chapter 12 numbers: no hidden layer and a LINEAR hidden layer both fail the same way; ReLU solves it', () => {
  const pts = makeData();
  const acc = (mode, H, steps = 3000) => { const n = makeNet(mode, H); trainSteps(n, pts, steps); return accuracy(n, pts); };
  assert.equal(acc('none').toFixed(3), '0.592');
  assert.equal(acc('linear', 8).toFixed(3), '0.592');
  assert.equal(acc('relu', 8).toFixed(3), '1.000');
  assert.equal(acc('relu', 4).toFixed(3), '0.992');
  assert.equal(acc('relu', 2, 6000).toFixed(3), '0.717');
});
