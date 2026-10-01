// The hero's 3D transformer is drawn from the real trained model, so what it shows must be real numbers.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { makeModel, forward } from '../src/lib/transformer.js';
import { LAYERS, NAMES, buildSnapshot, blend, captionFor, makeCamera, pointInPolygon, sameShape } from '../src/lib/xfmr3d-data.js';

const data = JSON.parse(readFileSync(new URL('../public/data/weights.json', import.meta.url), 'utf8'));
const at = (step) => makeModel(data.checkpoints[step], data.config);

test('every name offered can be run through every checkpoint, and fits the model\'s 8 positions', () => {
  for (const step of Object.keys(data.checkpoints)) {
    for (const name of NAMES) {
      const s = buildSnapshot(at(step), name);
      assert.ok(s.P >= 2 && s.P <= data.config.block_size, `${name}: ${s.P} positions`);
      assert.equal(s.emb.length, s.P * 16);
      assert.equal(s.hid.length, s.P * 64);
      assert.equal(s.attn.length, 4 * s.P * s.P);
    }
  }
});

test('the snapshot holds the model\'s own numbers: probabilities sum to 1 and attention rows sum to 1', () => {
  const m = at('3000');
  const s = buildSnapshot(m, 'emma');
  assert.ok(Math.abs(s.probs.reduce((a, b) => a + b, 0) - 1) < 1e-5);
  for (let h = 0; h < s.H; h++) for (let i = 0; i < s.P; i++) {
    let sum = 0; for (let j = 0; j < s.P; j++) sum += s.attn[(h * s.P + i) * s.P + j];
    assert.ok(Math.abs(sum - 1) < 1e-5, `head ${h} position ${i} adds to ${sum}`);
  }
  // identical to calling forward() directly
  const ids = [m.BOS, ...[...'emm'].map((c) => m.uchars.indexOf(c))];
  const last = forward(m, ids).at(-1);
  last.probs.forEach((p, i) => assert.ok(Math.abs(p - s.probs[i]) < 1e-6));
  assert.deepEqual(s.tokens, ['⏎', 'e', 'm', 'm']);
});

test('untrained: attention is even and the output is a guess; trained: attention sharpens', () => {
  const s0 = buildSnapshot(at('0'), 'emma'), s3 = buildSnapshot(at('3000'), 'emma');
  assert.ok(s0.evenAttention && !s3.evenAttention);
  assert.ok(s0.top[0].p < 0.07 && s3.top[0].p > 0.2);
  assert.ok(s0.top[0].p < 0.07 && /still guessing/.test(captionFor('out', s0)));
});

test('MLP activations are non-negative and the "awake" count matches them', () => {
  const s = buildSnapshot(at('3000'), 'mariah');
  assert.ok(s.hid.every((v) => v >= 0));
  let n = 0; for (let d = 0; d < 64; d++) if (s.hid[(s.P - 1) * 64 + d] > 0) n++;
  assert.equal(n, s.awake);
});

test('blend interpolates every array and leaves the endpoints alone', () => {
  const a = buildSnapshot(at('0'), 'emma'), b = buildSnapshot(at('3000'), 'emma');
  assert.ok(sameShape(a, b));
  const out = {};
  blend(a, b, 0, out); assert.deepEqual([...out.probs], [...a.probs]);
  blend(a, b, 1, out); assert.deepEqual([...out.probs], [...b.probs]);
  blend(a, b, 0.5, out);
  assert.ok(Math.abs(out.probs[3] - (a.probs[3] + b.probs[3]) / 2) < 1e-6);
});

test('every stage has a caption and points at a real chapter', () => {
  const s = buildSnapshot(at('3000'), 'emma');
  for (const L of LAYERS) { assert.ok(captionFor(L.id, s).length > 20, L.id); assert.ok(Number.isInteger(L.chapter) && L.chapter >= 1 && L.chapter <= 19); }
});

test('camera: higher things appear higher, nearer things appear lower on screen, and yaw spins around the middle', () => {
  const cam = makeCamera({ yaw: 0, pitch: 0.6, cx: 100, cy: 100, k: 10, centerY: 5 });
  const mid = cam.project(0, 5, 0), up = cam.project(0, 6, 0), near = cam.project(0, 5, 1);
  assert.ok(up[1] < mid[1]);
  assert.ok(near[1] > mid[1] && near[2] > mid[2]);
  assert.ok(Math.abs(mid[0] - 100) < 1e-9 && Math.abs(mid[1] - 100) < 1e-9);
  const turned = makeCamera({ yaw: Math.PI / 2, pitch: 0.6, cx: 100, cy: 100, k: 10, centerY: 5 }).project(1, 5, 0);
  assert.ok(Math.abs(turned[0] - 100) < 1e-9, 'a point on the x axis lines up with the middle after a quarter turn');
});

test('point-in-polygon', () => {
  const sq = [[0, 0], [10, 0], [10, 10], [0, 10]];
  assert.ok(pointInPolygon(5, 5, sq) && !pointInPolygon(11, 5, sq) && !pointInPolygon(5, -1, sq));
});
