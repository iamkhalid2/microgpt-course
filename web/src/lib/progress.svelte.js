// Everything the learner has done, saved in their browser (localStorage). Nothing leaves the device.
import { UNLOCKS, CODE_LINES } from './fileMap.js';

const KEY = 'igpt.progress.v1';
function load() { try { return JSON.parse(localStorage.getItem(KEY)) ?? {}; } catch { return {}; } }
const saved = load();

export const progress = $state({
  banked: saved.banked ?? {},   // chapterId -> true once the lesson's Bank is claimed
  beat: saved.beat ?? {},       // chapterId -> furthest beat revealed
  answers: saved.answers ?? {}, // predictId -> what you guessed
  code: saved.code ?? {},       // cellId -> your edited code
  solved: saved.solved ?? {},   // cellId -> true once its check passed
  game: saved.game ?? {},       // fake-or-fact score keeping
  reader: saved.reader ?? false, // show every beat at once (skip the drip)
  mode: saved.mode ?? 'plain'    // how exercises are done by default: 'plain' (English recipe) or 'code' (Python editor)
});

$effect.root(() => {
  $effect(() => {
    const snap = $state.snapshot(progress);
    try { localStorage.setItem(KEY, JSON.stringify(snap)); } catch {}
  });
});

export function bank(chapterId) { progress.banked[chapterId] = true; }
export function resetAll() { for (const k of Object.keys(progress)) { if (k === 'reader' || k === 'mode') continue; progress[k] = {}; } }

// Which lines of microgpt.py are lit, derived from banked chapters.
export function litLines() {
  const lit = new Map(); // line number -> chapter that unlocked it
  for (const [ch, ranges] of Object.entries(UNLOCKS)) {
    if (!progress.banked[ch]) continue;
    for (const [a, b] of ranges) for (let n = a; n <= b; n++) lit.set(n, ch);
  }
  return lit;
}
export function litCount() {
  const lit = litLines();
  let c = 0;
  for (const n of CODE_LINES) if (lit.has(n)) c++;
  return c;
}
