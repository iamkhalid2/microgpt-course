<script>
  import { getContext } from 'svelte';
  import { makeRng } from '../lib/rng.js';
  import { gauss } from '../lib/optim.js';

  // A moving average of noisy numbers (momentum), and why it needs a start-up correction.
  const beat = getContext('beat');
  let beta = $state(0.9);
  let fix = $state(false);
  let seed = $state(3);
  let acts = new Set();
  const N = 50;
  const series = $derived.by(() => { const r = makeRng(seed), g = () => gauss(r); return Array.from({ length: N }, (_, t) => 2 + 0.6 * Math.sin(t / 7) + 1.0 * g()); });
  const m = $derived.by(() => { let a = 0; return series.map((y, t) => { a = beta * a + (1 - beta) * y; return fix ? a / (1 - beta ** (t + 1)) : a; }); });
  const W = 520, H = 210, ML = 34, MR = 10, MT = 10, MB = 22;
  const X = (t) => ML + (t / (N - 1)) * (W - ML - MR), Y = (v) => MT + (1 - (v + 1) / 6) * (H - MT - MB);
  const path = (a) => a.map((v, t) => `${t ? 'L' : 'M'}${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  const touch = (k) => { acts.add(k); if (acts.size >= 3) beat?.complete(); };
</script>

<div class="widget wide">
  <div class="widget-title">Momentum · smoothing a noisy slope</div>
  <svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Noisy numbers and their moving average">
    {#each [0, 2, 4] as t}<line x1={ML} x2={W - MR} y1={Y(t)} y2={Y(t)} stroke="var(--grid)" /><text class="axis-text" x={ML - 5} y={Y(t) + 4} text-anchor="end">{t}</text>{/each}
    <line x1={ML} x2={W - MR} y1={Y(2)} y2={Y(2)} stroke="var(--ink-3)" opacity="0.5" />
    {#each series as y, t}<circle cx={X(t)} cy={Y(y)} r="2.6" style="fill: var(--ink-3); opacity: 0.55;" />{/each}
    <path d={path(m)} fill="none" stroke="var(--series-1)" stroke-width="2.4" stroke-linejoin="round" />
    <text class="axis-text" x={ML} y={H - 6}>step 1</text><text class="axis-text" x={W - MR} y={H - 6} text-anchor="end">step {N}</text>
  </svg>
  <div class="ctl">
    <label>Memory, β₁ <input type="range" min="0" max="0.98" step="0.01" bind:value={beta} oninput={() => touch('b')} aria-label="Memory" /> <strong>{beta.toFixed(2)}</strong></label>
    <label class="chk"><input type="checkbox" bind:checked={fix} onchange={() => touch('fix')} /> start-up correction (÷ (1 − β₁<sup>step</sup>))</label>
    <button class="btn ghost" onclick={() => { seed++; touch('seed'); }}>New noise</button>
  </div>
  <div class="cap"><strong>Grey dots</strong>: the slope measured at each step, wobbling around its true value (the line near 2). <strong>Blue line</strong>: each step, keep β₁ of the old average and mix in (1 − β₁) of the new reading: <span class="mono">m = β₁·m + (1 − β₁)·slope</span>. A longer memory gives a smoother line, but the average starts at 0, so without the correction it climbs up from the floor instead of starting at the right level.</div>
</div>

<style>
  .chart { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 12px; }
  .ctl { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin-top: 0.7rem; } .ctl label { display: flex; gap: 0.5rem; align-items: center; color: var(--ink-2); font-size: 0.85rem; } .ctl input[type='range'] { accent-color: var(--accent); width: 140px; }
  .cap { margin-top: 0.7rem; color: var(--ink-2); font-size: 0.85rem; }
</style>
