<script>
  import { getContext } from 'svelte';
  import { runOpt } from '../lib/opt.js';

  // Three optimisers race down a valley where one dial is B times steeper than the other.
  const beat = getContext('beat');
  let B = $state(1000);
  let mult = $state(0.9);        // plain gradient descent's learning rate, as a fraction of its explosion limit
  let acts = new Set();
  const limit = $derived(2 / B);
  const runs = $derived([
    { k: 'gd', name: 'Plain gradient descent', note: `learning rate ${(mult * limit).toPrecision(2)} (${mult.toFixed(2)}× its explosion limit)`, r: runOpt('gd', mult * limit, B), c: 'var(--series-2)' },
    { k: 'mom', name: 'Momentum', note: `learning rate ${(0.1 * limit).toPrecision(2)}`, r: runOpt('mom', 0.1 * limit, B), c: 'var(--series-4)' },
    { k: 'adam', name: 'Adam', note: 'learning rate 0.1 (the same for every steepness)', r: runOpt('adam', 0.1, B), c: 'var(--series-1)' },
  ]);
  const W = 520, H = 220, ML = 44, MR = 12, MT = 10, MB = 28;
  const X = (t) => ML + (Math.log10(Math.max(1, t)) / Math.log10(3000)) * (W - ML - MR);
  const Y = (l) => { const e = Math.max(-8, Math.min(2, Math.log10(Math.max(l, 1e-12)))); return MT + (1 - (e + 8) / 10) * (H - MT - MB); };
  const path = (h) => h.map((l, t) => `${t ? 'L' : 'M'}${X(t + 1).toFixed(1)},${Y(l).toFixed(1)}`).join(' ');
  function setB(v) { B = v; acts.add('B' + v); if (acts.size >= 3) beat?.complete(); }
  const fmt = (r) => (r.exploded ? 'exploded' : r.hit > 0 ? `${r.hit.toLocaleString()} steps` : 'not there after 3,000');
</script>

<div class="widget wide">
  <div class="widget-title">The race · one steep dial, one shallow dial</div>
  <div class="ctl">
    <span class="lab">How much steeper is the steep dial?</span>
    {#each [30, 1000, 100000] as v}<button class="chip" class:on={B === v} onclick={() => setB(v)}>{v.toLocaleString()}×</button>{/each}
    <label class="lab">Plain descent's learning rate <input type="range" min="0.2" max="1.3" step="0.05" bind:value={mult} oninput={() => acts.add('lr')} aria-label="Plain descent learning rate as a fraction of its limit" /> <strong>{mult.toFixed(2)}×</strong> limit</label>
  </div>
  <svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Loss against steps for three optimisers, log-log">
    {#each [-6, -3, 0] as t}<line x1={ML} x2={W - MR} y1={Y(10 ** t)} y2={Y(10 ** t)} stroke="var(--grid)" /><text class="axis-text" x={ML - 5} y={Y(10 ** t) + 4} text-anchor="end">1e{t}</text>{/each}
    {#each [1, 10, 100, 1000] as t}<text class="axis-text" x={X(t)} y={H - 10} text-anchor="middle">{t}</text>{/each}
    <line x1={ML} x2={W - MR} y1={Y(1e-3)} y2={Y(1e-3)} stroke="var(--ink-3)" stroke-width="1" opacity="0.6" /><text class="axis-text" x={W - MR} y={Y(1e-3) - 4} text-anchor="end">solved (loss 0.001)</text>
    <text class="axis-text" x={(ML + W - MR) / 2} y={H - 0} text-anchor="middle" style="font-size:10px">steps (log scale)</text>
    {#each runs as r}<path d={path(r.r.hist)} fill="none" style="stroke: {r.c}; stroke-width: 2;" stroke-linejoin="round" />{/each}
  </svg>
  <div class="rows">
    {#each runs as r}
      <div class="row"><span class="sw" style="background:{r.c}"></span><span class="nm">{r.name}<small>{r.note}</small></span><span class="res" class:bad={r.r.exploded || r.r.hit < 0}>{fmt(r.r)}</span></div>
    {/each}
  </div>
  <div class="cap">Try the steepness buttons, then push plain descent's learning rate past 1.0× its limit. (Loss is for a bowl 0.5(x² + B·y²), starting at x = −3, y = 1.2.)</div>
</div>

<style>
  .ctl { display: flex; gap: 0.5rem 0.9rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.7rem; } .lab { color: var(--ink-2); font-size: 0.85rem; display: flex; gap: 0.5rem; align-items: center; } .lab input { accent-color: var(--accent); width: 130px; }
  .chip { padding: 0.3rem 0.8rem; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); font-size: 0.8rem; } .chip.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .chart { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 12px; }
  .rows { margin-top: 0.7rem; display: grid; gap: 0.3rem; } .row { display: grid; grid-template-columns: 14px 1fr auto; gap: 0.7rem; align-items: center; padding: 0.35rem 0.6rem; background: var(--surface-2); border-radius: 9px; } .sw { width: 14px; height: 4px; border-radius: 2px; }
  .nm small { display: block; color: var(--ink-3); font-size: 0.74rem; } .res { font-weight: 700; font-variant-numeric: tabular-nums; } .res.bad { color: var(--pain); } .cap { margin-top: 0.6rem; color: var(--ink-3); font-size: 0.8rem; }
</style>
