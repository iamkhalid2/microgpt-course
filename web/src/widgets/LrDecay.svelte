<script>
  import { getContext } from 'svelte';
  import { cosineLr } from '../lib/opt.js';

  // microgpt's learning-rate schedule: big steps early, careful steps late (lines 175).
  const beat = getContext('beat');
  let step = $state(0);
  let total = $state(500);
  let acts = 0;
  const LR = 0.01;
  const cur = $derived(cosineLr(LR, step, total));
  const W = 520, H = 170, ML = 44, MR = 10, MT = 10, MB = 24;
  const X = (s) => ML + (s / total) * (W - ML - MR), Y = (v) => MT + (1 - v / LR) * (H - MT - MB);
  const curve = $derived(Array.from({ length: 101 }, (_, i) => { const s = (i / 100) * total; return `${i ? 'L' : 'M'}${X(s).toFixed(1)},${Y(cosineLr(LR, s, total)).toFixed(1)}`; }).join(' '));
  const touch = () => { if (++acts >= 3) beat?.complete(); };
</script>

<div class="widget wide">
  <div class="widget-title">Learning-rate decay · the cosine schedule</div>
  <svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Learning rate falling from 0.01 to 0 over training">
    {#each [0, 0.005, 0.01] as t}<line x1={ML} x2={W - MR} y1={Y(t)} y2={Y(t)} stroke="var(--grid)" /><text class="axis-text" x={ML - 5} y={Y(t) + 4} text-anchor="end">{t}</text>{/each}
    <path d={curve} fill="none" stroke="var(--series-1)" stroke-width="2.4" />
    <line x1={X(step)} x2={X(step)} y1={MT} y2={H - MB} stroke="var(--ink-3)" opacity="0.6" />
    <circle cx={X(step)} cy={Y(cur)} r="5.5" style="fill: var(--series-1); stroke: var(--surface); stroke-width: 2.5;" />
    <text class="axis-text" x={ML} y={H - 6}>step 0</text><text class="axis-text" x={W - MR} y={H - 6} text-anchor="end">step {total}</text>
  </svg>
  <div class="ctl">
    <label>Training step <input type="range" min="0" max={total} step="1" bind:value={step} oninput={touch} aria-label="Training step" /> <strong>{step}</strong></label>
    <label>Total steps <select bind:value={total} onchange={() => { step = Math.min(step, total); touch(); }}><option value={500}>500 (the file)</option><option value={3000}>3,000</option></select></label>
  </div>
  <div class="calc"><span class="mono">lr_t = 0.01 × 0.5 × (1 + cos(π × {step} / {total})) = <strong>{cur.toFixed(5)}</strong></span></div>
  <div class="cap">Early on the dials are far from good settings, so big steps pay off. Near the end they are close, and big steps would overshoot, so the steps shrink smoothly to nothing. This is the same trade-off you saw with the step size in Chapters 4 and 5, now handled automatically.</div>
</div>

<style>
  .chart { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 12px; } .ctl { display: flex; gap: 1.2rem; align-items: center; flex-wrap: wrap; margin: 0.7rem 0; } .ctl label { display: flex; gap: 0.6rem; align-items: center; color: var(--ink-2); font-size: 0.85rem; } .ctl input { accent-color: var(--accent); width: 200px; }
  select { padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); }
  .calc { margin: 0.4rem 0; padding: 0.5rem 0.8rem; background: var(--code-bg); border-radius: 8px; overflow-x: auto; font-size: 0.85rem; } .cap { color: var(--ink-2); font-size: 0.85rem; margin-top: 0.4rem; }
</style>
