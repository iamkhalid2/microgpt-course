<script>
  import { getContext } from 'svelte';
  import { rmsnormParts, rms } from '../lib/deep.js';

  // RMSNorm: divide a list by its own typical size, so it always has typical size 1.
  const beat = getContext('beat');
  let x = $state([2, -1, 0.5, 3, -2]);
  let k = $state(1);
  let acts = 0;
  const touch = () => { if (++acts >= 4) beat?.complete(); };
  const scaled = $derived(x.map((v) => v * k));
  const out = $derived(rmsnormParts(scaled).y);
  const f = (v) => v.toFixed(2);
  const mx = $derived(Math.max(...scaled.map(Math.abs), 1e-9));
</script>

<div class="widget wide">
  <div class="widget-title">RMSNorm · the same shape at any volume</div>
  <div class="grid">
    <div>
      <div class="h">The list, before</div>
      {#each x as v, i}<label class="r"><input type="range" min="-5" max="5" step="0.1" bind:value={x[i]} oninput={touch} aria-label="Number {i + 1}" /><b>{f(scaled[i])}</b></label>{/each}
      <label class="r vol">Volume ×<input type="range" min="0.1" max="20" step="0.1" bind:value={k} oninput={touch} aria-label="Volume" /><b>{k.toFixed(1)}</b></label>
    </div>
    <div>
      <div class="h">After RMSNorm</div>
      <div class="bars">{#each out as o}<div class="bc"><div class="bar" class:neg={o < 0} style="height:{Math.min(100, Math.abs(o) / 2.4 * 100)}%"></div><span>{f(o)}</span></div>{/each}</div>
      <div class="stat"><div><span class="l">typical size before</span><strong>{f(rms(scaled))}</strong></div><div><span class="l">typical size after</span><strong>{f(rms(out))}</strong></div></div>
    </div>
  </div>
  <div class="steps mono">① square each number, take the average → {f(rms(scaled) ** 2)} · ② 1 ÷ √(that) → {f(1 / Math.max(rms(scaled), 1e-5))} · ③ multiply every number by that</div>
  <div class="note">Turn the volume knob: the "before" list gets louder or quieter, but the "after" list never changes. RMSNorm throws away <em>how loud</em> and keeps <em>what shape</em>. (<strong>RMS</strong> = root mean square: square, average, square-root.)</div>
</div>

<style>
  .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 1.4rem; } @media (max-width: 760px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .h { font-weight: 600; font-size: 0.82rem; margin-bottom: 0.4rem; } .r { display: grid; grid-template-columns: 1fr 3rem; gap: 0.6rem; align-items: center; margin-bottom: 2px; font-size: 0.85rem; } .r input { accent-color: var(--accent); } .r b { text-align: right; font-variant-numeric: tabular-nums; } .r.vol { grid-template-columns: auto 1fr 3rem; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid var(--line); color: var(--ink-2); }
  .bars { display: flex; gap: 0.5rem; align-items: flex-end; height: 110px; background: var(--surface-2); border-radius: 10px; padding: 0.5rem; } .bc { flex: 1; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; gap: 2px; font-size: 0.72rem; color: var(--ink-3); }
  .bar { width: 100%; max-width: 28px; background: var(--series-1); border-radius: 4px 4px 0 0; min-height: 2px; } .bar.neg { background: var(--series-2); }
  .stat { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.6rem; } .stat div { background: var(--surface-2); border-radius: 10px; padding: 0.4rem 0.7rem; } .stat strong { display: block; font-size: 1.2rem; } .l { color: var(--ink-3); font-size: 0.74rem; }
  .steps { margin-top: 0.8rem; font-size: 0.78rem; background: var(--code-bg); padding: 0.5rem 0.8rem; border-radius: 8px; overflow-x: auto; white-space: nowrap; } .note { margin-top: 0.6rem; color: var(--ink-2); font-size: 0.85rem; }
</style>
