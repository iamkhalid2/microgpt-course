<script>
  import { getContext, onMount } from 'svelte';
  import { mdl, M, loadModel } from '../lib/model.svelte.js';
  import { data } from '../lib/data.svelte.js';
  import { modelLoss } from '../lib/transformer.js';
  import { ablationSample, lossBigram } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // Switch attention heads off in the real model and measure the damage (average loss on 600 fixed names).
  const beat = getContext('beat');
  let on = $state([true, true, true, true]);
  let idx = $state(4);
  let flips = 0;
  onMount(() => { loadModel().catch(() => {}); });

  const names = $derived(data.status === 'ready' ? ablationSample(data.docs) : []);
  const ready = $derived(mdl.status === 'ready' && names.length > 0);
  const model = $derived(ready ? M.models[M.steps[idx]] : null);
  const ablate = $derived(new Set(on.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0)));
  const table = $derived(ready ? lossBigram(names, data.vocab, data.docs) : 0);
  const base = $derived(model ? modelLoss(model, names) : 0);
  const cur = $derived(model ? modelLoss(model, names, { ablate }) : 0);
  const delta = $derived(cur - base);
  const per = $derived(model ? [0, 1, 2, 3].map((h) => modelLoss(model, names, { ablate: new Set([h]) }) - base) : []);
  const all = $derived(model ? modelLoss(model, names, { ablate: new Set([0, 1, 2, 3]) }) - base : 0);
  const mx = $derived(Math.max(0.03, ...per.map(Math.abs), Math.abs(all)));
  function flip(i) { on[i] = !on[i]; if (++flips >= 3) beat?.complete(); }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">Switch heads off · what does each one contribute?</div>
    {#if !ready}
      <div class="muted">Loading the trained model…</div>
    {:else}
      <div class="ctl">
        <span class="lab">Heads:</span>
        {#each on as v, i}<button class="tg" class:off={!v} onclick={() => flip(i)} aria-pressed={v}>head {i} {v ? 'ON' : 'OFF'}</button>{/each}
        <label class="lab">Training step <select bind:value={idx} aria-label="Training step">{#each M.steps as s, i}<option value={i}>{s}</option>{/each}</select></label>
      </div>
      <div class="big">
        <div><span class="l">loss with this setting</span><strong>{cur.toFixed(3)}</strong></div>
        <div><span class="l">all heads on</span><strong>{base.toFixed(3)}</strong></div>
        <div class:bad={delta > 0.005}><span class="l">difference</span><strong>{delta >= 0 ? '+' : ''}{delta.toFixed(3)}</strong></div>
      </div>
      <div class="h">Damage from switching off each head on its own (higher = it mattered more)</div>
      {#each per as d, h}
        <div class="row"><span class="nm">head {h} alone</span><span class="bar"><span class="fill" style="width:{Math.max(0, d) / mx * 100}%"></span></span><span class="v">{d >= 0 ? '+' : ''}{d.toFixed(3)}</span></div>
      {/each}
      <div class="row all"><span class="nm">all four off</span><span class="bar"><span class="fill b" style="width:{Math.max(0, all) / mx * 100}%"></span></span><span class="v">{all >= 0 ? '+' : ''}{all.toFixed(3)}</span></div>
      <div class="cap">Measured on 600 names, using the real model at the chosen moment of training. 3.31 would be pure chance. Lower is better. For reference, the counting table from Chapter 2 scores {table.toFixed(2)} on these same names (they are a little easier than average).</div>
    {/if}
  </div>
</DataGate>

<style>
  .ctl { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.8rem; } .lab { color: var(--ink-2); font-size: 0.85rem; }
  .tg { padding: 0.35rem 0.8rem; border-radius: 999px; border: 1px solid var(--accent); background: var(--accent); color: var(--on-accent); font-size: 0.8rem; }
  .tg.off { background: var(--surface); color: var(--ink-3); border-color: var(--line-strong); text-decoration: line-through; }
  select { padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); margin-left: 0.3rem; }
  .big { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 0.9rem; } .big div { background: var(--surface-2); border-radius: 10px; padding: 0.5rem 0.8rem; } .big div.bad { background: var(--pain-wash); }
  .big strong { display: block; font-size: 1.4rem; font-variant-numeric: tabular-nums; } .l { color: var(--ink-3); font-size: 0.74rem; }
  .h { font-weight: 600; margin: 0.3rem 0 0.4rem; font-size: 0.85rem; }
  .row { display: grid; grid-template-columns: 7rem 1fr 4rem; gap: 0.6rem; align-items: center; margin-bottom: 4px; } .row.all { margin-top: 0.4rem; padding-top: 0.4rem; border-top: 1px solid var(--line); }
  .nm { color: var(--ink-2); font-size: 0.85rem; } .bar { height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; } .fill { display: block; height: 100%; background: var(--series-1); border-radius: 0 4px 4px 0; } .fill.b { background: var(--series-2); }
  .v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; font-size: 0.85rem; } .cap { margin-top: 0.7rem; color: var(--ink-3); font-size: 0.78rem; }
</style>
