<script>
  import { getContext, onMount } from 'svelte';
  import { mdl, M, loadModel } from '../lib/model.svelte.js';
  import { data } from '../lib/data.svelte.js';
  import { modelLoss } from '../lib/gpt.js';
  import { ablationSample, lossBigram } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // Switch the two halves of the transformer block off in the real model: attention (all heads) and the MLP.
  const beat = getContext('beat');
  let att = $state(true), mlp = $state(true), idx = $state(4), flips = 0;
  onMount(() => { loadModel().catch(() => {}); });
  const names = $derived(data.status === 'ready' ? ablationSample(data.docs) : []);
  const ready = $derived(mdl.status === 'ready' && names.length > 0);
  const model = $derived(ready ? M.models[M.steps[idx]] : null);
  const L = (a, m) => modelLoss(model, names, { ablate: a ? new Set() : new Set([0, 1, 2, 3]), noMlp: !m });
  const table = $derived(ready ? lossBigram(names, data.vocab, data.docs) : 0);
  const base = $derived(model ? L(true, true) : 0);
  const cur = $derived(model ? L(att, mlp) : 0);
  const rows = $derived(model ? [
    { name: 'attention + MLP (the full model)', v: base },
    { name: 'MLP only (no attention)', v: L(false, true) },
    { name: 'attention only (no MLP)', v: L(true, false) },
    { name: 'neither: letter and position straight to the output', v: L(false, false) },
    { name: 'for reference: the counting table (Chapter 2)', v: table, ref: true },
  ] : []);
  const lo = 2.2, hi = 3.4;
  function flip(which) { if (which === 'a') att = !att; else mlp = !mlp; if (++flips >= 3) beat?.complete(); }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">Switch the two halves off · attention and the MLP</div>
    {#if !ready}<div class="muted">Loading the trained model…</div>{:else}
      <div class="ctl">
        <button class="tg" class:off={!att} onclick={() => flip('a')} aria-pressed={att}>attention {att ? 'ON' : 'OFF'}</button>
        <button class="tg" class:off={!mlp} onclick={() => flip('m')} aria-pressed={mlp}>MLP {mlp ? 'ON' : 'OFF'}</button>
        <label>Training step <select bind:value={idx} aria-label="Training step">{#each M.steps as s, i}<option value={i}>{s}</option>{/each}</select></label>
      </div>
      <div class="big"><span>loss with this setting</span><strong>{cur.toFixed(3)}</strong><em>({cur - base >= 0 ? '+' : ''}{(cur - base).toFixed(3)} vs the full model)</em></div>
      <div class="h">All four combinations</div>
      {#each rows as r}
        <div class="row"><span class="nm">{r.name}</span><span class="bar"><span class="fill" class:ref={r.ref} style="width:{Math.max(2, ((r.v - lo) / (hi - lo)) * 100)}%"></span></span><span class="v">{r.v.toFixed(3)}</span></div>
      {/each}
      <div class="cap">600 names, the real model at the chosen moment of training. Lower is better. Pure chance is 3.31. (This sample is a little easier than average: the counting table scores 2.38 here, against 2.45 on all names.)</div>
    {/if}
  </div>
</DataGate>

<style>
  .ctl { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.8rem; } .ctl label { color: var(--ink-2); font-size: 0.85rem; }
  .tg { padding: 0.4rem 0.9rem; border-radius: 999px; border: 1px solid var(--accent); background: var(--accent); color: var(--on-accent); font-size: 0.82rem; } .tg.off { background: var(--surface); color: var(--ink-3); border-color: var(--line-strong); text-decoration: line-through; }
  select { padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); margin-left: 0.3rem; }
  .big { background: var(--surface-2); border-radius: 12px; padding: 0.6rem 1rem; margin-bottom: 0.9rem; display: flex; gap: 0.8rem; align-items: baseline; flex-wrap: wrap; } .big span { color: var(--ink-3); font-size: 0.78rem; } .big strong { font-size: 1.6rem; font-variant-numeric: tabular-nums; } .big em { color: var(--ink-2); font-style: normal; font-size: 0.85rem; }
  .h { font-weight: 600; margin-bottom: 0.4rem; font-size: 0.85rem; } .row { display: grid; grid-template-columns: 17rem 1fr 3.4rem; gap: 0.6rem; align-items: center; margin-bottom: 4px; } @media (max-width: 640px) { .row { grid-template-columns: 1fr 3.4rem; } .row .bar { grid-column: 1 / -1; } }
  .nm { color: var(--ink-2); font-size: 0.82rem; } .bar { height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; } .fill { display: block; height: 100%; background: var(--series-1); border-radius: 0 4px 4px 0; } .fill.ref { background: var(--series-3); }
  .v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .cap { margin-top: 0.7rem; color: var(--ink-3); font-size: 0.78rem; }
</style>
