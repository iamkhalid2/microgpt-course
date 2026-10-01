<script>
  import { getContext, onMount } from 'svelte';
  import { mdl, M, loadModel } from '../lib/model.svelte.js';
  import { forward } from '../lib/transformer.js';
  import { rms } from '../lib/deep.js';

  // The residual stream of the real model: how big is what the blocks ADD, compared with what is already there?
  const beat = getContext('beat');
  let idx = $state(0);
  let touched = new Set([0]);
  onMount(() => { loadModel().catch(() => {}); });
  const ready = $derived(mdl.status === 'ready');
  const names = ['mariah', 'jonathan', 'emma', 'kayla', 'oliver', 'sophia', 'noah', 'isabella'];
  const stat = $derived.by(() => {
    if (!ready) return null;
    const m = M.models[M.steps[idx]];
    let a = 0, b = 0, c = 0, e = 0, n = 0;
    for (const nm of names) {
      for (const s of forward(m, [m.BOS, ...[...nm].map((ch) => m.uchars.indexOf(ch))].slice(0, 8))) { const L = s.layers[0]; a += rms(s.x1); b += rms(L.attnOut); c += rms(L.mlpOut); e += rms(L.xOut); n++; }
    }
    return { start: a / n, att: b / n, mlp: c / n, end: e / n };
  });
  function pick(i) { idx = i; touched.add(i); if (touched.size >= 3) beat?.complete(); }
  const sc = (v) => Math.min(100, (v / 1.3) * 100);
</script>

<div class="widget wide">
  <div class="widget-title">The real model's residual stream · what do the blocks add?</div>
  {#if !ready}<div class="muted">Loading the trained model…</div>{:else}
    <label class="sl" for="st">Training step: <strong>{M.steps[idx]}</strong></label>
    <input id="st" type="range" min="0" max={M.steps.length - 1} step="1" value={idx} oninput={(e) => pick(+e.currentTarget.value)} />
    <div class="ticks">{#each M.steps as s}<span>{s}</span>{/each}</div>
    <div class="rows">
      <div class="r"><span class="nm">The stream going in</span><span class="bar"><span class="f a" style="width:{sc(stat.start)}%"></span></span><span class="v">{stat.start.toFixed(2)}</span></div>
      <div class="r"><span class="nm">+ what attention adds</span><span class="bar"><span class="f b" style="width:{sc(stat.att)}%"></span></span><span class="v">{stat.att.toFixed(3)}</span></div>
      <div class="r"><span class="nm">+ what the MLP adds</span><span class="bar"><span class="f c" style="width:{sc(stat.mlp)}%"></span></span><span class="v">{stat.mlp.toFixed(3)}</span></div>
      <div class="r end"><span class="nm">= the stream going out</span><span class="bar"><span class="f a" style="width:{sc(stat.end)}%"></span></span><span class="v">{stat.end.toFixed(2)}</span></div>
    </div>
    <div class="cap">Typical size (RMS) of each list, averaged over 8 names. {idx === 0 ? 'At step 0 both blocks add exactly zero, because their output matrices start at zero (lines 86 and 88 of microgpt.py). The model begins by doing nothing but pass the letter through.' : 'The blocks learn to add edits to the stream. Attention\'s edit grows the most.'}</div>
  {/if}
</div>

<style>
  .sl { display: block; } input[type='range'] { width: 100%; accent-color: var(--accent); } .ticks { display: flex; justify-content: space-between; color: var(--ink-3); font-size: 0.72rem; margin-bottom: 0.8rem; }
  .r { display: grid; grid-template-columns: 12rem 1fr 3.6rem; gap: 0.6rem; align-items: center; margin-bottom: 5px; } @media (max-width: 640px) { .r { grid-template-columns: 1fr 3.6rem; } .r .bar { grid-column: 1 / -1; } }
  .r.end { margin-top: 0.4rem; padding-top: 0.4rem; border-top: 1px solid var(--line); } .nm { color: var(--ink-2); font-size: 0.85rem; }
  .bar { height: 16px; background: var(--surface-2); border-radius: 4px; overflow: hidden; } .f { display: block; height: 100%; border-radius: 0 4px 4px 0; transition: width 0.4s; } .f.a { background: var(--series-1); } .f.b { background: var(--series-2); } .f.c { background: var(--series-3); }
  .v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; } .cap { margin-top: 0.7rem; color: var(--ink-2); font-size: 0.85rem; }
</style>
