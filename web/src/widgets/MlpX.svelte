<script>
  import { getContext, onMount } from 'svelte';
  import { mdl, M, loadModel } from '../lib/model.svelte.js';
  import { forward } from '../lib/gpt.js';

  // The 64 "neurons" of the real model's MLP at one position of a name you type. Most are exactly zero.
  const beat = getContext('beat');
  let name = $state('mariah');
  let pos = $state(3);
  let acts = new Set();
  onMount(() => { loadModel().catch(() => {}); });
  const clean = $derived(name.toLowerCase().replace(/[^a-z]/g, '').slice(0, 7));
  const ready = $derived(mdl.status === 'ready');
  const model = $derived(ready ? M.models[M.steps[M.steps.length - 1]] : null);
  const run = $derived.by(() => { if (!model) return null; return forward(model, [model.BOS, ...[...clean].map((c) => model.uchars.indexOf(c))]); });
  const p = $derived(run ? Math.min(pos, run.length - 1) : 0);
  const act = $derived(run ? run[p].layers[0].act : []);
  const hid = $derived(run ? run[p].layers[0].h : []);
  const firing = $derived(act.filter((a) => a > 0).length);
  const mx = $derived(Math.max(1e-9, ...act));
  const toks = $derived(['⏎', ...clean]);
  function note(k) { acts.add(k); if (acts.size >= 2 && acts.has('pos')) beat?.complete(); }
</script>

<div class="widget wide">
  <div class="widget-title">The 64 neurons of the real MLP · which ones fire?</div>
  {#if !ready}<div class="muted">Loading the trained model…</div>{:else}
    <div class="ctl">
      <label>Name <input bind:value={name} oninput={() => note('name')} maxlength="7" spellcheck="false" aria-label="Name to read" /></label>
      <span class="lab">Position being read:</span>
      {#each toks as t, i}<button class="pb" class:on={i === p} onclick={() => { pos = i; note('pos'); }}>{t}</button>{/each}
    </div>
    <div class="neurons" role="img" aria-label="Activation of each of the 64 neurons">
      {#each act as a, i}<div class="n" title="neuron {i}: before the gate {hid[i].toFixed(2)}, after {a.toFixed(2)}"><div class="b" style="height:{a > 0 ? Math.max(6, (a / mx) * 100) : 0}%"></div></div>{/each}
    </div>
    <div class="sum"><strong>{firing}</strong> of 64 neurons are active here. The other <strong>{64 - firing}</strong> are exactly zero: their input was negative, so the ReLU gate blocked them.</div>
  {/if}
</div>

<style>
  .ctl { display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.9rem; } .lab { color: var(--ink-3); font-size: 0.82rem; } label { color: var(--ink-2); display: flex; gap: 0.5rem; align-items: center; }
  input { font-family: var(--font-mono); padding: 0.3rem 0.6rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); width: 8rem; }
  .pb { width: 2rem; height: 2rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-family: var(--font-mono); } .pb.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .neurons { display: grid; grid-template-columns: repeat(32, 1fr); gap: 3px; height: 120px; background: var(--surface-2); border-radius: 10px; padding: 8px; }
  .n { display: flex; align-items: flex-end; border-bottom: 1px solid var(--axis); } .b { width: 100%; background: var(--series-1); border-radius: 3px 3px 0 0; }
  .sum { margin-top: 0.7rem; color: var(--ink-2); }
</style>
