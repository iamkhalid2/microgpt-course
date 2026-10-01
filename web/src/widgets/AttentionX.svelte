<script>
  import { getContext, onMount } from 'svelte';
  import { mdl, M, loadModel } from '../lib/model.svelte.js';
  import { forward } from '../lib/gpt.js';
  import AttMatrix from './AttMatrix.svelte';

  // The REAL model's attention weights for any name you type. mode "single" shows one head; "all" shows all four.
  let { mode = 'single', initial = 'annabel', gateOn = 'name' } = $props();
  const beat = getContext('beat');
  let name = $state(initial);
  let head = $state(0);
  let idx = $state(4);                 // checkpoint index (default: fully trained)
  let sel = $state(null);
  let acts = new Set();
  onMount(() => { loadModel().catch(() => {}); });

  const clean = $derived(name.toLowerCase().replace(/[^a-z]/g, '').slice(0, 7));
  const ready = $derived(mdl.status === 'ready');
  const model = $derived(ready ? M.models[M.steps[idx]] : null);
  const run = $derived.by(() => {
    if (!model) return null;
    const ids = [model.BOS, ...[...clean].map((c) => model.uchars.indexOf(c))];
    return forward(model, ids);
  });
  const tokens = $derived(['⏎', ...clean]);
  const weightsOf = (h) => (run ? run.map((s) => s.layers[0].heads[h].weights) : []);
  const row = $derived(sel === null ? run.length - 1 : sel);
  const nextTop = $derived.by(() => {
    if (!run) return [];
    const p = run[Math.min(row, run.length - 1)].probs;
    return p.map((v, i) => ({ v, ch: i < model.uchars.length ? model.uchars[i] : '⏎' })).sort((a, b) => b.v - a.v).slice(0, 5);
  });
  function note(k) { acts.add(k); if (acts.size >= 3) beat?.complete(); }
</script>

<div class="widget wide">
  <div class="widget-title">The real model's attention · type a name</div>
  {#if !ready}
    <div class="muted">Loading the trained model…</div>
  {:else}
    <div class="ctl">
      <label>Name <input bind:value={name} oninput={() => { sel = null; note('name'); }} maxlength="7" spellcheck="false" aria-label="Name to read" /></label>
      <label>Training step
        <select bind:value={idx} onchange={() => note('step')} aria-label="Training step">{#each M.steps as s, i}<option value={i}>{s}</option>{/each}</select>
      </label>
      {#if mode === 'single'}
        <label>Head <select bind:value={head} onchange={() => note('head')} aria-label="Attention head">{#each [0, 1, 2, 3] as h}<option value={h}>head {h}</option>{/each}</select></label>
      {/if}
    </div>
    {#if clean.length === 0}
      <div class="muted">Type some letters (a–z).</div>
    {:else if mode === 'single'}
      <div class="one"><AttMatrix {tokens} weights={weightsOf(head)} {sel} onsel={(r) => { sel = r; note('row'); }} title="Head {head}: when reading the row letter, how much attention goes to each earlier position (%)" /></div>
    {:else}
      <div class="four">{#each [0, 1, 2, 3] as h}<AttMatrix {tokens} weights={weightsOf(h)} {sel} onsel={(r) => { sel = r; note('row'); }} title="Head {h}" cell={34} />{/each}</div>
    {/if}
    <div class="pred">
      After reading <strong class="mono">{tokens[Math.min(row, tokens.length - 1)]}</strong> (row {Math.min(row, tokens.length - 1) + 1}), the model's next-symbol guesses:
      {#each nextTop as t}<span class="chip"><b class="mono">{t.ch}</b> {(t.v * 100).toFixed(0)}%</span>{/each}
    </div>
    <div class="cap">Click a row to choose which letter's prediction to show. Each row's percentages add to 100%. Step 0 is the untrained model: every position gets equal attention.</div>
  {/if}
</div>

<style>
  .ctl { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.8rem; } .ctl label { display: flex; gap: 0.5rem; align-items: center; color: var(--ink-2); }
  input { font-family: var(--font-mono); padding: 0.3rem 0.6rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); width: 8rem; }
  select { padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); }
  .one { max-width: 420px; margin: 0 auto; } .four { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.8rem 1.4rem; max-width: 640px; margin: 0 auto; }
  .pred { margin-top: 0.8rem; display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
  .chip { background: var(--surface-2); border-radius: 999px; padding: 0.15rem 0.7rem; font-size: 0.85rem; } .chip b { margin-right: 0.3rem; }
  .cap { margin-top: 0.6rem; color: var(--ink-3); font-size: 0.8rem; }
</style>
