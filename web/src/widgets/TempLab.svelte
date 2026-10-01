<script>
  import { getContext, onMount } from 'svelte';
  import { mdl, M, loadModel } from '../lib/model.svelte.js';
  import { data } from '../lib/data.svelte.js';
  import { nextProbs, generate } from '../lib/transformer.js';
  import { makeRng } from '../lib/rng.js';

  // Temperature, with the REAL model. Top: how the next-letter probabilities reshape. Bottom: what 300 generated names look like.
  const beat = getContext('beat');
  let prefix = $state('ka');
  let T = $state(0.5);          // live, drives the bars
  let Tgen = $state(0.5);       // set when you let go of the slider, drives the (slower) name generation
  let idx = $state(4);
  let busy = $state(false);
  let names = $state([]);
  let stats = $state(null);
  let acts = new Set();
  onMount(() => { loadModel().catch(() => {}); });

  const ready = $derived(mdl.status === 'ready' && data.status === 'ready');
  const model = $derived(ready ? M.models[M.steps[idx]] : null);
  const clean = $derived(prefix.toLowerCase().replace(/[^a-z]/g, '').slice(0, 6));
  const dist = $derived.by(() => {
    if (!model) return null;
    const cur = nextProbs(model, clean, T), base = nextProbs(model, clean, 1);
    const lab = (i) => (i < model.uchars.length ? model.uchars[i] : '⏎');
    return cur.probs.map((p, i) => ({ ch: lab(i), p, b: base.probs[i] })).sort((a, b) => b.p - a.p).slice(0, 9);
  });

  function generateBatch() {
    if (!model) return;
    busy = true;
    setTimeout(() => {
      const set = new Set(data.docs), rng = makeRng(7);
      const out = Array.from({ length: 300 }, () => generate(model, rng, Tgen));
      stats = { real: (out.filter((n) => set.has(n)).length / out.length) * 100, uniq: (new Set(out).size / out.length) * 100, len: out.reduce((s, n) => s + n.length, 0) / out.length };
      names = out.slice(0, 14);
      busy = false;
    }, 20);
  }
  $effect(() => { Tgen; idx; if (ready) generateBatch(); });
  function release() { Tgen = T; acts.add('T' + T.toFixed(1)); if (acts.size >= 3) beat?.complete(); }
</script>

<div class="widget wide">
  <div class="widget-title">Temperature · the dial between safe and adventurous</div>
  {#if !ready}<div class="muted">Loading the trained model…</div>{:else}
    <div class="ctl">
      <label class="t">Temperature <input type="range" min="0.1" max="2" step="0.05" bind:value={T} oninput={() => (T = +T)} onchange={release} aria-label="Temperature" /> <strong>{T.toFixed(2)}</strong></label>
      <label>The name so far <input bind:value={prefix} maxlength="6" spellcheck="false" aria-label="Name so far" /></label>
      <label>Model after step <select bind:value={idx} aria-label="Training step" onchange={() => acts.add('step')}>{#each M.steps as s, i}<option value={i}>{s}</option>{/each}</select></label>
    </div>

    <div class="h">The model's probabilities for the next symbol after “{clean || '⏎'}”<span class="muted"> (pale bar: temperature 1)</span></div>
    {#each dist as d}
      <div class="r"><span class="mono ch">{d.ch}</span><span class="bar"><span class="pale" style="width:{d.b * 100}%"></span><span class="fill" style="width:{d.p * 100}%"></span></span><span class="v">{(d.p * 100).toFixed(0)}%</span></div>
    {/each}

    <div class="h gen">Names it writes at this temperature (from the very start, {busy ? 'generating…' : '300 samples'})</div>
    <div class="names">{#each names as n}<span class="mono n">{n || '·'}</span>{/each}</div>
    {#if stats}
      <div class="stats">
        <div><span class="l">are real names from the dataset</span><strong>{stats.real.toFixed(0)}%</strong></div>
        <div><span class="l">are different from each other</span><strong>{stats.uniq.toFixed(0)}%</strong></div>
        <div><span class="l">average length</span><strong>{stats.len.toFixed(1)}</strong></div>
      </div>
    {/if}
    <div class="cap">Let go of the slider to regenerate. Low temperature: the same few safe names, over and over. High temperature: everything different, and mostly nonsense. Try the name <span class="mono">xqz</span>: even for nonsense, the model <em>always</em> produces a continuation.</div>
  {/if}
</div>

<style>
  .ctl { display: flex; gap: 0.8rem 1.4rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.9rem; } label { color: var(--ink-2); display: flex; gap: 0.5rem; align-items: center; font-size: 0.85rem; } .t { max-width: 100%; } .t input { accent-color: var(--accent); width: 190px; min-width: 0; flex: 1; max-width: 190px; }
  input[aria-label='Name so far'] { font-family: var(--font-mono); padding: 0.3rem 0.6rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); width: 7rem; } select { padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); }
  .h { font-weight: 600; font-size: 0.85rem; margin-bottom: 0.4rem; } .h.gen { margin-top: 1rem; } .r { display: grid; grid-template-columns: 1.6rem 1fr 3rem; gap: 0.5rem; align-items: center; margin-bottom: 3px; } .ch { text-align: center; }
  .bar { position: relative; height: 16px; background: var(--surface-2); border-radius: 4px; overflow: hidden; } .pale { position: absolute; left: 0; top: 0; bottom: 0; background: var(--ink-3); opacity: 0.28; } .fill { position: absolute; left: 0; top: 4px; bottom: 4px; background: var(--series-1); border-radius: 0 3px 3px 0; transition: width 0.15s; } .v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; font-size: 0.85rem; }
  .names { display: flex; flex-wrap: wrap; gap: 0.35rem; } .n { background: var(--code-bg); padding: 0.25rem 0.6rem; border-radius: 8px; font-size: 0.95rem; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-top: 0.8rem; } .stats div { background: var(--surface-2); border-radius: 10px; padding: 0.45rem 0.7rem; } .stats strong { display: block; font-size: 1.3rem; font-variant-numeric: tabular-nums; } .l { color: var(--ink-3); font-size: 0.74rem; }
  .cap { margin-top: 0.8rem; color: var(--ink-3); font-size: 0.82rem; }
</style>
