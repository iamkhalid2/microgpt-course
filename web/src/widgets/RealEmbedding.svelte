<script>
  import { getContext, onMount } from 'svelte';
  import { mdl, M, loadModel } from '../lib/model.svelte.js';
  import { pca } from '../lib/pca.js';

  // The real model's table of letter coordinates (16 numbers per letter), squashed to 2D, at five moments of training.
  const beat = getContext('beat');
  let idx = $state(0);
  let touched = new Set([0]);
  onMount(() => { loadModel().catch(() => {}); });

  const VOW = 'aeiou';
  const letters = $derived(mdl.status === 'ready' ? M.cfg.uchars : []);
  const axes = $derived.by(() => {
    if (mdl.status !== 'ready') return null;
    const final = M.models[M.steps[M.steps.length - 1]].w.wte.slice(0, letters.length);
    return pca(final, 2);
  });
  const step = $derived(mdl.status === 'ready' ? M.steps[idx] ?? 0 : 0);
  const pts = $derived.by(() => {
    if (!axes) return [];
    const w = M.models[step].w.wte;
    return axes.project(w).map((p, i) => ({ i, x: p[0], y: p[1], ch: i < letters.length ? letters[i] : '⏎', vow: i < letters.length && VOW.includes(letters[i]), bos: i >= letters.length }));
  });
  const dist = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
  const stats = $derived.by(() => {
    if (mdl.status !== 'ready') return null;
    const w = M.models[step].w.wte;
    const vv = [], vc = [];
    for (let i = 0; i < letters.length; i++) for (let j = i + 1; j < letters.length; j++) {
      const a = VOW.includes(letters[i]), b = VOW.includes(letters[j]);
      if (a && b) vv.push(dist(w[i], w[j])); else if (a !== b) vc.push(dist(w[i], w[j]));
    }
    const avg = (v) => v.reduce((s, x) => s + x, 0) / v.length;
    return { vv: avg(vv), vc: avg(vc) };
  });

  const S = 440, PAD = 30;
  const ext = $derived.by(() => {
    if (!axes || mdl.status !== 'ready') return { lx: 2.4, ly: 1.4 };
    const fin = axes.project(M.models[M.steps[M.steps.length - 1]].w.wte);
    return { lx: Math.max(...fin.map((p) => Math.abs(p[0]))) * 1.1, ly: Math.max(...fin.map((p) => Math.abs(p[1]))) * 1.15 };
  });
  const X = (v) => S / 2 + (v / ext.lx) * (S / 2 - PAD);
  const Y = (v) => S / 2 - (v / ext.ly) * (S / 2 - PAD);
  function pick(i) { idx = i; touched.add(i); if (touched.size >= 3 || i === M.steps.length - 1) beat?.complete(); }
</script>

<div class="widget wide">
  <div class="widget-title">The real model's map of letters · drag through its training</div>
  {#if !axes}
    <div class="muted">Loading the trained model…</div>
  {:else}
    <div class="grid">
      <svg viewBox="0 0 {S} {S}" class="map" role="img" aria-label="Two-dimensional map of letter embeddings">
        <line x1={PAD / 2} x2={S - PAD / 2} y1={S / 2} y2={S / 2} stroke="var(--grid)" /><line y1={PAD / 2} y2={S - PAD / 2} x1={S / 2} x2={S / 2} stroke="var(--grid)" />
        {#each pts as p (p.i)}
          <g class="pt" style="transform: translate({X(p.x)}px, {Y(p.y)}px)">
            <circle r="9" style="fill: {p.bos ? 'var(--ink-3)' : p.vow ? 'var(--series-2)' : 'var(--series-1)'}; stroke: var(--surface); stroke-width: 2;" />
            <text class="ch" text-anchor="middle" y="4">{p.ch}</text>
          </g>
        {/each}
      </svg>
      <div class="side">
        <label class="sl" for="stp">Training step: <strong>{step}</strong></label>
        <input id="stp" type="range" min="0" max={M.steps.length - 1} step="1" value={idx} oninput={(e) => pick(+e.currentTarget.value)} />
        <div class="ticks">{#each M.steps as s}<span>{s}</span>{/each}</div>
        <div class="leg"><span class="k o"></span> vowels (a e i o u) <span class="k b"></span> consonants <span class="k g"></span> ⏎ start/end</div>
        {#if stats}
          <div class="stat">
            <div><span class="l">average distance between two vowels</span><strong>{stats.vv.toFixed(2)}</strong></div>
            <div><span class="l">average distance, vowel to consonant</span><strong>{stats.vc.toFixed(2)}</strong></div>
          </div>
        {/if}
        <div class="note">{step === 0 ? 'Step 0: the table is random noise. Every letter sits in nearly the same spot, and no letter is closer to any other.' : step < 1000 ? 'Learning has begun to pull letters apart. Notice the vowels drifting to one side.' : 'The model was never told what a vowel is. It invented that distinction because it helps predict the next letter.'}</div>
      </div>
    </div>
  {/if}
</div>

<style>
  .grid { display: grid; grid-template-columns: minmax(260px, 440px) 1fr; gap: 1.4rem; align-items: start; }
  @media (max-width: 800px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .map { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 12px; }
  .pt { transition: transform 0.9s cubic-bezier(0.3, 0.8, 0.3, 1); }
  .ch { font-family: var(--font-mono); font-size: 11px; font-weight: 700; fill: #fff; pointer-events: none; }
  .sl { display: block; margin-bottom: 0.2rem; }
  input[type='range'] { width: 100%; accent-color: var(--accent); }
  .ticks { display: flex; justify-content: space-between; color: var(--ink-3); font-size: 0.72rem; margin-bottom: 0.7rem; }
  .leg { display: flex; flex-wrap: wrap; gap: 0.2rem 0.8rem; align-items: center; color: var(--ink-2); font-size: 0.8rem; margin-bottom: 0.7rem; }
  .k { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 0.3rem; vertical-align: middle; } .k.o { background: var(--series-2); } .k.b { background: var(--series-1); } .k.g { background: var(--ink-3); }
  .stat { display: grid; gap: 0.4rem; margin-bottom: 0.7rem; }
  .stat div { background: var(--surface-2); border-radius: 10px; padding: 0.4rem 0.7rem; display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; }
  .stat strong { font-size: 1.15rem; font-variant-numeric: tabular-nums; } .l { color: var(--ink-3); font-size: 0.78rem; }
  .note { color: var(--ink-2); font-size: 0.85rem; min-height: 3.4rem; }
</style>
