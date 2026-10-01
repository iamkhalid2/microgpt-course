<script>
  import { getContext, onMount } from 'svelte';
  import { mdl, M, loadModel } from '../lib/model.svelte.js';
  import { forward } from '../lib/transformer.js';

  // One letter's whole journey through the real model: every list of 16 numbers along the way, as a strip of coloured cells.
  const beat = getContext('beat');
  let name = $state('ka');
  let pos = $state(2);
  let idx = $state(4);
  let acts = new Set();
  onMount(() => { loadModel().catch(() => {}); });

  const clean = $derived(name.toLowerCase().replace(/[^a-z]/g, '').slice(0, 7));
  const ready = $derived(mdl.status === 'ready');
  const model = $derived(ready ? M.models[M.steps[idx]] : null);
  const run = $derived.by(() => (model ? forward(model, [model.BOS, ...[...clean].map((c) => model.uchars.indexOf(c))]) : null));
  const toks = $derived(['⏎', ...clean]);
  const p = $derived(run ? Math.min(pos, run.length - 1) : 0);
  const rec = $derived(run ? run[p] : null);
  const L = $derived(rec ? rec.layers[0] : null);
  const stages = $derived(rec ? [
    { v: rec.tokEmb, t: 'the letter\'s coordinates', n: 'wte row · line 109' },
    { v: rec.posEmb, t: 'the position\'s coordinates', n: 'wpe row · line 110' },
    { v: rec.x0, t: 'added together', n: 'line 111' },
    { v: rec.x1, t: 'after RMSNorm', n: 'line 112' },
    { v: L.attnOut, t: 'what attention adds', n: 'lines 118–133', add: true },
    { v: L.xAfterAttn, t: 'the lane after attention', n: 'line 134' },
    { v: L.mlpOut, t: 'what the MLP adds', n: 'lines 137–140', add: true },
    { v: L.xOut, t: 'the lane after the MLP', n: 'line 141' },
  ] : []);
  const top = $derived(rec ? rec.probs.map((v, i) => ({ v, ch: i < model.uchars.length ? model.uchars[i] : '⏎' })).sort((a, b) => b.v - a.v).slice(0, 7) : []);
  const color = (v) => (v >= 0 ? `color-mix(in oklab, var(--series-2) ${Math.round(Math.min(1, v / 2.2) * 100)}%, var(--surface))` : `color-mix(in oklab, var(--series-1) ${Math.round(Math.min(1, -v / 2.2) * 100)}%, var(--surface))`);
  function note(k) { acts.add(k); if (acts.size >= 2) beat?.complete(); }
</script>

<div class="widget wide">
  <div class="widget-title">X-ray · one letter's journey through the real model</div>
  {#if !ready}<div class="muted">Loading the trained model…</div>{:else}
    <div class="ctl">
      <label>Name <input bind:value={name} oninput={() => { pos = clean.length; note('name'); }} maxlength="7" spellcheck="false" aria-label="Name to read" /></label>
      <span class="lab">Reading:</span>
      {#each toks as t, i}<button class="pb" class:on={i === p} onclick={() => { pos = i; note('pos'); }}>{t}</button>{/each}
      <label>Training step <select bind:value={idx} onchange={() => note('step')} aria-label="Training step">{#each M.steps as s, i}<option value={i}>{s}</option>{/each}</select></label>
    </div>
    <div class="stages">
      {#each stages as s}
        <div class="st" class:add={s.add}>
          <div class="lab2"><strong>{s.add ? '+ ' : ''}{s.t}</strong><small>{s.n}</small></div>
          <div class="strip">{#each s.v as v}<span class="c" style="background:{color(v)}" title={v.toFixed(3)}></span>{/each}</div>
        </div>
      {/each}
    </div>
    <div class="legend"><span class="k b"></span> negative <span class="k o"></span> positive (each strip is 16 numbers; hover for values)</div>
    <div class="out">
      <div class="h">Finally: 16 numbers → 27 scores (<code>lm_head</code>, line 143) → softmax → probabilities for the next symbol after <strong class="mono">{toks[p]}</strong>:</div>
      {#each top as t}<div class="r"><span class="mono ch">{t.ch}</span><span class="bar"><span class="fill" style="width:{(t.v / top[0].v) * 100}%"></span></span><span class="v">{(t.v * 100).toFixed(0)}%</span></div>{/each}
    </div>
  {/if}
</div>

<style>
  .ctl { display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.9rem; } .lab { color: var(--ink-3); font-size: 0.82rem; } label { color: var(--ink-2); display: flex; gap: 0.5rem; align-items: center; font-size: 0.85rem; }
  input { font-family: var(--font-mono); padding: 0.3rem 0.6rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); width: 7rem; } select { padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); }
  .pb { width: 2rem; height: 2rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-family: var(--font-mono); } .pb.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .stages { display: grid; gap: 0.35rem; } .st { display: grid; grid-template-columns: 15rem 1fr; gap: 0.8rem; align-items: center; padding: 0.3rem 0.5rem; border-radius: 9px; background: var(--surface-2); } .st.add { background: var(--accent-wash); }
  @media (max-width: 700px) { .st { grid-template-columns: 1fr; gap: 0.2rem; } }
  .lab2 strong { display: block; font-size: 0.85rem; } .lab2 small { color: var(--ink-3); font-size: 0.72rem; font-family: var(--font-mono); }
  .strip { display: grid; grid-template-columns: repeat(16, 1fr); gap: 2px; } .c { height: 22px; border-radius: 4px; border: 1px solid var(--line); }
  .legend { margin: 0.5rem 0 0.9rem; color: var(--ink-3); font-size: 0.76rem; display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; } .k { display: inline-block; width: 12px; height: 12px; border-radius: 3px; } .k.b { background: var(--series-1); } .k.o { background: var(--series-2); }
  .out .h { font-size: 0.85rem; color: var(--ink-2); margin-bottom: 0.4rem; } .r { display: grid; grid-template-columns: 1.6rem 1fr 3rem; gap: 0.5rem; align-items: center; margin-bottom: 3px; } .ch { text-align: center; } .bar { height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; } .fill { display: block; height: 100%; background: var(--series-1); border-radius: 0 4px 4px 0; } .v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; font-size: 0.85rem; }
</style>
