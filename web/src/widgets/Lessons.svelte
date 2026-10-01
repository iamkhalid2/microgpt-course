<script>
  import { getContext, onMount } from 'svelte';
  import { mdl, M, loadModel } from '../lib/model.svelte.js';
  import { forward } from '../lib/transformer.js';

  // One name is not one lesson but one lesson PER POSITION: at each step the model is shown the true letters so far
  // and asked for the next one. The loss for the name is the average of these.
  const beat = getContext('beat');
  let name = $state('emma');
  let idx = $state(4);
  let acts = new Set();
  onMount(() => { loadModel().catch(() => {}); });
  const ready = $derived(mdl.status === 'ready');
  const clean = $derived(name.toLowerCase().replace(/[^a-z]/g, '').slice(0, 12));
  const model = $derived(ready ? M.models[M.steps[idx]] : null);
  const rows = $derived.by(() => {
    if (!model || !clean) return [];
    const ids = [model.BOS, ...[...clean].map((c) => model.uchars.indexOf(c)), model.BOS];
    const used = Math.min(model.block_size, ids.length - 1);
    const steps = forward(model, ids.slice(0, used));
    const lab = (i) => (i === model.BOS ? '⏎' : model.uchars[i]);
    return Array.from({ length: ids.length - 1 }, (_, i) => {
      if (i >= used) return { i, from: lab(ids[i]), to: lab(ids[i + 1]), skipped: true };
      const p = steps[i].probs[ids[i + 1]];
      return { i, from: lab(ids[i]), to: lab(ids[i + 1]), p, loss: -Math.log(p) };
    });
  });
  const used = $derived(rows.filter((r) => !r.skipped));
  const avg = $derived(used.length ? used.reduce((s, r) => s + r.loss, 0) / used.length : 0);
  function note(k) { acts.add(k); if (acts.size >= 2) beat?.complete(); }
</script>

<div class="widget wide">
  <div class="widget-title">One name, many lessons · the real model</div>
  {#if !ready}<div class="muted">Loading the trained model…</div>{:else}
    <div class="ctl">
      <label>Name <input bind:value={name} oninput={() => note('name')} maxlength="12" spellcheck="false" aria-label="Name to read" /></label>
      <label>Model after training step <select bind:value={idx} onchange={() => note('step')} aria-label="Training step">{#each M.steps as s, i}<option value={i}>{s}</option>{/each}</select></label>
    </div>
    <div class="tbl">
      <div class="r head"><span>#</span><span>the model is shown</span><span>and must predict</span><span>probability it gave</span><span>surprise (loss)</span></div>
      {#each rows as r}
        <div class="r" class:skip={r.skipped}>
          <span>{r.i + 1}</span><span class="mono big">{r.from}</span><span class="mono big">{r.to}</span>
          {#if r.skipped}<span class="muted" style="grid-column: span 2">not used: the model only has {model.block_size} positions (block_size)</span>
          {:else}<span>{(r.p * 100).toFixed(1)}%</span><span class="bar"><span class="fill" style="width:{Math.min(100, (r.loss / 6) * 100)}%"></span><em>{r.loss.toFixed(2)}</em></span>{/if}
        </div>
      {/each}
      <div class="r tot"><span></span><span class="lab" style="grid-column: span 2">average of the lessons = this name's loss</span><span></span><strong>{avg.toFixed(3)}</strong></div>
    </div>
    <div class="cap">The "shown" letters are always the <em>true</em> ones from the name, not the model's own guesses (this is called <strong>teacher forcing</strong>). Pure chance would score 3.296 on every row. {name.length > 7 ? 'Names longer than 7 letters get cut off at 8 lessons.' : ''}</div>
  {/if}
</div>

<style>
  .ctl { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.8rem; } label { color: var(--ink-2); display: flex; gap: 0.5rem; align-items: center; font-size: 0.85rem; }
  input { font-family: var(--font-mono); padding: 0.3rem 0.6rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); width: 9rem; } select { padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); }
  .tbl { display: grid; gap: 3px; } .r { display: grid; grid-template-columns: 2rem 7rem 7rem 8rem 1fr; gap: 0.6rem; align-items: center; padding: 0.3rem 0.6rem; border-radius: 8px; background: var(--surface-2); font-size: 0.85rem; }
  @media (max-width: 700px) { .r { grid-template-columns: 1.3rem 3.4rem 3.6rem 5.2rem minmax(0, 1fr); gap: 0.3rem; font-size: 0.78rem; } .r.head { font-size: 0.6rem; letter-spacing: 0; align-items: end; } }
  .r.head { background: none; color: var(--ink-3); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; } .r.skip { opacity: 0.5; } .big { font-size: 1.05rem; }
  .bar { position: relative; height: 18px; background: var(--surface); border-radius: 4px; overflow: hidden; } .fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--series-2); border-radius: 0 4px 4px 0; } .bar em { position: relative; font-style: normal; font-size: 0.74rem; padding-left: 0.4rem; line-height: 18px; }
  .r.tot { background: var(--accent-wash); margin-top: 0.3rem; } .tot strong { font-size: 1.15rem; font-variant-numeric: tabular-nums; }
  .cap { margin-top: 0.7rem; color: var(--ink-3); font-size: 0.82rem; }
</style>
