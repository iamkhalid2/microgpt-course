<script>
  import { getContext } from 'svelte';
  import { softmax } from '../lib/stats.js';

  // Raw scores (any numbers) -> exp (all positive) -> shares of the total (probabilities).
  const beat = getContext('beat');
  const letters = ['a', 'e', 'n', 'r', '⏎'];
  let z = $state([2.0, 1.0, 0.5, -1.0, 0.0]);
  let shift = $state(0);
  let huge = $state(false);
  let acts = 0;
  const touch = () => { if (++acts >= 4) beat?.complete(); };

  const scores = $derived(z.map((v) => (huge ? v * 500 : v) + shift));
  const ex = $derived(scores.map((v) => Math.exp(v)));
  const naive = $derived.by(() => { const t = ex.reduce((a, b) => a + b, 0); return ex.map((e) => e / t); });
  const safe = $derived(softmax(scores));
  const base = $derived(softmax(z));
  const fmt = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 1e4 ? v.toExponential(1) : v.toFixed(2)) : String(v));
  const pct = (v) => (Number.isFinite(v) ? (v * 100).toFixed(1) + '%' : 'NaN');
  const maxEx = $derived(Math.max(...ex.filter(Number.isFinite), 1e-9));
</script>

<div class="widget wide">
  <div class="widget-title">Softmax · from any scores to probabilities</div>
  <div class="grid head"><span>letter</span><span>raw score from the dials</span><span>e<sup>score</sup> (always positive)</span><span>share of the total = probability</span></div>
  {#each letters as l, i}
    <div class="grid">
      <span class="mono lt">{l}</span>
      <span class="sl"><input type="range" min="-6" max="6" step="0.1" bind:value={z[i]} oninput={touch} aria-label="Score for {l}" disabled={huge} /><output>{fmt(scores[i])}</output></span>
      <span class="bar"><span class="fill a" style="width:{Number.isFinite(ex[i]) ? Math.max(1, (ex[i] / maxEx) * 100) : 100}%"></span><em>{fmt(ex[i])}</em></span>
      <span class="bar"><span class="fill b" style="width:{Number.isFinite(safe[i]) ? safe[i] * 100 : 0}%"></span><em>{pct(safe[i])}</em></span>
    </div>
  {/each}
  <div class="tot">Probabilities add up to <strong>{pct(safe.reduce((a, b) => a + b, 0))}</strong>.</div>

  <div class="two">
    <div class="box">
      <div class="h">Add the same number to every score</div>
      <label class="sl2"><input type="range" min="-10" max="10" step="0.5" bind:value={shift} oninput={touch} disabled={huge} /> <strong>{shift > 0 ? '+' : ''}{shift}</strong></label>
      <div class="note">Probabilities change by at most {(Math.max(...safe.map((p, i) => Math.abs(p - base[i]))) * 100).toFixed(4)}%: <strong>only the gaps between scores matter</strong>.</div>
    </div>
    <div class="box">
      <div class="h">Make the scores enormous (×500)</div>
      <button class="btn" onclick={() => { huge = !huge; shift = 0; touch(); }}>{huge ? 'Back to normal' : 'Make them huge'}</button>
      <div class="note">Plain e<sup>score</sup> ÷ total: <strong class:bad={huge}>{naive.map(pct).join(', ')}</strong><br />Subtract the biggest score first: <strong class="good">{safe.map(pct).join(', ')}</strong></div>
    </div>
  </div>
</div>

<style>
  .grid { display: grid; grid-template-columns: 2.2rem 1.2fr 1fr 1fr; gap: 0.7rem; align-items: center; padding: 0.25rem 0; }
  .grid.head { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; border-bottom: 1px solid var(--line); }
  @media (max-width: 760px) { .grid { grid-template-columns: 1.8rem minmax(0, 1fr) minmax(0, 1fr); } .grid > :nth-child(3) { display: none; } .grid.head > :first-child { visibility: hidden; } .sl input { min-width: 40px; } output { min-width: 2.4rem; } }
  .lt { font-size: 1.1rem; text-align: center; }
  .sl { display: flex; gap: 0.5rem; align-items: center; }
  .sl input { flex: 1; accent-color: var(--accent); min-width: 80px; }
  output { min-width: 3.2rem; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .bar { position: relative; height: 22px; background: var(--surface-2); border-radius: 5px; overflow: hidden; }
  .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 0 5px 5px 0; }
  .fill.a { background: var(--series-1); opacity: 0.55; } .fill.b { background: var(--series-1); }
  .bar em { position: relative; font-style: normal; font-size: 0.78rem; padding-left: 0.4rem; line-height: 22px; font-variant-numeric: tabular-nums; mix-blend-mode: normal; color: var(--ink); }
  .tot { margin: 0.5rem 0 0.9rem; color: var(--ink-2); }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 760px) { .two { grid-template-columns: minmax(0, 1fr); } }
  .box { background: var(--surface-2); border-radius: 12px; padding: 0.7rem 0.9rem; }
  .h { font-weight: 600; margin-bottom: 0.4rem; }
  .sl2 { display: flex; gap: 0.6rem; align-items: center; }
  .sl2 input { flex: 1; accent-color: var(--accent); }
  .note { color: var(--ink-2); font-size: 0.82rem; margin-top: 0.5rem; word-break: break-word; }
  .bad { color: var(--pain); } .good { color: var(--good); }
</style>
