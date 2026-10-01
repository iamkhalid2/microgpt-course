<script>
  import { getContext, onDestroy } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { makeNB, stepNB, lossNB, probsNB } from '../lib/nb.js';
  import { tokenLabel, sum } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';
  import MiniHeat from './MiniHeat.svelte';

  // The bigram table again, but now made of 729 trainable dials. Gradient descent on the softmax-and-loss recipe
  // (using the "predicted minus actual" error signal) discovers the counting table on its own.
  const beat = getContext('beat');
  let nb = $state.raw(null);
  let lr = $state(40);
  let steps = $state(0);
  let hist = $state([]);
  let auto = $state(false);
  let sel = $state(null);
  let timer;
  onDestroy(() => clearInterval(timer));

  function init() { nb = makeNB(data.bi); steps = 0; hist = [lossNB(nb)]; sel = data.vocab.uchars.indexOf('q'); }
  $effect(() => { if (data.status === 'ready' && !nb) init(); });

  function run(n) {
    for (let i = 0; i < n; i++) stepNB(nb, lr);
    steps = nb.steps;
    hist = [...hist, lossNB(nb)];
    nb = nb;                                 // new reference: redraw the heatmaps
    if (steps >= 200) beat?.complete();
  }
  function toggle() {
    if (auto) { clearInterval(timer); auto = false; return; }
    auto = true;
    timer = setInterval(() => { run(10); if (steps >= 2000) { clearInterval(timer); auto = false; } }, 60);
  }
  function reset() { clearInterval(timer); auto = false; init(); }

  const loss = $derived(hist[hist.length - 1] ?? 0);
  const learned = $derived(nb ? nb.W.map((_, r) => (nb.rowN[r] ? probsNB(nb, r) : new Array(nb.V).fill(0))) : []);
  const counted = $derived(nb ? nb.counts.map((row, r) => (nb.rowN[r] ? row.map((c) => c / nb.rowN[r]) : row.map(() => 0))) : []);
  const top = $derived.by(() => {
    if (!nb || sel === null || !nb.rowN[sel]) return [];
    return counted[sel].map((p, c) => ({ c, pc: p, pl: learned[sel][c] })).sort((a, b) => b.pc - a.pc).slice(0, 6);
  });
  const W = 360, H = 130, M = 8;
  const line = $derived.by(() => {
    if (hist.length < 2) return '';
    const hi = Math.log(27) + 0.02, lo = 2.40;
    return hist.map((v, i) => `${i ? 'L' : 'M'}${(M + (i / (hist.length - 1)) * (W - 2 * M)).toFixed(1)},${(M + (1 - (Math.min(v, hi) - lo) / (hi - lo)) * (H - 2 * M)).toFixed(1)}`).join(' ');
  });
  const yC = (v) => M + (1 - (v - 2.40) / (Math.log(27) + 0.02 - 2.40)) * (H - 2 * M);
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">A table made of dials · train it with softmax</div>
    {#if nb}
      <div class="ctl">
        <label class="lr">Learning rate <input type="range" min="2" max="100" step="1" bind:value={lr} /> <strong>{lr}</strong></label>
        <button class="btn primary" onclick={() => run(1)}>1 step</button>
        <button class="btn" onclick={() => run(25)}>25 steps</button>
        <button class="btn" onclick={toggle}>{auto ? '❚❚ Pause' : '▶ Auto-run'}</button>
        <button class="btn ghost" onclick={reset}>Start over</button>
      </div>
      <div class="stats">
        <div><span class="l">steps</span><strong>{steps}</strong></div>
        <div><span class="l">loss</span><strong>{loss.toFixed(4)}</strong></div>
        <div><span class="l">counting table</span><strong>{data.losses.bigram.toFixed(4)}</strong></div>
        <div><span class="l">gap</span><strong>{(loss - data.losses.bigram).toFixed(4)}</strong></div>
      </div>
      <svg viewBox="0 0 {W} {H}" class="spark" role="img" aria-label="Loss falling toward the counting table's loss">
        <line x1={M} x2={W - M} y1={yC(data.losses.bigram)} y2={yC(data.losses.bigram)} stroke="var(--series-3)" stroke-width="1.5" />
        <text class="axis-text" x={W - M} y={yC(data.losses.bigram) - 5} text-anchor="end">counting table {data.losses.bigram.toFixed(3)}</text>
        <text class="axis-text" x={M + 2} y={yC(Math.log(27)) + 12}>pure chance 3.296</text>
        <path d={line} fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" />
      </svg>
      <div class="maps">
        <div><div class="h">What the dials say (after softmax)</div><MiniHeat rows={learned} hl={sel} /></div>
        <div><div class="h">What counting said (Chapter 2)</div><MiniHeat rows={counted} hl={sel} /></div>
        <div class="rowcmp">
          <label class="h" for="rs">Compare one row:</label>
          <select id="rs" bind:value={sel}>{#each data.vocab.uchars as ch, i}<option value={i}>after “{ch}”</option>{/each}</select>
          {#each top as t}
            <div class="cmp"><span class="mono">{tokenLabel(t.c, data.vocab)}</span><span class="b2"><span class="x" style="width:{t.pc * 100}%"></span><span class="y" style="width:{t.pl * 100}%"></span></span><span class="v">{(t.pl * 100).toFixed(0)}% / {(t.pc * 100).toFixed(0)}%</span></div>
          {/each}
          <div class="lab">dials / counting</div>
        </div>
      </div>
    {/if}
  </div>
</DataGate>

<style>
  .ctl { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.7rem; }
  .lr { display: flex; gap: 0.5rem; align-items: center; color: var(--ink-2); margin-right: 0.5rem; }
  .lr input { accent-color: var(--accent); width: 120px; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-bottom: 0.6rem; }
  .stats div { background: var(--surface-2); border-radius: 10px; padding: 0.4rem 0.6rem; }
  .stats strong { display: block; font-size: 1.05rem; font-variant-numeric: tabular-nums; }
  .l, .lab { color: var(--ink-3); font-size: 0.74rem; }
  .spark { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 10px; margin-bottom: 0.8rem; }
  .maps { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; align-items: start; }
  @media (max-width: 860px) { .maps { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); } .rowcmp { grid-column: 1 / -1; } }
  .h { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin-bottom: 0.3rem; display: block; }
  select { padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); margin-bottom: 0.5rem; }
  .cmp { display: grid; grid-template-columns: 1.4rem 1fr 4.6rem; gap: 0.4rem; align-items: center; margin-bottom: 3px; }
  .b2 { position: relative; height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
  .x { position: absolute; left: 0; top: 0; height: 7px; background: var(--series-3); }
  .y { position: absolute; left: 0; bottom: 0; height: 7px; background: var(--series-1); }
  .v { font-size: 0.74rem; color: var(--ink-2); text-align: right; font-variant-numeric: tabular-nums; }
</style>
