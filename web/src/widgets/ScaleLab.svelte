<script>
  import { getContext, onMount } from 'svelte';
  import { bowlRun, medianAttempts } from '../lib/optim.js';

  // What happens to random hill-climbing as the number of dials grows? A real simulation, run right here.
  const beat = getContext('beat');
  const DS = [1, 2, 4, 8, 16, 32, 64, 128];
  let rows = $state(null);
  let pick = $state(5);               // index into DS
  let run = $state(null);
  let seed = 11;

  onMount(() => { setTimeout(() => { rows = DS.map((d) => ({ d, n: medianAttempts(d) })); }, 30); });
  const maxN = $derived(rows ? Math.max(...rows.map((r) => r.n)) : 1);
  function go() {
    seed++;
    const r = bowlRun(DS[pick], seed, { keepEvery: 1 });
    run = { d: DS[pick], attempts: r.attempts, hist: r.history };
    beat?.complete();
  }

  const W = 420, H = 190, ML = 40, MR = 12, MT = 10, MB = 28;
  const ly = (v) => MT + (1 - (Math.log10(Math.max(v, 0.01)) + 2) / 2) * (H - MT - MB);   // log scale: 0.01 .. 1
  const path = $derived.by(() => {
    if (!run) return '';
    const n = run.hist.length, step = Math.max(1, Math.floor(n / 300));
    let d = '';
    for (let i = 0; i < n; i += step) d += `${d ? 'L' : 'M'}${(ML + (i / (n - 1)) * (W - ML - MR)).toFixed(1)},${ly(run.hist[i]).toFixed(1)} `;
    return d;
  });
</script>

<div class="widget wide">
  <div class="widget-title">What happens with more dials?</div>
  {#if !rows}
    <div class="muted">Running the experiment…</div>
  {:else}
    <div class="two">
      <div>
        <div class="sub">Attempts needed to get the loss under 0.01 (median of 5 runs)</div>
        {#each rows as r}
          <div class="row">
            <span class="d">{r.d} dial{r.d === 1 ? '' : 's'}</span>
            <span class="bar"><span class="fill" style="width:{(r.n / maxN) * 100}%"></span></span>
            <span class="n">{r.n.toLocaleString()}</span>
            <span class="per">{(r.n / r.d).toFixed(1)} per dial</span>
          </div>
        {/each}
        <div class="cap">Each row is a bowl-shaped problem with that many dials. Double the dials, roughly double the attempts.</div>
      </div>
      <div>
        <div class="sub">Watch one run</div>
        <label class="pk">Dials: <input type="range" min="0" max="7" step="1" bind:value={pick} aria-label="Number of dials" /> <strong>{DS[pick]}</strong></label>
        <button class="btn primary" onclick={go}>Run the hill-climber</button>
        {#if run}
          <svg viewBox="0 0 {W} {H}" class="plot" role="img" aria-label="Loss falling over attempts, log scale">
            {#each [1, 0.1, 0.01] as t}
              <line x1={ML} x2={W - MR} y1={ly(t)} y2={ly(t)} stroke="var(--grid)" />
              <text class="axis-text" x={ML - 6} y={ly(t) + 4} text-anchor="end">{t}</text>
            {/each}
            <line x1={ML} x2={W - MR} y1={H - MB} y2={H - MB} stroke="var(--axis)" />
            <path d={path} fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" />
            <text class="axis-text" x={ML} y={H - 8}>0</text>
            <text class="axis-text" x={W - MR} y={H - 8} text-anchor="end">{run.attempts.toLocaleString()} attempts</text>
          </svg>
          <div class="cap">{run.d} dial{run.d === 1 ? '' : 's'}: reached 0.01 after <strong>{run.attempts.toLocaleString()}</strong> attempts. (Log scale: each gridline is 10× lower.)</div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .two { display: grid; grid-template-columns: 1.2fr 1fr; gap: 1.5rem; }
  @media (max-width: 820px) { .two { grid-template-columns: minmax(0, 1fr); } }
  .sub { font-weight: 600; margin-bottom: 0.5rem; }
  .row { display: grid; grid-template-columns: 5.2rem 1fr 3.4rem 5.4rem; gap: 0.6rem; align-items: center; margin-bottom: 0.3rem; }
  @media (max-width: 520px) { .row { grid-template-columns: 4.2rem 1fr 3rem; } .per { display: none; } }
  .d { color: var(--ink-2); font-size: 0.82rem; }
  .bar { height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
  .fill { display: block; height: 100%; background: var(--series-1); border-radius: 0 4px 4px 0; }
  .n { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .per { color: var(--ink-3); font-size: 0.74rem; }
  .cap { color: var(--ink-2); font-size: 0.82rem; margin-top: 0.5rem; }
  .pk { display: flex; gap: 0.6rem; align-items: center; margin: 0.2rem 0 0.6rem; color: var(--ink-2); }
  .pk input { flex: 1; accent-color: var(--accent); }
  .plot { width: 100%; height: auto; display: block; margin-top: 0.6rem; }
</style>
