<script>
  import { getContext, onDestroy } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { tokenLabel, sum } from '../lib/stats.js';
  import { makeRng, sampleIndex } from '../lib/rng.js';
  import DataGate from './DataGate.svelte';

  // Generate a name one letter at a time by looking up the previous letter's row. This loop is the heart of generation.
  const beat = getContext('beat');
  const rng = makeRng((Date.now() >>> 0) % 2147483647);
  let path = $state([]);          // tokens written so far (excluding the initial start)
  let choosing = $state(null);    // token just drawn (highlighted briefly)
  let finished = $state(0);
  let timer;
  onDestroy(() => clearTimeout(timer));

  const cur = $derived(path.length ? path[path.length - 1] : data.vocab.BOS);
  const done = $derived(path.length > 0 && path[path.length - 1] === data.vocab.BOS);
  const row = $derived(data.bi[cur]);
  const tot = $derived(sum(row));
  const bars = $derived(row.map((n, c) => ({ c, n, p: n / tot })).filter((x) => x.n > 0).sort((a, b) => b.n - a.n));
  const shown = $derived(bars.slice(0, 8));
  const rest = $derived(bars.slice(8).reduce((s, b) => s + b.p, 0));
  const maxP = $derived(bars[0]?.p ?? 1);

  function step() {
    if (choosing !== null) return;
    if (done || path.length >= 14) { path = []; return; }
    const t = sampleIndex(row, rng);
    choosing = t;
    timer = setTimeout(() => {
      path = [...path, t]; choosing = null;
      if (t === data.vocab.BOS || path.length >= 14) { finished++; if (finished >= 2) beat?.complete(); }
    }, 650);
  }
  function auto() {
    if (choosing !== null) return;
    if (done) path = [];
    const run = () => {
      if (path.length && (path[path.length - 1] === data.vocab.BOS || path.length >= 14)) return;
      const t = sampleIndex(data.bi[path.length ? path[path.length - 1] : data.vocab.BOS], rng);
      path = [...path, t];
      if (t === data.vocab.BOS || path.length >= 14) { finished++; if (finished >= 2) beat?.complete(); return; }
      timer = setTimeout(run, 260);
    };
    run();
  }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">Walk the chain · make a name letter by letter</div>
    <div class="strip mono" aria-live="polite">
      <span class="tk start">⏎</span>
      {#each path as t}<span class="tk" class:stop={t === data.vocab.BOS}>{tokenLabel(t, data.vocab)}</span>{/each}
      {#if choosing !== null}<span class="tk pend">{tokenLabel(choosing, data.vocab)}</span>{/if}
    </div>

    {#if !done}
      <div class="rt">Last symbol: <strong class="mono">{tokenLabel(cur, data.vocab)}</strong> {cur === data.vocab.BOS ? '(the start)' : ''}. Its row says what tends to come next:</div>
      {#each shown as b}
        <div class="br" class:pick={choosing === b.c}>
          <span class="bl mono">{tokenLabel(b.c, data.vocab)}</span>
          <span class="bar"><span class="fill" class:endc={b.c === data.vocab.BOS} style="width:{(b.p / maxP) * 100}%"></span></span>
          <span class="bv">{(b.p * 100).toFixed(1)}%</span>
        </div>
      {/each}
      {#if rest > 0.002}<div class="more muted">…and {bars.length - shown.length} rarer symbols share the remaining {(rest * 100).toFixed(1)}%.</div>{/if}
    {:else}
      <div class="rt"><strong>Done.</strong> The model wrote <strong class="mono">{path.slice(0, -1).map((t) => data.vocab.uchars[t]).join('') || '(nothing)'}</strong> and then chose ⏎ to end it.</div>
    {/if}

    <div class="foot">
      <button class="btn primary" onclick={step} disabled={choosing !== null}>{done ? 'New name' : 'Pick the next letter'}</button>
      <button class="btn" onclick={auto} disabled={choosing !== null}>Write a whole name</button>
      <span class="muted">The pick is random, but weighted by these percentages.</span>
    </div>
  </div>
</DataGate>

<style>
  .strip { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 1rem; min-height: 2.8rem; }
  .tk { min-width: 2.2rem; height: 2.6rem; display: grid; place-items: center; border-radius: 9px; background: var(--code-bg); font-size: 1.3rem; padding: 0 0.3rem; }
  .tk.start, .tk.stop { background: var(--gold-wash); border: 1px solid var(--gold); }
  .tk.pend { background: var(--accent-wash); border: 1px dashed var(--accent); animation: blink 0.4s ease infinite alternate; }
  @keyframes blink { to { opacity: 0.5; } }
  .rt { color: var(--ink-2); margin-bottom: 0.6rem; }
  .br { display: grid; grid-template-columns: 1.6rem 1fr 3.4rem; gap: 0.6rem; align-items: center; margin-bottom: 4px; max-width: 520px; padding: 1px 4px; border-radius: 6px; }
  .br.pick { background: var(--accent-wash); }
  .bl { text-align: center; }
  .bar { height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
  .fill { display: block; height: 100%; background: var(--series-1); border-radius: 0 4px 4px 0; }
  .fill.endc { background: var(--series-2); }
  .bv { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink-2); }
  .more { font-size: 0.82rem; margin: 0.3rem 0 0; }
  .foot { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; margin-top: 1rem; }
</style>
