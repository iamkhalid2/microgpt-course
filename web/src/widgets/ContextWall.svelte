<script>
  import { getContext, onMount } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { contextCoverage } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // Why can't we just remember more letters? Because the table outgrows the data, fast.
  const beat = getContext('beat');
  let rows = $state(null);
  let n = $state(1);
  let moved = 0;

  onMount(() => {
    const go = () => { if (data.status !== 'ready') return setTimeout(go, 150); setTimeout(() => { rows = contextCoverage(data.train, data.test, data.vocab, 6); }, 30); };
    go();
  });
  const r = $derived(rows ? rows[n - 1] : null);
  const logW = (v) => Math.max(1.5, (Math.log10(v) / 9) * 100);
  const fmt = (v) => (v >= 1e9 ? (v / 1e9).toFixed(0) + ' billion' : v >= 1e6 ? (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + ' million' : v.toLocaleString());
  function change(e) { n = +e.currentTarget.value; if (++moved >= 3) beat?.complete(); }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">The wall · remember more letters?</div>
    {#if !r}
      <div class="muted">Counting every context in 205,000 training positions…</div>
    {:else}
      <div class="ctl">
        <label for="nsl">Remember the last <strong>{n}</strong> letter{n === 1 ? '' : 's'}</label>
        <input id="nsl" type="range" min="1" max="6" value={n} oninput={change} />
      </div>
      <div class="cmp">
        <div class="line"><span class="l">Rows the table would need</span><span class="bar"><span class="f a" style="width:{logW(r.possible)}%"></span></span><span class="v">{fmt(r.possible)}</span></div>
        <div class="line"><span class="l">Rows that ever get a single example</span><span class="bar"><span class="f c" style="width:{logW(r.seen)}%"></span></span><span class="v">{fmt(r.seen)}</span></div>
        <div class="line"><span class="l">Examples in the training names</span><span class="bar"><span class="f b" style="width:{logW(r.events)}%"></span></span><span class="v">{fmt(r.events)}</span></div>
      </div>
      <div class="avg">Each filled row is learned from about <strong>{r.events / r.seen >= 10 ? Math.round(r.events / r.seen).toLocaleString() : (r.events / r.seen).toFixed(1)}</strong> example{r.events / r.seen < 1.5 ? '' : 's'} on average. {r.events / r.seen < 10 ? 'Its "probabilities" are mostly luck.' : 'Plenty to trust.'}</div>
      <div class="hero">
        <div class="big" class:bad={r.unseenRate > 0.2}>{(r.unseenRate * 100).toFixed(r.unseenRate < 0.1 ? 1 : 0)}%</div>
        <div>of the moments in names the table <strong>never studied</strong> arrive with a {n}-letter memory it has <strong>never met</strong>. At those moments it has nothing to say.</div>
      </div>
      <div class="foot muted">Bars use a log scale (each tick is 10× bigger). Tested on 10% of names held back from the table.</div>
    {/if}
  </div>
</DataGate>

<style>
  .ctl { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .ctl input { flex: 1; min-width: 180px; accent-color: var(--accent); }
  .line { display: grid; grid-template-columns: 13rem 1fr 7.5rem; gap: 0.8rem; align-items: center; margin-bottom: 0.5rem; }
  @media (max-width: 640px) { .line { grid-template-columns: 1fr; gap: 0.2rem; } }
  .l { color: var(--ink-2); font-size: 0.85rem; }
  .bar { height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
  .f { display: block; height: 100%; border-radius: 0 4px 4px 0; transition: width 0.4s ease; }
  .f.a { background: var(--series-2); }
  .f.b { background: var(--series-1); }
  .f.c { background: var(--series-3); }
  .avg { color: var(--ink-2); margin: 0.6rem 0 0; }
  .v { font-variant-numeric: tabular-nums; font-weight: 600; text-align: right; }
  .hero { display: flex; gap: 1.1rem; align-items: center; margin-top: 1.2rem; padding: 0.9rem 1rem; background: var(--surface-2); border-radius: 12px; }
  .big { font-family: var(--font-display); font-size: 2.6rem; font-weight: 700; line-height: 1; min-width: 5.5rem; }
  .big.bad { color: var(--pain); }
  .foot { font-size: 0.78rem; margin-top: 0.6rem; }
</style>
