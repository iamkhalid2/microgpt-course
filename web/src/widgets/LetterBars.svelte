<script>
  import { getContext } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { tokenLabel, sum } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // How often does each symbol occur? Counts first, then the same thing as shares of the whole: probabilities.
  const beat = getContext('beat');
  let sortBy = $state('alpha');   // 'alpha' | 'freq'
  let unit = $state('count');     // 'count' | 'percent'
  let hover = $state(null);
  let touched = 0;

  const total = $derived(sum(data.uni));
  const items = $derived.by(() => {
    const a = data.uni.map((c, id) => ({ id, c, p: c / total }));
    return sortBy === 'freq' ? a.slice().sort((x, y) => y.c - x.c) : a;
  });
  const maxC = $derived(Math.max(...data.uni));
  const fmt = (it) => (unit === 'count' ? it.c.toLocaleString() : (it.p * 100).toFixed(1) + '%');
  const note = () => { if (++touched >= 2) beat?.complete(); };
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">Letter counts · all {data.docs.length.toLocaleString()} names</div>
    <div class="ctl">
      <span class="grp" role="group" aria-label="Sort"><button class:on={sortBy === 'alpha'} onclick={() => { sortBy = 'alpha'; note(); }}>A → Z</button><button class:on={sortBy === 'freq'} onclick={() => { sortBy = 'freq'; note(); }}>Most common first</button></span>
      <span class="grp" role="group" aria-label="Unit"><button class:on={unit === 'count'} onclick={() => { unit = 'count'; note(); }}>How many times</button><button class:on={unit === 'percent'} onclick={() => { unit = 'percent'; note(); }}>Share of all</button></span>
    </div>
    <div class="plot" role="img" aria-label="Bar chart of how often each letter appears">
      {#each items as it (it.id)}
        <div class="col" class:end={it.id === data.vocab.BOS} onmouseenter={() => (hover = it)} onmouseleave={() => (hover = null)} role="presentation">
          <div class="val">{hover?.id === it.id || it === items[0] ? fmt(it) : ''}</div>
          <div class="bar" style="height:{(it.c / maxC) * 100}%"></div>
          <div class="lab">{tokenLabel(it.id, data.vocab)}</div>
        </div>
      {/each}
    </div>
    <div class="read">
      {#if hover}
        <strong class="mono">{tokenLabel(hover.id, data.vocab)}</strong>
        {hover.id === data.vocab.BOS ? 'ends a name' : 'appears'} <strong>{hover.c.toLocaleString()}</strong> times out of {total.toLocaleString()} → <strong>{(hover.p * 100).toFixed(2)}%</strong>, or a probability of <strong>{hover.p.toFixed(4)}</strong>
      {:else}<span class="muted">Hover a bar. ⏎ is the end-of-name symbol: every name ends exactly once, so it appears {data.docs.length.toLocaleString()} times.</span>{/if}
    </div>
  </div>
</DataGate>

<style>
  .ctl { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 0.9rem; }
  .grp { display: inline-flex; border: 1px solid var(--line-strong); border-radius: 999px; overflow: hidden; }
  .grp button { border: 0; background: var(--surface); color: var(--ink-2); padding: 0.35rem 0.8rem; }
  .grp button.on { background: var(--accent); color: var(--on-accent); }
  .plot { display: flex; align-items: flex-end; gap: 3px; height: 210px; border-bottom: 1px solid var(--axis); padding-top: 1.4rem; }
  .col { flex: 1; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; position: relative; min-width: 0; }
  .bar { width: 100%; max-width: 24px; background: var(--series-1); border-radius: 4px 4px 0 0; min-height: 2px; transition: height 0.45s ease; }
  .col.end .bar { background: var(--series-2); }
  .plot:has(.col:hover) .col:not(:hover) .bar { opacity: 0.5; }
  .lab { position: absolute; bottom: -1.45rem; font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink-2); }
  .val { position: absolute; top: 0; font-size: 0.7rem; color: var(--ink-2); white-space: nowrap; transform: translateY(-4px); }
  .read { margin-top: 2rem; min-height: 2.6rem; color: var(--ink-2); font-size: 0.88rem; }
</style>
