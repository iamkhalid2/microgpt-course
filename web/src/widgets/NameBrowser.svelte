<script>
  import { getContext } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { makeRng } from '../lib/rng.js';
  import DataGate from './DataGate.svelte';

  // Look at the raw material: the real names, and how long they are.
  const beat = getContext('beat');
  let seed = $state(7);
  let clicks = 0;
  const sample = $derived.by(() => { const r = makeRng(seed); return Array.from({ length: 24 }, () => r.pick(data.docs)); });
  const hist = $derived.by(() => {
    const c = new Array(16).fill(0);
    for (const d of data.docs) c[d.length]++;
    return c;
  });
  const maxH = $derived(Math.max(...hist));
  const longest = $derived(data.docs.reduce((a, b) => (b.length > a.length ? b : a), ''));
  const mean = $derived(data.docs.reduce((s, d) => s + d.length, 0) / data.docs.length);
  let hover = $state(null);
  function shuffle() { seed = Math.floor(Math.random() * 1e9); if (++clicks >= 2) beat?.complete(); }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">The raw material · {data.docs.length.toLocaleString()} real names</div>
    <div class="names">
      {#each sample as n}<span class="mono n">{n}</span>{/each}
    </div>
    <button class="btn" onclick={shuffle}>↻ Show 24 others</button>

    <div class="hist" role="img" aria-label="How many names have each length">
      <div class="htitle">How long are the names?</div>
      <div class="bars">
        {#each hist as c, len}
          {#if len >= 2}
            <div class="col" onmouseenter={() => (hover = len)} onmouseleave={() => (hover = null)} role="presentation">
              <div class="bar" style="height:{(c / maxH) * 100}%" class:hl={hover === len}></div>
              <div class="l">{len}</div>
            </div>
          {/if}
        {/each}
      </div>
      <div class="tip2">{hover ? `${hist[hover].toLocaleString()} names have ${hover} letters` : `Average length: ${mean.toFixed(1)} letters. Longest: “${longest}” (${longest.length}).`}</div>
    </div>
  </div>
</DataGate>

<style>
  .names { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.8rem; }
  .n { background: var(--code-bg); padding: 0.25rem 0.6rem; border-radius: 8px; font-size: 0.92rem; }
  .hist { margin-top: 1.2rem; }
  .htitle { color: var(--ink-3); font-size: 0.78rem; margin-bottom: 0.4rem; }
  .bars { display: flex; align-items: flex-end; gap: 6px; height: 120px; border-bottom: 1px solid var(--axis); }
  .col { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%; max-width: 40px; position: relative; }
  .bar { width: 100%; max-width: 24px; background: var(--series-1); border-radius: 4px 4px 0 0; min-height: 1px; transition: opacity 0.15s; }
  .bars:has(.hl) .bar:not(.hl) { opacity: 0.45; }
  .l { position: absolute; bottom: -1.3rem; font-size: 0.72rem; color: var(--ink-3); }
  .tip2 { margin-top: 1.7rem; color: var(--ink-2); font-size: 0.85rem; min-height: 1.3rem; }
</style>
