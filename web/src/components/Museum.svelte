<script>
  import { data } from '../lib/data.svelte.js';
  import { progress } from '../lib/progress.svelte.js';
  import { chapters } from '../chapters/index.js';
  import { href } from '../lib/router.svelte.js';

  // The "Model Museum": every model you've built, scored by the same ruler (average surprise, lower is better).
  let { open = $bindable(false) } = $props();

  const replay = $state({ at500: null, at3000: null });
  fetch(new URL('data/long_run.json', document.baseURI)).then((r) => r.json()).then((d) => {
    replay.at500 = d.meanLoss[d.steps.indexOf(500)]; replay.at3000 = d.meanLoss[d.steps.indexOf(3000)];
  }).catch(() => {});

  const scored = $derived(!!progress.banked.ch3 && data.losses);
  const rows = $derived(data.losses ? [
    { name: 'Pure chance', note: 'every symbol equally likely', v: data.losses.chance, shown: scored },
    { name: 'Letter frequencies', note: 'ignores what came before', v: data.losses.unigram, shown: scored },
    { name: 'Bigram table', note: 'remembers one letter back', v: data.losses.bigram, shown: scored },
    { name: 'microgpt.py · 500 steps', note: 'the real file, barely trained. Built in Act IV', v: replay.at500, shown: true, target: true },
    { name: 'microgpt.py · 3,000 steps', note: 'the real file, trained 6× longer', v: replay.at3000, shown: true, target: true },
  ] : []);
  const maxV = $derived(data.losses?.chance ?? 3.3);
</script>

{#if open}
  <div class="scrim" onclick={() => (open = false)} role="presentation"></div>
  <div class="pop widget" role="dialog" aria-label="Model Museum">
    <div class="widget-title">Model Museum <button class="btn ghost x" onclick={() => (open = false)} aria-label="Close">✕</button></div>
    {#if !scored}
      <p class="empty">Nothing on display yet. Finish <a href={href('ch/3')} onclick={() => (open = false)}>Chapter 3: Scoring</a> and your first models are scored here, side by side.</p>
      <div class="row target"><div class="nm">microgpt.py · 3,000 steps<span>the target you're walking toward</span></div><div class="bar"><div class="fill" style="width:{replay.at3000 ? (replay.at3000 / 3.3) * 100 : 0}%"></div></div><div class="val">{replay.at3000?.toFixed(2) ?? '…'}</div></div>
    {:else}
      <p class="cap">Average surprise per letter. <strong>Lower is better.</strong></p>
      {#each rows as r}
        <div class="row" class:target={r.target}>
          <div class="nm">{r.name}<span>{r.note}</span></div>
          <div class="bar"><div class="fill" style="width:{r.v ? (r.v / maxV) * 100 : 0}%"></div></div>
          <div class="val">{r.v?.toFixed(3) ?? '…'}</div>
        </div>
      {/each}
    {/if}
  </div>
{/if}

<style>
  .scrim { position: fixed; inset: 0; z-index: 60; }
  .pop { position: fixed; z-index: 70; top: calc(var(--header-h) + 8px); right: 12px; width: min(440px, calc(100vw - 24px)); }
  .x { position: absolute; right: 0.7rem; top: 0.5rem; padding: 0.2rem 0.6rem; }
  .cap { margin: 0 0 0.8rem; color: var(--ink-2); }
  .empty { color: var(--ink-2); }
  .row { display: grid; grid-template-columns: 1fr 90px 52px; gap: 0.7rem; align-items: center; padding: 0.5rem 0; border-top: 1px solid var(--line); }
  .nm { font-weight: 600; font-size: 0.88rem; }
  .nm span { display: block; font-weight: 400; color: var(--ink-3); font-size: 0.75rem; }
  .bar { height: 10px; background: var(--surface-2); border-radius: 5px; overflow: hidden; }
  .fill { height: 100%; background: var(--series-1); border-radius: 5px; }
  .target .fill { background: var(--series-3); }
  .val { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .target { grid-template-columns: 1fr 90px 52px; }
</style>
