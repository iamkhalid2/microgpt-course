<script>
  import { getContext } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { tokenLabel, sum } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // The bigram table as a heatmap. Row = the letter you just wrote. Column = the letter that follows.
  // Each row is scaled to add up to 100%, so a dark cell means "this is very likely next".
  const beat = getContext('beat');
  let sel = $state(null);      // selected row token
  let hov = $state(null);      // {r, c}
  let touched = new Set();

  const V = $derived(data.vocab.V);
  // display order: start row first, then a..z; columns a..z then end
  const rows = $derived([data.vocab.BOS, ...Array.from({ length: data.vocab.BOS }, (_, i) => i)]);
  const cols = $derived(Array.from({ length: data.vocab.V }, (_, i) => i));
  const rowTotal = $derived(data.bi.map((r) => sum(r)));
  const prob = (r, c) => (rowTotal[r] ? data.bi[r][c] / rowTotal[r] : 0);
  const shade = (p) => `color-mix(in oklab, var(--seq-700) ${Math.round(Math.sqrt(p) * 100)}%, var(--seq-100))`;
  const name = (t, role) => (t === data.vocab.BOS ? (role === 'row' ? 'the start of a name' : 'the end of the name') : `“${data.vocab.uchars[t]}”`);

  const CELL = 30, LM = 50, TM = 36;
  const W = $derived(LM + V * CELL + 6), H = $derived(TM + V * CELL + 6);

  const top = $derived.by(() => {
    if (sel === null) return [];
    return cols.map((c) => ({ c, p: prob(sel, c), n: data.bi[sel][c] })).filter((x) => x.n > 0).sort((a, b) => b.p - a.p).slice(0, 9);
  });
  const maxTop = $derived(top.length ? top[0].p : 1);

  function pick(r) { sel = r; touched.add(r); if (touched.size >= 3) beat?.complete(); }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">The bigram table · what follows what</div>
    <div class="read" aria-live="polite">
      {#if hov}
        After {name(hov.r, 'row')}, the next symbol is {name(hov.c, 'col')}
        <strong>{(prob(hov.r, hov.c) * 100).toFixed(1)}%</strong> of the time
        <span class="muted">({data.bi[hov.r][hov.c].toLocaleString()} of {rowTotal[hov.r].toLocaleString()})</span>
      {:else}<span class="muted">Hover a cell. Click any row to see that row as a list. (Try <strong>q</strong>, the start row, and <strong>y</strong>.)</span>{/if}
    </div>
    <svg viewBox="0 0 {W} {H}" class="grid" role="img" aria-label="Heatmap: probability of each next letter given the previous letter">
      <text class="axis-text" x={LM + (V * CELL) / 2} y="12" text-anchor="middle">…the NEXT letter is →</text>
      <text class="axis-text" transform="translate(11 {TM + (V * CELL) / 2}) rotate(-90)" text-anchor="middle">the letter just written ↓</text>
      {#each cols as c}
        <text class="axis-text cl" x={LM + c * CELL + CELL / 2} y={TM - 5} text-anchor="middle" style="fill:{c === data.vocab.BOS ? 'var(--series-2)' : ''}">{tokenLabel(c, data.vocab)}</text>
      {/each}
      {#each rows as r, ri}
        <text class="axis-text rl" x={LM - 6} y={TM + ri * CELL + CELL / 2 + 4} text-anchor="end" style="font-weight:{sel === r ? 700 : 400}" onclick={() => pick(r)} role="presentation">{r === data.vocab.BOS ? 'start' : data.vocab.uchars[r]}</text>
        {#each cols as c}
          <rect x={LM + c * CELL + 1} y={TM + ri * CELL + 1} width={CELL - 2} height={CELL - 2} rx="4"
            style="fill:{shade(prob(r, c))}; stroke:{hov && hov.r === r && hov.c === c ? 'var(--ink)' : 'none'}; stroke-width:2;"
            onmouseenter={() => (hov = { r, c })} onmouseleave={() => (hov = null)} onclick={() => pick(r)} role="presentation" />
        {/each}
      {/each}
      {#if sel !== null}
        <rect x={LM} y={TM + rows.indexOf(sel) * CELL} width={V * CELL} height={CELL} rx="5" style="fill:none; stroke:var(--ink); stroke-width:2; pointer-events:none;" />
      {/if}
    </svg>

    {#if sel !== null}
      <div class="row">
        <div class="rt">After {name(sel, 'row')}, here's what comes next (top {top.length}):</div>
        {#each top as t}
          <div class="br">
            <span class="bl mono">{tokenLabel(t.c, data.vocab)}</span>
            <span class="bar"><span class="fill" class:endc={t.c === data.vocab.BOS} style="width:{(t.p / maxTop) * 100}%"></span></span>
            <span class="bv">{(t.p * 100).toFixed(1)}%</span>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</DataGate>

<style>
  .read { min-height: 1.5rem; margin-bottom: 0.5rem; color: var(--ink); }
  .grid { width: 100%; height: auto; display: block; max-width: 880px; margin: 0 auto; }
  .cl { font-family: var(--font-mono); font-size: 11px; }
  .rl { font-family: var(--font-mono); font-size: 11px; cursor: pointer; }
  rect { cursor: pointer; }
  .row { margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--line); max-width: 560px; }
  .rt { color: var(--ink-2); margin-bottom: 0.5rem; }
  .br { display: grid; grid-template-columns: 1.6rem 1fr 3.4rem; gap: 0.6rem; align-items: center; margin-bottom: 4px; }
  .bl { text-align: center; }
  .bar { height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
  .fill { display: block; height: 100%; background: var(--series-1); border-radius: 0 4px 4px 0; }
  .fill.endc { background: var(--series-2); }
  .bv { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink-2); }
</style>
