<script>
  // Attention weights as a triangle: each row is the letter being read, each column an earlier position it may look at.
  let { tokens, weights, sel = null, onsel = null, title = '', cell = 40 } = $props();
  const shade = (w) => `color-mix(in oklab, var(--seq-700) ${Math.round(Math.sqrt(w) * 100)}%, var(--seq-100))`;
  const n = $derived(tokens.length);
  const L = 26, T = 22;
</script>
<div class="am">
  {#if title}<div class="t">{title}</div>{/if}
  <svg viewBox="0 0 {L + n * cell} {T + n * cell}" role="img" aria-label="Attention weights: rows are the letter being read, columns the earlier positions it looks at">
    {#each tokens as tk, c}<text class="axis-text" x={L + c * cell + cell / 2} y="14" text-anchor="middle">{tk}</text>{/each}
    {#each weights as row, r}
      <text class="axis-text rl" x={L - 6} y={T + r * cell + cell / 2 + 4} text-anchor="end" style="font-weight:{sel === r ? 700 : 400}">{tokens[r]}</text>
      {#each row as w, c}
        <rect x={L + c * cell + 1} y={T + r * cell + 1} width={cell - 2} height={cell - 2} rx="5" style="fill:{shade(w)};" onclick={() => onsel?.(r)} role="presentation" />
        {#if w >= 0.12}<text class="wv" x={L + c * cell + cell / 2} y={T + r * cell + cell / 2 + 4} text-anchor="middle" style="fill:{w > 0.45 ? '#fff' : 'var(--ink)'}">{Math.round(w * 100)}</text>{/if}
      {/each}
    {/each}
    {#if sel !== null}<rect x={L} y={T + sel * cell} width={n * cell} height={cell} rx="6" style="fill:none; stroke:var(--ink); stroke-width:1.6; pointer-events:none;" />{/if}
  </svg>
</div>
<style>
  .t { font-size: 0.78rem; color: var(--ink-2); font-weight: 600; margin-bottom: 0.2rem; text-align: center; }
  svg { width: 100%; height: auto; display: block; } rect { cursor: pointer; }
  .rl { font-family: var(--font-mono); font-size: 11px; } .wv { font-family: var(--font-ui); font-size: 11px; font-weight: 600; pointer-events: none; }
</style>
