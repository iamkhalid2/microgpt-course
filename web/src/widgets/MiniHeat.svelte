<script>
  // A small 27x27 heatmap of probabilities (rows sum to 1). Same colour scale as the big bigram table.
  let { rows, labels, hl = null, size = 9 } = $props();
  const V = $derived(rows.length);
  const shade = (p) => `color-mix(in oklab, var(--seq-700) ${Math.round(Math.sqrt(p) * 100)}%, var(--seq-100))`;
</script>
<svg viewBox="0 0 {V * size + 2} {V * size + 2}" class="mh" role="img" aria-label="Heatmap of next-symbol probabilities">
  {#each rows as row, r}
    {#each row as p, c}<rect x={1 + c * size} y={1 + r * size} width={size - 0.6} height={size - 0.6} rx="1.5" style="fill:{shade(p)};" />{/each}
  {/each}
  {#if hl !== null}<rect x="0.5" y={0.5 + hl * size} width={V * size + 1} height={size + 0.5} rx="2" style="fill:none; stroke:var(--ink); stroke-width:1.4;" />{/if}
</svg>
<style>.mh { width: 100%; height: auto; display: block; }</style>
