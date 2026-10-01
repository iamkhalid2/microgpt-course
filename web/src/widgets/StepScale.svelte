<script>
  import { getContext } from 'svelte';
  import { stepSizes } from '../lib/opt.js';

  // Two dials whose slopes are on completely different scales. How big a step does each optimiser take on each?
  const beat = getContext('beat');
  let e1 = $state(2), e2 = $state(-2);          // slope = 10^e
  let acts = 0;
  const g1 = $derived(10 ** e1), g2 = $derived(10 ** e2);
  const plain = $derived([stepSizes('plain', g1)[0], stepSizes('plain', g2)[0]]);
  const adam = $derived([stepSizes('adam', g1)[5], stepSizes('adam', g2)[5]]);
  const fmt = (v) => (v >= 0.01 ? v.toFixed(3) : v.toExponential(1));
  const touch = () => { if (++acts >= 3) beat?.complete(); };
  const ratio = $derived(g1 / g2);
</script>

<div class="widget wide">
  <div class="widget-title">Per-dial step sizes · same learning rate (0.01), very different slopes</div>
  <div class="sl"><label>Dial A's slope <input type="range" min="-3" max="3" step="0.1" bind:value={e1} oninput={touch} aria-label="Slope of dial A (power of ten)" /> <strong>{fmt(g1)}</strong></label></div>
  <div class="sl"><label>Dial B's slope <input type="range" min="-3" max="3" step="0.1" bind:value={e2} oninput={touch} aria-label="Slope of dial B (power of ten)" /> <strong>{fmt(g2)}</strong></label></div>
  <table class="t">
    <thead><tr><td></td><td>step taken by dial A</td><td>step taken by dial B</td><td>A's step ÷ B's step</td></tr></thead>
    <tbody>
      <tr><td>Plain: lr × slope</td><td>{fmt(plain[0])}</td><td>{fmt(plain[1])}</td><td class="r">{(plain[0] / plain[1]).toLocaleString(undefined, { maximumFractionDigits: 0 })}×</td></tr>
      <tr class="ad"><td>Adam: lr × m̂ / (√v̂ + ε)</td><td>{fmt(adam[0])}</td><td>{fmt(adam[1])}</td><td class="r">{(adam[0] / adam[1]).toFixed(2)}×</td></tr>
    </tbody>
  </table>
  <div class="cap">The slopes differ by a factor of <strong>{ratio.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong>. Plain descent moves the dials by <strong>that same factor</strong> apart, so the steep dial lurches and the shallow dial barely moves. Adam divides each dial's step by its <em>own typical slope size</em> (<span class="mono">√v̂</span>), so <strong>every dial moves about the learning rate each step</strong>, steep or shallow. (Adam's column shows step 6; the first few steps are the same.)</div>
</div>

<style>
  .sl label { display: flex; gap: 0.7rem; align-items: center; color: var(--ink-2); margin-bottom: 0.3rem; } .sl input { accent-color: var(--accent); flex: 1; max-width: 22rem; } .sl strong { min-width: 5rem; font-variant-numeric: tabular-nums; }
  .t { width: 100%; border-collapse: collapse; margin: 0.8rem 0; font-variant-numeric: tabular-nums; } .t td { padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--line); } .t tbody td:not(:first-child) { white-space: nowrap; } .t thead td { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; } .t .r { font-weight: 700; } .t tr.ad td { background: var(--accent-wash); }
  .cap { color: var(--ink-2); font-size: 0.85rem; }
</style>
