<script>
  import { getContext } from 'svelte';
  import DataGate from './DataGate.svelte';

  // Surprise = -ln(probability). Rare things are very surprising; certain things aren't surprising at all.
  // And logs turn multiplying into adding, which is what rescues us from the vanishing products.
  const beat = getContext('beat');
  let ea = $state(1.2);   // panel A: probability = 10^-ea
  let e1 = $state(1.2);   // panel B: two probabilities, 10^-e1 and 10^-e2
  let e2 = $state(2.5);
  const pa = $derived(Math.pow(10, -ea));
  let moved = 0;
  const p = (e) => Math.pow(10, -e);
  const s = (pr) => -Math.log(pr);
  const p1 = $derived(p(e1)), p2 = $derived(p(e2));
  const fmtP = (v) => (v >= 0.01 ? v.toFixed(3) : v.toExponential(1));
  const touch = () => { if (++moved >= 4) beat?.complete(); };

  const W = 560, H = 250, ML = 40, MR = 14, MT = 14, MB = 34;
  const xs = (pr) => ML + pr * (W - ML - MR);
  const ys = (v) => MT + (1 - Math.min(v, 7) / 7) * (H - MT - MB);
  const curve = (() => { let d = ''; for (let i = 0; i <= 200; i++) { const pr = Math.max(0.0009, i / 200); d += `${i ? 'L' : 'M'}${xs(pr).toFixed(1)},${ys(s(pr)).toFixed(1)} `; } return d; })();
  const chance = 1 / 27;
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">The surprise machine</div>

    <div class="two">
      <div>
        <div class="sub">A. How surprising is an event of probability <em>p</em>?</div>
        <svg viewBox="0 0 {W} {H}" class="plot" role="img" aria-label="Surprise as a function of probability: huge near zero, zero at one">
          {#each [0, 1, 2, 3, 4, 5, 6, 7] as t}
            <line x1={ML} x2={W - MR} y1={ys(t)} y2={ys(t)} stroke="var(--grid)" />
            <text class="axis-text" x={ML - 7} y={ys(t) + 4} text-anchor="end">{t}</text>
          {/each}
          {#each [0, 0.25, 0.5, 0.75, 1] as t}<text class="axis-text" x={xs(t)} y={H - 14} text-anchor="middle">{t}</text>{/each}
          <text class="axis-text" x={(ML + W - MR) / 2} y={H - 1} text-anchor="middle">probability p</text>
          <line x1={ML} x2={W - MR} y1={H - MB} y2={H - MB} stroke="var(--axis)" />
          <path d={curve} fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linecap="round" />
          <!-- pure-chance landmark -->
          <circle cx={xs(chance)} cy={ys(s(chance))} r="4" style="fill: var(--series-2); stroke: var(--surface); stroke-width: 2;" />
          <text class="axis-text" x={xs(chance) + 9} y={ys(s(chance)) - 6}>1 in 27 → 3.30</text>
          {#if pa >= 0.0009}
            <circle cx={xs(pa)} cy={ys(s(pa))} r="5.5" style="fill: var(--series-1); stroke: var(--surface); stroke-width: 2.5;" />
          {/if}
        </svg>
        <label class="sl">p <input type="range" min="0" max="3" step="0.01" bind:value={ea} oninput={touch} aria-label="Probability, log scale" />
          <span class="rd"><strong>{fmtP(pa)}</strong> → surprise <strong>{s(pa).toFixed(2)}</strong></span></label>
        <div class="cap">Certain things (p = 1) are zero surprise. The rarer, the more surprising, and without limit: as p → 0, surprise → ∞.</div>
      </div>

      <div>
        <div class="sub">B. Logs turn “multiply” into “add”</div>
        <label class="sl">p₁ <input type="range" min="0" max="4" step="0.01" bind:value={e1} oninput={touch} aria-label="First probability" /></label>
        <label class="sl">p₂ <input type="range" min="0" max="4" step="0.01" bind:value={e2} oninput={touch} aria-label="Second probability" /></label>
        <table class="t">
          <tbody>
            <tr><td></td><td class="h">probability</td><td class="h">surprise</td></tr>
            <tr><td>first</td><td>{fmtP(p1)}</td><td>{s(p1).toFixed(3)}</td></tr>
            <tr><td>second</td><td>{fmtP(p2)}</td><td>{s(p2).toFixed(3)}</td></tr>
            <tr class="tot"><td>both</td><td>{fmtP(p1 * p2)} <span class="op">(×)</span></td><td>{(s(p1) + s(p2)).toFixed(3)} <span class="op">(+)</span></td></tr>
          </tbody>
        </table>
        <div class="cap">Multiplying the probabilities gives a tiny, awkward number. Adding the surprises gives a friendly one, and it's the <em>same information</em>: <span class="mono">surprise(p₁×p₂) = surprise(p₁) + surprise(p₂)</span>.</div>
      </div>
    </div>
  </div>
</DataGate>

<style>
  .two { display: grid; grid-template-columns: 1.25fr 1fr; gap: 1.6rem; }
  @media (max-width: 820px) { .two { grid-template-columns: 1fr; } }
  .sub { font-weight: 600; margin-bottom: 0.4rem; }
  .plot { width: 100%; height: auto; display: block; }
  .sl { display: flex; gap: 0.6rem; align-items: center; margin: 0.4rem 0; color: var(--ink-2); flex-wrap: wrap; }
  .sl input { flex: 1; min-width: 120px; accent-color: var(--accent); }
  .rd { min-width: 11rem; font-variant-numeric: tabular-nums; }
  .cap { color: var(--ink-2); font-size: 0.85rem; margin-top: 0.5rem; }
  .t { width: 100%; border-collapse: collapse; margin-top: 0.4rem; font-variant-numeric: tabular-nums; }
  .t td { padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--line); }
  .t .h { color: var(--ink-3); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .t .tot td { font-weight: 700; background: var(--accent-wash); border-bottom: 0; }
  .op { color: var(--ink-3); font-weight: 400; }
</style>
