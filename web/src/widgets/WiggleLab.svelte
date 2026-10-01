<script>
  import { getContext } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { lossVC, slopesVC } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // The wiggle test: nudge a dial by h, see how much the loss changes, divide by h.
  // Shrink h and the answer settles on one number: the slope at that exact spot.
  const beat = getContext('beat');
  let x = $state(0.5);           // where dial A is now (dial B is held at 50%)
  let e = $state(-0.4);          // h = 10^e
  let done = false;

  const B = 0.5;
  const h = $derived(Math.pow(10, e));
  const L = (a) => lossVC(data.vc, a, B);
  const y0 = $derived(L(x));
  const y1 = $derived(L(x + h));
  const est = $derived((y1 - y0) / h);
  const exact = $derived(slopesVC(data.vc, x, B).da);
  const hs = [0.4, 0.1, 0.01, 0.001, 0.00001];
  const table = $derived(hs.map((hh) => ({ hh, s: (L(x + hh) - y0) / hh })));
  $effect(() => { if (!done && e < -3) { done = true; beat?.complete(); } });

  const W = 560, H = 270, ML = 44, MR = 14, MT = 12, MB = 32, YLO = 0.55, YHI = 1.5;
  const X = (a) => ML + a * (W - ML - MR);
  const Y = (v) => MT + (1 - (Math.min(v, YHI + 0.3) - YLO) / (YHI - YLO)) * (H - MT - MB);
  const curve = $derived.by(() => { let d = ''; for (let i = 0; i <= 160; i++) { const a = 0.04 + (i / 160) * 0.92; d += `${i ? 'L' : 'M'}${X(a).toFixed(1)},${Y(L(a)).toFixed(1)} `; } return d; });
  // secant extended across the plot
  const sx0 = 0.04, sx1 = 0.96;
  const secY = (a) => y0 + est * (a - x);
  const tanY = (a) => y0 + exact * (a - x);
  const clampLine = (f) => `M${X(sx0)},${Y(f(sx0))} L${X(sx1)},${Y(f(sx1))}`;
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">The wiggle test · how steep is the loss right here?</div>
    <div class="two">
      <div>
        <svg viewBox="0 0 {W} {H}" class="plot" role="img" aria-label="Loss as dial A changes, with a line through two nearby points">
          {#each [0.6, 0.8, 1.0, 1.2, 1.4] as t}
            <line x1={ML} x2={W - MR} y1={Y(t)} y2={Y(t)} stroke="var(--grid)" />
            <text class="axis-text" x={ML - 7} y={Y(t) + 4} text-anchor="end">{t.toFixed(1)}</text>
          {/each}
          {#each [0, 0.25, 0.5, 0.75, 1] as t}<text class="axis-text" x={X(t)} y={H - 14} text-anchor="middle">{t}</text>{/each}
          <text class="axis-text" x={(ML + W - MR) / 2} y={H - 1} text-anchor="middle">Dial A (dial B held at 0.5)</text>
          <text class="axis-text" transform="translate(11 {H / 2}) rotate(-90)" text-anchor="middle">loss</text>
          <line x1={ML} x2={W - MR} y1={H - MB} y2={H - MB} stroke="var(--axis)" />
          <path d={clampLine(tanY)} fill="none" style="stroke: var(--ink-3); stroke-width: 1; opacity: 0.55;" />
          <path d={curve} fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linecap="round" />
          <path d={clampLine(secY)} fill="none" style="stroke: var(--series-2); stroke-width: 2;" />
          <circle cx={X(x)} cy={Y(y0)} r="5.5" style="fill: var(--series-1); stroke: var(--surface); stroke-width: 2.5;" />
          <circle cx={X(x + h)} cy={Y(y1)} r="5.5" style="fill: var(--series-2); stroke: var(--surface); stroke-width: 2.5;" />
        </svg>
        <div class="leg"><span class="k b"></span> loss as dial A moves <span class="k o"></span> the line through your two test points <span class="k g"></span> the true slope here</div>
      </div>
      <div class="side">
        <label class="sl">Where dial A sits <input type="range" min="0.05" max="0.95" step="0.01" bind:value={x} aria-label="Dial A position" /> <strong>{x.toFixed(2)}</strong></label>
        <label class="sl">Size of the wiggle, h <input type="range" min="-5" max="-0.3" step="0.05" bind:value={e} aria-label="Wiggle size (log)" /> <strong>{h < 0.001 ? h.toExponential(0) : h.toFixed(3)}</strong></label>
        <div class="calc2">
          <div>loss at A = {x.toFixed(2)} → <strong>{y0.toFixed(4)}</strong></div>
          <div>loss at A = {(x + h).toFixed(h < 0.001 ? 5 : 3)} → <strong>{y1.toFixed(4)}</strong></div>
          <div class="eq">slope ≈ (change in loss) ÷ (change in dial) = <strong>{est.toFixed(4)}</strong></div>
        </div>
        <table class="t">
          <thead><tr><td>wiggle h</td><td>slope estimate</td></tr></thead>
          <tbody>{#each table as r}<tr><td>{r.hh}</td><td>{r.s.toFixed(4)}</td></tr>{/each}
            <tr class="ex"><td>true slope</td><td>{exact.toFixed(4)}</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>
</DataGate>

<style>
  .two { display: grid; grid-template-columns: 1.3fr 1fr; gap: 1.4rem; }
  @media (max-width: 860px) { .two { grid-template-columns: minmax(0, 1fr); } }
  .plot { width: 100%; height: auto; display: block; }
  .leg { color: var(--ink-3); font-size: 0.74rem; display: flex; flex-wrap: wrap; gap: 0.3rem 0.9rem; align-items: center; }
  .k { display: inline-block; width: 18px; height: 3px; border-radius: 2px; margin-right: 0.3rem; vertical-align: middle; }
  .k.b { background: var(--series-1); } .k.o { background: var(--series-2); } .k.g { background: var(--ink-3); }
  .sl { display: flex; gap: 0.6rem; align-items: center; margin-bottom: 0.5rem; color: var(--ink-2); flex-wrap: wrap; }
  .sl input { flex: 1; min-width: 100px; accent-color: var(--accent); }
  .calc2 { background: var(--surface-2); border-radius: 10px; padding: 0.6rem 0.8rem; margin: 0.5rem 0; font-variant-numeric: tabular-nums; }
  .eq { margin-top: 0.3rem; padding-top: 0.3rem; border-top: 1px solid var(--line-strong); }
  .t { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .t td { padding: 0.25rem 0.4rem; border-bottom: 1px solid var(--line); }
  .t thead td { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .t .ex td { font-weight: 700; background: var(--good-wash); border-bottom: 0; }
</style>
