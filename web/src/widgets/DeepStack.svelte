<script>
  import { getContext } from 'svelte';
  import { runDeep, verdict, peak } from '../lib/deep.js';

  // A stack of blocks with random starting dials. Watch the signal (forward) and the slope (backward) at every layer.
  const beat = getContext('beat');
  let std = $state(0.02);
  let L = $state(16);
  let norm = $state(false);
  let skip = $state(false);
  let acts = new Set();
  const res = $derived(runDeep({ L, std, norm, skip }));
  const v = $derived(verdict(res.fwd));
  const W = 330, H = 190, ML = 40, MR = 8, MT = 10, MB = 24;
  const FLOOR = -12, CEIL = 12;           // log10 range shown
  const ly = (val) => { const l = Number.isFinite(val) && val > 0 ? Math.log10(val) : FLOOR; return MT + (1 - (Math.max(FLOOR, Math.min(CEIL, l)) - FLOOR) / (CEIL - FLOOR)) * (H - MT - MB); };
  const lx = (i, n) => ML + (i / Math.max(1, n)) * (W - ML - MR);
  const path = (a) => a.map((val, i) => `${i ? 'L' : 'M'}${lx(i, a.length - 1).toFixed(1)},${ly(val).toFixed(1)}`).join(' ');
  function set(fn, k) { fn(); acts.add(k); if (acts.size >= 3) beat?.complete(); }
  const presets = [['A plain stack', false, false], ['+ RMSNorm', true, false], ['+ skip lane', false, true], ['Both (what microgpt uses)', true, true]];
</script>

<div class="widget wide">
  <div class="widget-title">Deep stacks · what survives?</div>
  <div class="ctl">
    {#each presets as [name, n, s]}<button class="chip" class:on={norm === n && skip === s} onclick={() => set(() => { norm = n; skip = s; }, name)}>{name}</button>{/each}
  </div>
  <div class="ctl2">
    <label>Layers <input type="range" min="1" max="24" step="1" bind:value={L} oninput={() => acts.add('L')} aria-label="Number of layers" /> <strong>{L}</strong></label>
    <label>Starting dial size
      <select bind:value={std} onchange={() => set(() => {}, 'std')} aria-label="Starting dial size"><option value={0.02}>small, 0.02 (what microgpt uses)</option><option value={0.2}>ten times bigger, 0.2</option></select>
    </label>
  </div>
  <div class="two">
    <div>
      <div class="h">The signal going forward: size at each layer</div>
      <svg viewBox="0 0 {W} {H}" class="ch" role="img" aria-label="Signal size by layer, log scale">
        {#each [-10, -5, 0, 5, 10] as t}<line x1={ML} x2={W - MR} y1={ly(10 ** t)} y2={ly(10 ** t)} stroke="var(--grid)" /><text class="axis-text" x={ML - 5} y={ly(10 ** t) + 4} text-anchor="end">1e{t}</text>{/each}
        <path d={path(res.fwd)} fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" />
        <text class="axis-text" x={ML} y={H - 6}>layer 0</text><text class="axis-text" x={W - MR} y={H - 6} text-anchor="end">layer {L}</text>
      </svg>
    </div>
    <div>
      <div class="h">The slope coming back: size at each layer</div>
      <svg viewBox="0 0 {W} {H}" class="ch" role="img" aria-label="Slope size by layer, log scale">
        {#each [-10, -5, 0, 5, 10] as t}<line x1={ML} x2={W - MR} y1={ly(10 ** t)} y2={ly(10 ** t)} stroke="var(--grid)" /><text class="axis-text" x={ML - 5} y={ly(10 ** t) + 4} text-anchor="end">1e{t}</text>{/each}
        <path d={path(res.bwd)} fill="none" stroke="var(--series-2)" stroke-width="2" stroke-linejoin="round" />
        <text class="axis-text" x={ML} y={H - 6}>layer 0 (the start)</text><text class="axis-text" x={W - MR} y={H - 6} text-anchor="end">layer {L} (the loss)</text>
      </svg>
    </div>
  </div>
  <div class="verdict" class:bad={v !== 'healthy'} class:ok={v === 'healthy'}>
    {#if v === 'vanished'}<strong>The signal vanished.</strong> By the last layer it is {res.fwd[L].toExponential(1)}: essentially zero. Nothing the early layers learned can reach the output, and no slope can reach them.
    {:else if v === 'exploded'}<strong>The signal exploded.</strong> {res.fwd.every(Number.isFinite) ? 'At its worst the signal reached ' + peak(res.fwd).toExponential(1) : 'The signal grew past the largest number a computer can store'}. The loss would be garbage and training would crash.
    {:else}<strong>Healthy.</strong> The signal stays at a sensible size ({res.fwd[0].toFixed(2)} at the start, {res.fwd[L].toFixed(2)} at the end), and slopes of a useful size reach the first layer.{/if}
  </div>
</div>

<style>
  .ctl { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.6rem; } .chip { padding: 0.3rem 0.8rem; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); font-size: 0.8rem; } .chip.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .ctl2 { display: flex; gap: 1.2rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.8rem; } .ctl2 label { display: flex; gap: 0.5rem; align-items: center; color: var(--ink-2); font-size: 0.85rem; } .ctl2 input { accent-color: var(--accent); width: 130px; }
  select { padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; } @media (max-width: 760px) { .two { grid-template-columns: minmax(0, 1fr); } }
  .h { font-weight: 600; font-size: 0.8rem; margin-bottom: 0.2rem; } .ch { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 10px; }
  .verdict { margin-top: 0.8rem; padding: 0.6rem 0.9rem; border-radius: 10px; } .verdict.bad { background: var(--pain-wash); } .verdict.ok { background: var(--good-wash); }
</style>
