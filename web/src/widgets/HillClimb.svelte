<script>
  import { getContext, onDestroy } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { lossVC, bestVC } from '../lib/stats.js';
  import { makeRng } from '../lib/rng.js';
  import { gauss } from '../lib/optim.js';
  import DataGate from './DataGate.svelte';
  import Landscape from './Landscape.svelte';

  // Let the computer do what you just did: try a random nudge, keep it only if the loss drops.
  const beat = getContext('beat');
  let rng = makeRng((Date.now() >>> 0) % 2147483647);
  let exp = $state(-1.3);                    // step size = 10^exp
  let cur = $state({ a: 0.5, b: 0.5 });
  let best = $state(Infinity);
  let attempts = $state(0), accepted = $state(0);
  let trail = $state([{ a: 0.5, b: 0.5 }]);
  let rejects = $state([]);
  let hist = $state([]);
  let auto = $state(false);
  let timer;
  onDestroy(() => clearInterval(timer));

  const sigma = $derived(Math.pow(10, exp));
  const opt = $derived(bestVC(data.vc));
  const minL = $derived(lossVC(data.vc, opt.a, opt.b));
  $effect(() => { if (data.vc && best === Infinity) { best = lossVC(data.vc, cur.a, cur.b); hist = [best]; } });

  function nudge() {
    const p = { a: cur.a + sigma * gauss(rng), b: cur.b + sigma * gauss(rng) };
    const l = lossVC(data.vc, p.a, p.b);
    attempts++;
    if (l < best) { cur = p; best = l; accepted++; trail = [...trail.slice(-400), p]; }
    else rejects = [...rejects.slice(-80), p];
    hist = [...hist, best];
    if (attempts >= 50) beat?.complete();
  }
  const many = (n) => { for (let i = 0; i < n; i++) nudge(); };
  function toggleAuto() {
    if (auto) { clearInterval(timer); auto = false; return; }
    auto = true;
    timer = setInterval(() => { many(2); if (attempts >= 1500) { clearInterval(timer); auto = false; } }, 40);
  }
  function restart() {
    clearInterval(timer); auto = false; rng = makeRng((Date.now() >>> 0) % 2147483647);
    cur = { a: 0.5, b: 0.5 }; best = lossVC(data.vc, 0.5, 0.5); attempts = 0; accepted = 0; trail = [{ a: 0.5, b: 0.5 }]; rejects = []; hist = [best];
  }

  // loss-over-attempts sparkline
  const W = 360, H = 120, M = 6;
  const line = $derived.by(() => {
    if (hist.length < 2) return '';
    const n = hist.length, hi = hist[0], lo = minL - 0.002;
    return hist.map((v, i) => `${i ? 'L' : 'M'}${(M + (i / Math.max(1, n - 1)) * (W - 2 * M)).toFixed(1)},${(M + (1 - (v - lo) / (hi - lo)) * (H - 2 * M)).toFixed(1)}`).join(' ');
  });
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">The random hill-climber · same two dials, now the computer turns them</div>
    <div class="grid">
      <div>
        <div class="ctl">
          <label for="sg">Size of each nudge: <strong>{sigma < 0.01 ? sigma.toFixed(3) : sigma.toFixed(2)}</strong></label>
          <input id="sg" type="range" min="-3" max="-0.3" step="0.05" bind:value={exp} />
          <div class="hint"><span>tiny</span><span>huge</span></div>
        </div>
        <div class="btns">
          <button class="btn primary" onclick={() => many(1)}>Try 1 nudge</button>
          <button class="btn" onclick={() => many(25)}>Try 25</button>
          <button class="btn" onclick={toggleAuto}>{auto ? '❚❚ Pause' : '▶ Auto-run'}</button>
          <button class="btn ghost" onclick={restart}>Start over</button>
        </div>
        <div class="stats">
          <div><span class="l">attempts</span><strong>{attempts}</strong></div>
          <div><span class="l">kept</span><strong>{accepted}</strong></div>
          <div><span class="l">loss now</span><strong>{best.toFixed(4)}</strong></div>
          <div><span class="l">gap to best</span><strong>{(best - minL).toFixed(4)}</strong></div>
        </div>
        <svg viewBox="0 0 {W} {H}" class="spark" role="img" aria-label="Loss over attempts">
          <line x1={M} x2={W - M} y1={H - M} y2={H - M} stroke="var(--axis)" />
          <path d={line} fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" />
        </svg>
        <div class="lab">Loss after each attempt (the floor is the best possible score)</div>
      </div>
      <div>
        <Landscape c={data.vc} reveal={true} {trail} {rejects} {cur} />
        <div class="lab">Black line: dial settings that were kept. Grey dots: nudges that were rejected.</div>
      </div>
    </div>
  </div>
</DataGate>

<style>
  .grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 1.4rem; align-items: start; }
  @media (max-width: 820px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .ctl input { width: 100%; accent-color: var(--accent); }
  .hint { display: flex; justify-content: space-between; color: var(--ink-3); font-size: 0.72rem; }
  .btns { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.8rem 0; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-bottom: 0.7rem; }
  .stats div { background: var(--surface-2); border-radius: 10px; padding: 0.4rem 0.6rem; }
  .stats strong { display: block; font-size: 1.1rem; font-variant-numeric: tabular-nums; }
  .l, .lab { color: var(--ink-3); font-size: 0.74rem; }
  .spark { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 10px; }
</style>
