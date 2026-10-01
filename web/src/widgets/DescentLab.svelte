<script>
  import { getContext, onDestroy } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { lossVC, bestVC, wiggleVC } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';
  import Landscape from './Landscape.svelte';

  // Gradient descent: look at the slope of each dial, then step the opposite way. Repeat.
  const beat = getContext('beat');
  let lr = $state(0.2);
  let a = $state(0.5), b = $state(0.5);
  let trail = $state([{ a: 0.5, b: 0.5 }]);
  let hist = $state([]);
  let steps = $state(0);
  let last = $state(null);
  let broke = $state(false);
  let auto = $state(false);
  let reached = $state(null);
  let timer;
  onDestroy(() => clearInterval(timer));

  const opt = $derived(bestVC(data.vc));
  const minL = $derived(lossVC(data.vc, opt.a, opt.b));
  const loss = $derived(lossVC(data.vc, a, b));
  $effect(() => { if (data.vc && hist.length === 0) hist = [lossVC(data.vc, 0.5, 0.5)]; });

  function step() {
    if (broke) return;
    const g = wiggleVC(data.vc, a, b);
    const na = a - lr * g.da, nb = b - lr * g.db;
    last = { a, b, ga: g.da, gb: g.db, na, nb };
    steps++;
    a = na; b = nb;
    const l = lossVC(data.vc, a, b);
    if (!Number.isFinite(l)) { broke = true; clearInterval(timer); auto = false; hist = [...hist, hist[0] * 1.6]; return; }
    trail = [...trail, { a, b }];
    hist = [...hist, l];
    if (reached === null && l < minL + 0.003) reached = steps;
    if (steps >= 5) beat?.complete();
  }
  function toggleAuto() {
    if (auto) { clearInterval(timer); auto = false; return; }
    auto = true;
    timer = setInterval(() => { step(); if (steps >= 80 || broke) { clearInterval(timer); auto = false; } }, 350);
  }
  function reset() { clearInterval(timer); auto = false; a = 0.5; b = 0.5; trail = [{ a: 0.5, b: 0.5 }]; hist = [lossVC(data.vc, 0.5, 0.5)]; steps = 0; last = null; broke = false; reached = null; }

  const W = 360, H = 120, M = 6;
  const line = $derived.by(() => {
    if (hist.length < 2) return '';
    const hi = Math.max(...hist), lo = minL - 0.002;
    return hist.map((v, i) => `${i ? 'L' : 'M'}${(M + (i / Math.max(1, hist.length - 1)) * (W - 2 * M)).toFixed(1)},${(M + (1 - (v - lo) / (hi - lo)) * (H - 2 * M)).toFixed(1)}`).join(' ');
  });
  const f = (v) => (v >= 0 ? '+' : '') + v.toFixed(3);
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">Follow the slope · gradient descent</div>
    <div class="grid">
      <div>
        <label class="sl">Learning rate (how big a step): <input type="range" min="0.02" max="1.2" step="0.01" bind:value={lr} /> <strong>{lr.toFixed(2)}</strong></label>
        <div class="btns">
          <button class="btn primary" onclick={step} disabled={broke}>Take one step</button>
          <button class="btn" onclick={toggleAuto} disabled={broke}>{auto ? '❚❚ Pause' : '▶ Auto-run'}</button>
          <button class="btn ghost" onclick={reset}>Start over</button>
        </div>
        {#if last}
          <table class="upd">
            <thead><tr><td></td><td>was</td><td>slope</td><td>step = lr × slope</td><td>now</td></tr></thead>
            <tbody>
              <tr><td>Dial A</td><td>{last.a.toFixed(3)}</td><td>{f(last.ga)}</td><td>−{(lr * last.ga).toFixed(3)}</td><td><strong>{last.na.toFixed(3)}</strong></td></tr>
              <tr><td>Dial B</td><td>{last.b.toFixed(3)}</td><td>{f(last.gb)}</td><td>−{(lr * last.gb).toFixed(3)}</td><td><strong>{last.nb.toFixed(3)}</strong></td></tr>
            </tbody>
          </table>
        {:else}
          <div class="muted">The rule: <span class="mono">new dial = old dial − learning rate × slope</span>. Take a step to see it happen.</div>
        {/if}
        <div class="stats">
          <div><span class="l">steps</span><strong>{steps}</strong></div>
          <div><span class="l">loss</span><strong>{broke ? '∞' : loss.toFixed(4)}</strong></div>
          <div><span class="l">loss tests used</span><strong>{steps * 3}</strong></div>
        </div>
        {#if broke}
          <div class="warn"><strong>Too big a step.</strong> The dial was thrown out of its allowed range (a probability can't go past 100% or below 0%), so the loss is ∞. Press “Start over”, and lower the learning rate.</div>
        {:else if reached !== null}
          <div class="ok">Within 0.003 of the best possible loss after <strong>{reached} step{reached === 1 ? '' : 's'}</strong> ({reached * 3} loss tests). Random nudging needed a median of 22 attempts.</div>
        {/if}
        <svg viewBox="0 0 {W} {H}" class="spark" role="img" aria-label="Loss after each step">
          <line x1={M} x2={W - M} y1={H - M} y2={H - M} stroke="var(--axis)" />
          <path d={line} fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" />
        </svg>
      </div>
      <div>
        <Landscape c={data.vc} reveal={true} {trail} cur={{ a, b }} />
        <div class="lab">The dots on the line are where the dials have been.</div>
      </div>
    </div>
  </div>
</DataGate>

<style>
  .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 1.4rem; align-items: start; }
  @media (max-width: 860px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .sl { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; color: var(--ink-2); }
  .sl input { flex: 1; min-width: 100px; accent-color: var(--accent); }
  .btns { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.7rem 0; }
  .upd { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; margin-bottom: 0.5rem; }
  .upd td { padding: 0.3rem 0.4rem; border-bottom: 1px solid var(--line); text-align: right; }
  .upd td:first-child { text-align: left; font-weight: 600; }
  .upd thead td { color: var(--ink-3); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin: 0.6rem 0; }
  .stats div { background: var(--surface-2); border-radius: 10px; padding: 0.4rem 0.6rem; }
  .stats strong { display: block; font-size: 1.1rem; }
  .l, .lab { color: var(--ink-3); font-size: 0.74rem; }
  .warn, .ok { padding: 0.55rem 0.8rem; border-radius: 10px; margin-bottom: 0.5rem; }
  .warn { background: var(--pain-wash); } .ok { background: var(--good-wash); }
  .spark { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 10px; }
</style>
