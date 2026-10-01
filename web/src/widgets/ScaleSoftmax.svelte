<script>
  import { getContext } from 'svelte';
  import { makeRng } from '../lib/rng.js';
  import { gauss } from '../lib/optim.js';
  import { softmax } from '../lib/stats.js';

  // Why attention divides its scores by sqrt(d): dot products of longer lists are bigger, which makes softmax overconfident.
  const beat = getContext('beat');
  const DS = [1, 4, 16, 64, 256, 1024];
  let pick = $state(1);
  let seed = $state(7);
  let moved = 0;

  const d = $derived(DS[pick]);
  const sim = $derived.by(() => {
    const r = makeRng(seed * 1000 + pick);
    const g = () => gauss(r);
    const vec = () => Array.from({ length: d }, g);
    const q = vec(), keys = Array.from({ length: 6 }, vec);
    const raw = keys.map((k) => k.reduce((s, ki, i) => s + ki * q[i], 0));
    // typical size of a raw score over many random pairs
    let ss = 0; const N = 400;
    for (let n = 0; n < N; n++) { const a = vec(), b = vec(); const s = a.reduce((t, ai, i) => t + ai * b[i], 0); ss += s * s; }
    return { raw, scaled: raw.map((v) => v / Math.sqrt(d)), typ: Math.sqrt(ss / N) };
  });
  const pRaw = $derived(softmax(sim.raw)), pScaled = $derived(softmax(sim.scaled));
  const mx = (a) => Math.max(...a);
  function change(e) { pick = +e.currentTarget.value; if (++moved >= 3) beat?.complete(); }
</script>

<div class="widget wide">
  <div class="widget-title">Break it · remove the divide-by-√d</div>
  <div class="ctl">
    <label>Length of each list (d): <input type="range" min="0" max="5" step="1" value={pick} oninput={change} aria-label="Length of the lists" /> <strong>{d}</strong></label>
    <button class="btn ghost" onclick={() => seed++}>New random draw</button>
  </div>
  <div class="two">
    <div class="col">
      <div class="h">Without scaling: softmax(query · key)</div>
      {#each pRaw as p, i}<div class="b"><span class="bar"><span class="fill bad" style="width:{p * 100}%"></span></span><em>{(p * 100).toFixed(0)}%</em></div>{/each}
      <div class="note">Biggest weight: <strong class:bad={mx(pRaw) > 0.9}>{(mx(pRaw) * 100).toFixed(0)}%</strong></div>
    </div>
    <div class="col">
      <div class="h">With scaling: softmax(query · key ÷ √d)</div>
      {#each pScaled as p, i}<div class="b"><span class="bar"><span class="fill" style="width:{p * 100}%"></span></span><em>{(p * 100).toFixed(0)}%</em></div>{/each}
      <div class="note">Biggest weight: <strong>{(mx(pScaled) * 100).toFixed(0)}%</strong></div>
    </div>
  </div>
  <div class="foot">Typical size of one raw score at this length (measured over 400 random pairs): <strong>±{sim.typ.toFixed(1)}</strong>. It grows like √d ({Math.sqrt(d).toFixed(1)}), so dividing by √d keeps scores in a friendly range whatever the length.</div>
</div>

<style>
  .ctl { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.8rem; } .ctl label { display: flex; gap: 0.6rem; align-items: center; color: var(--ink-2); } .ctl input { accent-color: var(--accent); width: 160px; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; } @media (max-width: 700px) { .two { grid-template-columns: minmax(0, 1fr); } }
  .h { font-weight: 600; margin-bottom: 0.4rem; font-size: 0.85rem; }
  .b { display: grid; grid-template-columns: 1fr 2.6rem; gap: 0.5rem; align-items: center; margin-bottom: 3px; }
  .bar { height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; } .fill { display: block; height: 100%; background: var(--series-1); border-radius: 0 4px 4px 0; transition: width 0.25s; } .fill.bad { background: var(--series-2); }
  em { font-style: normal; font-size: 0.78rem; color: var(--ink-2); font-variant-numeric: tabular-nums; text-align: right; }
  .note { color: var(--ink-2); font-size: 0.82rem; margin-top: 0.4rem; } .bad { color: var(--pain); }
  .foot { margin-top: 0.8rem; background: var(--surface-2); border-radius: 10px; padding: 0.6rem 0.8rem; color: var(--ink-2); font-size: 0.85rem; }
</style>
