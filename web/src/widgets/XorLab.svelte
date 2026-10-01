<script>
  import { getContext, onDestroy } from 'svelte';
  import { makeData, makeNet, trainSteps, accuracy, probAt } from '../lib/xor.js';

  // Why a hidden layer needs a nonlinearity: train three small networks on the same XOR-like pattern.
  const beat = getContext('beat');
  const pts = makeData();
  const MODES = [
    { k: 'none', label: 'No hidden layer', note: 'one scorecard straight to the answer' },
    { k: 'linear', label: 'Hidden layer, no ReLU', note: 'two scorecards stacked, nothing in between' },
    { k: 'relu', label: 'Hidden layer + ReLU', note: 'a gate between the two scorecards' },
  ];
  let mode = $state('none');
  let net = $state.raw(makeNet('none'));
  let steps = $state(0);
  let version = $state(0);
  let auto = $state(false);
  let trained = $state({});
  let timer;
  onDestroy(() => clearInterval(timer));

  function pick(k) { clearInterval(timer); auto = false; mode = k; net = makeNet(k); steps = 0; version++; }
  function train(n) {
    trainSteps(net, pts, n); steps = net.steps; version++;
    trained[mode] = Math.max(trained[mode] ?? 0, steps);
    if (Object.values(trained).filter((v) => v >= 300).length >= 2) beat?.complete();
  }
  function toggle() {
    if (auto) { clearInterval(timer); auto = false; return; }
    auto = true; timer = setInterval(() => { train(60); if (steps >= 3000) { clearInterval(timer); auto = false; } }, 50);
  }
  const acc = $derived((version, accuracy(net, pts)));
  const G = 28;
  const grid = $derived.by(() => { version; const out = []; for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) out.push({ i, j, p: probAt(net, -1 + ((i + 0.5) / G) * 2, 1 - ((j + 0.5) / G) * 2) }); return out; });
  const S = 300;
</script>

<div class="widget wide">
  <div class="widget-title">Why a gate between the layers · the XOR puzzle</div>
  <div class="modes">{#each MODES as m}<button class="chip" class:on={mode === m.k} onclick={() => pick(m.k)}>{m.label}</button>{/each}</div>
  <div class="grid">
    <svg viewBox="0 0 {S} {S}" class="plot" role="img" aria-label="Points of two kinds, with the network's decision shading">
      {#each grid as c}<rect x={(c.i / G) * S} y={(c.j / G) * S} width={S / G + 0.5} height={S / G + 0.5} style="fill: color-mix(in oklab, var(--series-1) {Math.round(c.p * 60)}%, color-mix(in oklab, var(--series-2) {Math.round((1 - c.p) * 60)}%, var(--surface)));" />{/each}
      <line x1={S / 2} x2={S / 2} y1="0" y2={S} stroke="var(--axis)" /><line y1={S / 2} y2={S / 2} x1="0" x2={S} stroke="var(--axis)" />
      {#each pts as p}
        <circle cx={(p.x + 1) * S / 2} cy={(1 - p.y) * S / 2} r="3.4" style="fill: {p.t ? 'var(--series-1)' : 'var(--series-2)'}; stroke: var(--surface); stroke-width: 1.4;" />
      {/each}
    </svg>
    <div class="side">
      <div class="note"><strong>The puzzle:</strong> blue points are where x and y have the <em>same sign</em>; orange where they differ. A single straight line can't separate them.</div>
      <div class="note2">{MODES.find((m) => m.k === mode).note}</div>
      <div class="stats"><div><span class="l">steps</span><strong>{steps}</strong></div><div><span class="l">points classified correctly</span><strong>{(acc * 100).toFixed(1)}%</strong></div></div>
      <div class="btns">
        <button class="btn primary" onclick={() => train(100)}>Train 100 steps</button>
        <button class="btn" onclick={toggle}>{auto ? '❚❚ Pause' : '▶ Train to 3,000'}</button>
        <button class="btn ghost" onclick={() => pick(mode)}>Reset</button>
      </div>
      <div class="note">Train each of the three networks for a while. Chance is 50%.</div>
    </div>
  </div>
</div>

<style>
  .modes { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
  .chip { padding: 0.3rem 0.8rem; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); font-size: 0.8rem; } .chip.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .grid { display: grid; grid-template-columns: minmax(240px, 320px) 1fr; gap: 1.4rem; align-items: start; } @media (max-width: 760px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .plot { width: 100%; height: auto; display: block; border-radius: 12px; }
  .stats { display: grid; grid-template-columns: 1fr 1.6fr; gap: 0.5rem; margin: 0.7rem 0; } .stats div { background: var(--surface-2); border-radius: 10px; padding: 0.4rem 0.7rem; } .stats strong { display: block; font-size: 1.3rem; font-variant-numeric: tabular-nums; } .l { color: var(--ink-3); font-size: 0.74rem; }
  .btns { display: flex; gap: 0.5rem; flex-wrap: wrap; } .note { color: var(--ink-2); font-size: 0.85rem; margin: 0.6rem 0; } .note2 { color: var(--ink-3); font-size: 0.8rem; font-style: italic; }
</style>
