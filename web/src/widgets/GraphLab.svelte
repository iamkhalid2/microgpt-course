<script>
  import { getContext } from 'svelte';
  import { PRESETS, nodesOf, backward, wiggle } from '../lib/graph.js';

  // Watch the chain rule run backwards through a computation graph, one node at a time.
  const beat = getContext('beat');
  let { presets = ['chain', 'twice', 'surprise'] } = $props();
  let pick = $state(presets[0]);
  let vals = $state({});
  let step = $state(0);          // 0 = forward only; k = k backward steps shown
  let seen = new Set();

  const preset = $derived(PRESETS[pick]);
  const nodes = $derived(nodesOf(preset));
  const inputs = $derived(Object.fromEntries(preset.edit.map((id) => [id, vals[pick + id] ?? nodes.find((n) => n.id === id).value])));
  const bw = $derived(backward(nodes, inputs));
  const maxStep = $derived(bw.trace.length - 1);
  const g = $derived(bw.trace[Math.min(step, maxStep)].grad);
  const cur = $derived(step > 0 ? bw.trace[Math.min(step, maxStep)] : null);
  const fmt = (v) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(2) : +v.toFixed(4) + '');
  const P = (id) => preset.pos[id];
  const byId = $derived(Object.fromEntries(nodes.map((n) => [n.id, n])));

  function setIn(id, v) { vals[pick + id] = v; step = 0; }
  function next() { step = Math.min(maxStep, step + 1); if (step >= maxStep) { seen.add(pick); if (seen.size >= 1) beat?.complete(); } }
  function choose(p) { pick = p; step = 0; }
  const W = 600, H = 290, NW = 112, NH = 52;
  const edges = $derived(nodes.flatMap((n) => (n.inputs ?? []).map((i, j) => ({ from: i, to: n.id, local: bw.loc[n.id]?.[j] }))));
  const active = (e) => cur && e.to === cur.node;
  const done = (id) => step > 0 && bw.trace.slice(1, step + 1).some((t) => t.node === id);
  const sentence = $derived.by(() => {
    if (step === 0) return 'Forward pass done: every box holds its value. Now press “Next step” to send slopes backward, starting at the loss, whose slope with respect to itself is 1.';
    const t = bw.trace[Math.min(step, maxStep)];
    const n = byId[t.node];
    const parts = n.inputs.map((inp, j) => `${inp}: local slope ${fmt(t.local[j])} × ${fmt(t.upstream)} = ${fmt(t.local[j] * t.upstream)}`);
    return `${n.label}. Slope of the loss w.r.t. ${n.id} is ${fmt(t.upstream)}. Pass it back: ${parts.join('; ')}. (Added to what each already has.)`;
  });
</script>

<div class="widget wide">
  <div class="widget-title">The chain rule, running backwards</div>
  <div class="top">
    {#each presets as p}<button class="chip" class:on={pick === p} onclick={() => choose(p)}>{PRESETS[p].title}</button>{/each}
  </div>

  <div class="gscroll"><svg viewBox="0 0 {W} {H}" class="g" role="img" aria-label="Computation graph with values and slopes">
    <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill: var(--ink-3);" /></marker></defs>
    {#each edges as e}
      {@const a = P(e.from)}
      {@const b = P(e.to)}
      <line x1={a[0] + NW / 2} y1={a[1]} x2={b[0] - NW / 2} y2={b[1]} marker-end="url(#ah)" style="stroke: {active(e) ? 'var(--pain)' : 'var(--line-strong)'}; stroke-width: {active(e) ? 2.6 : 1.6};" />
      {#if active(e)}<text class="el" x={(a[0] + b[0]) / 2} y={(a[1] + b[1]) / 2 - 8} text-anchor="middle">× {fmt(e.local)}</text>{/if}
    {/each}
    {#each nodes as n}
      {@const p = P(n.id)}
      <g>
        <rect x={p[0] - NW / 2} y={p[1] - NH / 2} width={NW} height={NH} rx="10" style="fill: var(--surface); stroke: {cur && cur.node === n.id ? 'var(--pain)' : done(n.id) ? 'var(--series-3)' : 'var(--line-strong)'}; stroke-width: {cur && cur.node === n.id ? 2.6 : 1.6};" />
        <text class="nl" x={p[0]} y={p[1] - 7} text-anchor="middle">{n.op === 'input' || n.op === 'const' ? n.id : n.label ?? n.id}</text>
        <text class="nv" x={p[0]} y={p[1] + 10} text-anchor="middle">= {fmt(bw.val[n.id])}</text>
        {#if step > 0 && n.op !== 'const' && (g[n.id] !== 0 || n.id === nodes[nodes.length - 1].id)}
          <text class="ng" x={p[0]} y={p[1] + NH / 2 + 15} text-anchor="middle">slope {fmt(g[n.id])}</text>
        {/if}
      </g>
    {/each}
  </svg></div>

  <div class="ctl">
    {#each preset.edit as id}
      <label class="in">{id} = <input type="number" step="any" value={inputs[id]} oninput={(e) => setIn(id, +e.currentTarget.value)} /></label>
    {/each}
    <span class="sp"></span>
    <button class="btn" onclick={() => (step = 0)} disabled={step === 0}>Reset</button>
    <button class="btn primary" onclick={next} disabled={step >= maxStep}>Next step ▶</button>
  </div>
  <div class="says">{sentence}</div>

  {#if step >= maxStep}
    <table class="t">
      <thead><tr><td>input</td><td>slope found by going backwards</td><td>slope found by the wiggle test</td></tr></thead>
      <tbody>
        {#each preset.edit as id}
          <tr><td class="mono">{id}</td><td><strong>{fmt(bw.grad[id])}</strong></td><td>{fmt(wiggle(nodes, inputs, id, 1e-6))}</td></tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>

<style>
  .top { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.6rem; }
  .chip { padding: 0.3rem 0.8rem; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); font-size: 0.8rem; }
  .chip.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .gscroll { overflow-x: auto; border-radius: 12px; }
  .g { min-width: 520px; width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 12px; }
  .nl { font-family: var(--font-mono); font-size: 12px; fill: var(--ink); }
  .nv { font-family: var(--font-mono); font-size: 12px; fill: var(--ink-2); font-weight: 700; }
  .ng { font-family: var(--font-ui); font-size: 11.5px; fill: var(--pain); font-weight: 700; }
  .el { font-family: var(--font-ui); font-size: 11.5px; fill: var(--pain); font-weight: 700; }
  .ctl { display: flex; gap: 0.7rem; align-items: center; margin-top: 0.7rem; flex-wrap: wrap; }
  .sp { flex: 1; }
  .in { color: var(--ink-2); font-family: var(--font-mono); }
  .in input { width: 5.2rem; padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-family: var(--font-mono); }
  .says { margin-top: 0.7rem; padding: 0.6rem 0.8rem; background: var(--accent-wash); border-radius: 10px; min-height: 3.2rem; }
  .t { width: 100%; border-collapse: collapse; margin-top: 0.7rem; font-variant-numeric: tabular-nums; }
  .t td { padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--line); }
  .t thead td { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
