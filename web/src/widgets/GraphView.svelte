<script>
  import { layoutGraph } from '../lib/layout.js';

  // Draws any small graph: { nodes: [{id, label, data, grad}], edges: [[fromId, toId]] }.
  // Used for the "X-ray" of the cards a learner's own Python code just built.
  let { graph, title = 'X-ray: the graph your code just built' } = $props();
  const L = $derived(layoutGraph(graph));
  const fmt = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(1) : +v.toFixed(3) + '') : String(v));
</script>

<div class="xray">
  <div class="t">{title} <span class="muted">({graph.nodes.length} cards)</span></div>
  <div class="scroll">
    <svg viewBox="0 0 {L.width} {L.height}" style="min-width:{Math.min(L.width, 900)}px" role="img" aria-label="Computation graph">
      <defs><marker id="xh" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill: var(--ink-3);" /></marker></defs>
      {#each graph.edges as [f, t]}
        <line x1={L.pos[f][0] + L.w / 2} y1={L.pos[f][1]} x2={L.pos[t][0] - L.w / 2} y2={L.pos[t][1]} marker-end="url(#xh)" style="stroke: var(--line-strong); stroke-width: 1.4;" />
      {/each}
      {#each graph.nodes as n}
        <rect x={L.pos[n.id][0] - L.w / 2} y={L.pos[n.id][1] - L.h / 2} width={L.w} height={L.h} rx="9" style="fill: var(--surface); stroke: var(--line-strong); stroke-width: 1.4;" />
        <text class="nl" x={L.pos[n.id][0]} y={L.pos[n.id][1] - 5} text-anchor="middle">{n.label || 'card'}</text>
        <text class="nv" x={L.pos[n.id][0]} y={L.pos[n.id][1] + 9} text-anchor="middle">{fmt(n.data)}</text>
        {#if n.grad}<text class="ng" x={L.pos[n.id][0]} y={L.pos[n.id][1] + L.h / 2 + 12} text-anchor="middle">slope {fmt(n.grad)}</text>{/if}
      {/each}
    </svg>
  </div>
</div>

<style>
  .xray { margin-top: 0.8rem; background: var(--surface-2); border-radius: 12px; padding: 0.6rem 0.8rem; }
  .t { font-size: 0.78rem; color: var(--ink-2); font-weight: 600; margin-bottom: 0.3rem; }
  .scroll { overflow-x: auto; }
  svg { width: 100%; height: auto; display: block; }
  .nl { font-family: var(--font-mono); font-size: 11.5px; fill: var(--ink-2); }
  .nv { font-family: var(--font-mono); font-size: 12px; fill: var(--ink); font-weight: 700; }
  .ng { font-family: var(--font-ui); font-size: 10.5px; fill: var(--pain); font-weight: 700; }
</style>
