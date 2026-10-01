<script>
  import { getContext } from 'svelte';

  // The dot product as an "agreement score". Drag the tips of the two arrows.
  const beat = getContext('beat');
  let a = $state({ x: 2, y: 1 });
  let b = $state({ x: 1, y: 2 });
  let drag = $state(null);
  let moved = new Set();

  const S = 320, C = S / 2, U = 44;      // pixels per unit
  const sx = (v) => C + v * U, sy = (v) => C - v * U;
  const dot = $derived(a.x * b.x + a.y * b.y);
  const la = $derived(Math.hypot(a.x, a.y)), lb = $derived(Math.hypot(b.x, b.y));
  const cos = $derived(la && lb ? dot / (la * lb) : 0);
  const word = $derived(dot > 0.35 ? 'pointing the same way: they AGREE' : dot < -0.35 ? 'pointing opposite ways: they DISAGREE' : 'at right angles: they are UNRELATED');

  function pos(e) {
    const r = e.currentTarget.ownerSVGElement?.getBoundingClientRect() ?? e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width * S - C) / U, y = -((e.clientY - r.top) / r.height * S - C) / U;
    const q = (v) => Math.max(-3.4, Math.min(3.4, Math.round(v * 10) / 10));
    return { x: q(x), y: q(y) };
  }
  function down(which, e) { drag = which; moved.add(which); e.currentTarget.setPointerCapture(e.pointerId); }
  function move(e) { if (!drag) return; const p = pos(e); if (drag === 'a') a = p; else b = p; if (moved.size >= 2) beat?.complete(); }
  function up() { drag = null; }
</script>

<div class="widget wide">
  <div class="widget-title">The dot product · an agreement score</div>
  <div class="grid">
    <svg viewBox="0 0 {S} {S}" class="plane" role="img" aria-label="Two arrows you can drag" onpointermove={move} onpointerup={up}>
      {#each [-3, -2, -1, 1, 2, 3] as t}
        <line x1={sx(t)} x2={sx(t)} y1="0" y2={S} stroke="var(--grid)" /><line y1={sy(t)} y2={sy(t)} x1="0" x2={S} stroke="var(--grid)" />
      {/each}
      <line x1="0" x2={S} y1={C} y2={C} stroke="var(--axis)" /><line y1="0" y2={S} x1={C} x2={C} stroke="var(--axis)" />
      <line x1={C} y1={C} x2={sx(a.x)} y2={sy(a.y)} style="stroke: var(--series-1); stroke-width: 3;" stroke-linecap="round" />
      <line x1={C} y1={C} x2={sx(b.x)} y2={sy(b.y)} style="stroke: var(--series-2); stroke-width: 3;" stroke-linecap="round" />
      <circle class="h" cx={sx(a.x)} cy={sy(a.y)} r="11" style="fill: var(--series-1); stroke: var(--surface); stroke-width: 3;" onpointerdown={(e) => down('a', e)} role="presentation" />
      <circle class="h" cx={sx(b.x)} cy={sy(b.y)} r="11" style="fill: var(--series-2); stroke: var(--surface); stroke-width: 3;" onpointerdown={(e) => down('b', e)} role="presentation" />
      <text class="axis-text" x={sx(a.x) + 14} y={sy(a.y) - 10}>a</text><text class="axis-text" x={sx(b.x) + 14} y={sy(b.y) - 10}>b</text>
    </svg>
    <div class="side">
      <div class="eq mono"><span class="ca">a = ({a.x}, {a.y})</span><br /><span class="cb">b = ({b.x}, {b.y})</span></div>
      <div class="calc2">
        <div>multiply matching numbers, then add:</div>
        <div class="mono big">({a.x} × {b.x}) + ({a.y} × {b.y}) = <strong>{dot.toFixed(2)}</strong></div>
      </div>
      <div class="verdict" class:pos={dot > 0.35} class:neg={dot < -0.35}>{word}</div>
      <div class="note">The bigger the arrows, the louder the verdict. The sign tells you whether they agree. (Angle between them: {(Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI).toFixed(0)}°.)</div>
    </div>
  </div>
</div>

<style>
  .grid { display: grid; grid-template-columns: minmax(240px, 340px) 1fr; gap: 1.4rem; align-items: center; }
  @media (max-width: 760px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .plane { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 12px; touch-action: none; }
  .h { cursor: grab; }
  .eq { margin-bottom: 0.7rem; line-height: 1.7; } .ca { color: var(--series-1); } .cb { color: var(--series-2); }
  .calc2 { background: var(--surface-2); border-radius: 10px; padding: 0.6rem 0.8rem; }
  .big { font-size: 1.05rem; margin-top: 0.2rem; }
  .verdict { margin-top: 0.7rem; padding: 0.5rem 0.8rem; border-radius: 10px; background: var(--surface-2); font-weight: 600; }
  .verdict.pos { background: var(--good-wash); } .verdict.neg { background: var(--pain-wash); }
  .note { color: var(--ink-3); font-size: 0.82rem; margin-top: 0.6rem; }
</style>
