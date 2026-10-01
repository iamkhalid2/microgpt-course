<script>
  import { lossVC, bestVC } from '../lib/stats.js';

  // A map of the loss for every setting of the two dials. Light = low loss (good), dark = high loss (bad).
  // Hidden until revealed, so you can first find the valley by feel.
  let { c, reveal = false, trail = [], rejects = [], cur = null, onpick = null, arrows = [] } = $props();

  const S = 300, PAD = 38, GRID = 36, LO = 0.02, HI = 0.98;
  const best = $derived(bestVC(c));
  const minL = $derived(lossVC(c, best.a, best.b));
  const cells = $derived.by(() => {
    const out = [];
    for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) {
      const a = LO + ((i + 0.5) / GRID) * (HI - LO), b = LO + ((j + 0.5) / GRID) * (HI - LO);
      out.push({ i, j, L: lossVC(c, a, b) });
    }
    return out;
  });
  const X = (a) => PAD + a * (S - 2 * PAD);
  const Y = (b) => S - PAD - b * (S - 2 * PAD);
  const cw = (S - 2 * PAD) * ((HI - LO) / 1) / GRID;
  const shade = (L) => `color-mix(in oklab, var(--seq-700) ${Math.round(Math.min(1, Math.sqrt(Math.max(0, L - minL) / 1.3)) * 100)}%, var(--seq-100))`;

  function pick(e) {
    if (!onpick) return;
    const r = e.currentTarget.getBoundingClientRect();
    const a = ((e.clientX - r.left) / r.width * S - PAD) / (S - 2 * PAD);
    const b = 1 - ((e.clientY - r.top) / r.height * S - PAD) / (S - 2 * PAD);
    if (a > 0.01 && a < 0.99 && b > 0.01 && b < 0.99) onpick(a, b);
  }
  // downhill arrows: point away from the slope; length grows with steepness
  const shapes = $derived(arrows.map((r) => {
    const len = 7 + 24 * Math.min(1, Math.hypot(r.da, r.db) / 1.4);
    const ux = -r.da, uy = r.db;                 // screen y points down, so a rising b is "up" the page
    const n = Math.hypot(ux, uy) || 1;
    const dx = ux / n, dy = uy / n;
    const x0 = X(r.a), y0 = Y(r.b), x1 = x0 + dx * len, y1 = y0 + dy * len;
    const head = `${x1},${y1} ${x1 - dx * 6 + dy * 3.2},${y1 - dy * 6 - dx * 3.2} ${x1 - dx * 6 - dy * 3.2},${y1 - dy * 6 + dx * 3.2}`;
    return { x0, y0, x1, y1, head, big: r.big };
  }));
  const path = $derived(trail.map((p, i) => `${i ? 'L' : 'M'}${X(p.a).toFixed(1)},${Y(p.b).toFixed(1)}`).join(' '));
</script>

<svg viewBox="0 0 {S} {S}" class="map" role="img" aria-label="Map of loss for every setting of dial A and dial B" onclick={pick} onkeydown={null}>
  <rect x={PAD} y={PAD} width={S - 2 * PAD} height={S - 2 * PAD} style="fill: var(--surface-2);" />
  {#if reveal}
    {#each cells as k}
      <rect x={X(LO + (k.i / GRID) * (HI - LO))} y={Y(LO + ((k.j + 1) / GRID) * (HI - LO))} width={cw + 0.4} height={cw + 0.4} style="fill:{shade(k.L)};" />
    {/each}
    <circle cx={X(best.a)} cy={Y(best.b)} r="9" style="fill:none; stroke: var(--ink); stroke-width: 1.5;" />
    <text class="axis-text" x={X(best.a) + 13} y={Y(best.b) - 8}>best possible</text>
  {:else}
    {#each [0.25, 0.5, 0.75] as t}
      <line x1={X(t)} x2={X(t)} y1={PAD} y2={S - PAD} stroke="var(--grid)" />
      <line x1={PAD} x2={S - PAD} y1={Y(t)} y2={Y(t)} stroke="var(--grid)" />
    {/each}
  {/if}
  {#each [0, 0.5, 1] as t}
    <text class="axis-text" x={X(t)} y={S - PAD + 14} text-anchor="middle">{t}</text>
    <text class="axis-text" x={PAD - 6} y={Y(t) + 4} text-anchor="end">{t}</text>
  {/each}
  <text class="axis-text" x={S / 2} y={S - 4} text-anchor="middle">Dial A →</text>
  <text class="axis-text" transform="translate(8 {S / 2}) rotate(-90)" text-anchor="middle">Dial B →</text>
  {#each rejects as p}<circle cx={X(p.a)} cy={Y(p.b)} r="2" style="fill: var(--ink-3); opacity: 0.4;" />{/each}
  {#if trail.length > 1}<path d={path} fill="none" style="stroke: var(--ink); stroke-width: 1.2; opacity: 0.75;" />{/if}
  {#each shapes as r}
    <line x1={r.x0} y1={r.y0} x2={r.x1} y2={r.y1} style="stroke: var(--ink); stroke-width: {r.big ? 2.4 : 1.4};" stroke-linecap="round" />
    <polygon points={r.head} style="fill: var(--ink);" />
  {/each}
  {#if cur}<circle cx={X(cur.a)} cy={Y(cur.b)} r="6" style="fill: var(--accent); stroke: var(--surface); stroke-width: 2.5;" />{/if}
</svg>

<style>
  .map { width: 100%; max-width: 340px; height: auto; display: block; cursor: crosshair; }
</style>
