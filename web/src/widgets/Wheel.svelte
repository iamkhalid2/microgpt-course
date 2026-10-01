<script>
  import { getContext, onDestroy } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { tokenLabel, sum } from '../lib/stats.js';
  import { makeRng, sampleIndex } from '../lib/rng.js';
  import DataGate from './DataGate.svelte';

  // A wheel whose wedges are as wide as each letter is common. Spin it, and you've sampled from a probability distribution.
  const beat = getContext('beat');
  const rng = makeRng((Date.now() >>> 0) % 2147483647);
  const C = 150, R = 138;
  let rot = $state(0);
  let spinning = $state(false);
  let drawn = $state([]);
  let landed = $state(null);
  let timer;
  let spins = 0;
  onDestroy(() => clearTimeout(timer));

  const total = $derived(sum(data.uni));
  const wedges = $derived.by(() => {
    let a = 0;
    return data.uni.map((c, id) => {
      const th = (c / total) * 360; const w = { id, a0: a, a1: a + th, th }; a += th; return w;
    });
  });
  const pt = (deg, r = R) => { const t = ((deg - 90) * Math.PI) / 180; return [C + r * Math.cos(t), C + r * Math.sin(t)]; };
  const d = (w) => { const [x0, y0] = pt(w.a0), [x1, y1] = pt(w.a1); return `M${C},${C} L${x0.toFixed(2)},${y0.toFixed(2)} A${R},${R} 0 ${w.th > 180 ? 1 : 0} 1 ${x1.toFixed(2)},${y1.toFixed(2)} Z`; };

  const done = $derived(drawn.length > 0 && drawn[drawn.length - 1] === data.vocab.BOS);
  const name = $derived(drawn.filter((i) => i !== data.vocab.BOS).map((i) => data.vocab.uchars[i]).join(''));

  function spin() {
    if (spinning) return;
    if (done || drawn.length >= 14) { drawn = []; landed = null; }
    spinning = true;
    const id = sampleIndex(data.uni, rng);
    const w = wedges[id];
    const at = w.a0 + w.th * (0.15 + 0.7 * rng());      // a random spot inside the chosen wedge
    const target = -at;
    rot = rot + 1080 + ((((target - rot) % 360) + 360) % 360);
    timer = setTimeout(() => {
      drawn = [...drawn, id]; landed = id; spinning = false;
      if (++spins >= 3) beat?.complete();
    }, 1500);
  }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">The letter wheel</div>
    <div class="wrap">
      <div class="wheelbox">
        <svg viewBox="0 0 300 300" class="wheel" role="img" aria-label="Wheel with a wedge for each letter, sized by how common it is">
          <g style="transform: rotate({rot}deg); transform-origin: {C}px {C}px; transition: transform {spinning ? 1.4 : 0}s cubic-bezier(0.15, 0.8, 0.2, 1);">
            {#each wedges as w}
              <path d={d(w)} style="fill: {w.id === data.vocab.BOS ? 'var(--series-2)' : w.id % 2 ? 'var(--seq-300)' : 'var(--seq-500)'}; stroke: var(--surface); stroke-width: 1.5;" />
              {#if w.th > 9}
                {@const [tx, ty] = pt((w.a0 + w.a1) / 2, R * 0.78)}
                <text x={tx} y={ty + 4} text-anchor="middle" class="wl" transform="rotate({(w.a0 + w.a1) / 2} {tx} {ty})">{tokenLabel(w.id, data.vocab)}</text>
              {/if}
            {/each}
          </g>
          <path d="M150,2 L140,-14 L160,-14 Z" transform="translate(0 16)" style="fill: var(--ink);" />
          <circle cx={C} cy={C} r="7" style="fill: var(--surface); stroke: var(--line-strong);" />
        </svg>
      </div>
      <div class="side">
        <p class="cap">Wider wedge = more common letter. The orange wedge is ⏎ ("the name ends here"): about 14% of the wheel, because <em>every</em> name ends, and names are short.</p>
        <button class="btn primary" onclick={spin} disabled={spinning}>{done ? 'Start a new name' : spinning ? 'Spinning…' : drawn.length ? 'Spin again' : 'Spin the wheel'}</button>
        <div class="name mono" aria-live="polite">
          {#each drawn as t}<span class:stop={t === data.vocab.BOS}>{tokenLabel(t, data.vocab)}</span>{/each}
          {#if !drawn.length}<span class="muted">…</span>{/if}
        </div>
        {#if done}<div class="says">A "name": <strong class="mono">{name || '(nothing)'}</strong>. Spin on to make another.</div>{/if}
        <div class="py mono">token = random.choices(range(27), weights=counts)[0]</div>
      </div>
    </div>
  </div>
</DataGate>

<style>
  .wrap { display: grid; grid-template-columns: minmax(220px, 320px) 1fr; gap: 1.6rem; align-items: center; }
  @media (max-width: 700px) { .wrap { grid-template-columns: minmax(0, 1fr); } }
  .wrap > * { min-width: 0; }
  .wheelbox { padding-top: 1rem; }
  .wheel { width: 100%; max-width: 340px; height: auto; overflow: visible; display: block; margin: 0 auto; }
  .wl { font-family: var(--font-mono); font-size: 13px; fill: #fff; font-weight: 600; }
  .cap { color: var(--ink-2); margin-top: 0; }
  .name { font-size: 1.7rem; margin: 0.9rem 0 0.3rem; letter-spacing: 0.06em; min-height: 2.2rem; display: flex; gap: 0.05em; }
  .stop { color: var(--series-2); }
  .says { color: var(--ink-2); }
  .py { margin-top: 0.9rem; font-size: 0.76rem; color: var(--ink-2); background: var(--code-bg); padding: 0.4rem 0.6rem; border-radius: 8px; overflow-x: auto; white-space: nowrap; }
</style>
