<script>
  import { onMount, getContext } from 'svelte';
  import TableView from '../components/TableView.svelte';
  import { data } from '../lib/data.svelte.js';

  // A replay of ONE real training run of microgpt.py (recorded by tools/record_replay.py). Nothing here is simulated.
  const beat = getContext('beat');
  let d = $state(null);
  let idx = $state(0);
  let playing = $state(false);
  let timer;
  let touched = false;

  onMount(() => {
    fetch(new URL('data/replay.json', document.baseURI)).then((r) => r.json()).then((j) => { d = j; });
    return () => clearInterval(timer);
  });

  const steps = $derived(d ? Object.keys(d.snaps).map(Number).sort((a, b) => a - b) : []);
  const smooth = $derived.by(() => {
    if (!d) return [];
    const W = 25, L = d.losses, out = [];
    for (let i = 0; i < L.length; i++) { const a = Math.max(0, i - W + 1); let s = 0; for (let j = a; j <= i; j++) s += L[j]; out.push(s / (i - a + 1)); }
    return out;
  });
  const chance = Math.log(27);
  const W = 640, H = 240, ML = 44, MR = 18, MT = 14, MB = 30;
  const lossAt = (s) => (s === 0 ? smooth[0] : smooth[Math.min(s, smooth.length) - 1]);
  const lo = $derived(smooth.length ? Math.floor((Math.min(...smooth) - 0.05) * 4) / 4 : 2);
  const hi = $derived(smooth.length ? Math.ceil((Math.max(...smooth, chance) + 0.05) * 4) / 4 : 4);
  const x = (s) => ML + (s / 500) * (W - ML - MR);
  const y = (v) => MT + (1 - (v - lo) / (hi - lo)) * (H - MT - MB);
  const path = $derived(smooth.map((v, i) => `${i ? 'L' : 'M'}${x(i + 1).toFixed(1)},${y(v).toFixed(1)}`).join(' '));
  const yTicks = $derived.by(() => { const t = []; for (let v = lo; v <= hi + 1e-9; v += 0.5) t.push(v); return t; });

  const step = $derived(steps[idx] ?? 0);
  const names = $derived(d ? d.snaps[String(step)] : []);
  const docSet = $derived(new Set(data.docs));
  const inData = $derived(names.filter((n) => docSet.has(n)).length);
  const cur = $derived(d ? lossAt(step) : 0);

  function touch() { if (!touched) { touched = true; beat?.complete(); } }
  function pick(i) { idx = Math.max(0, Math.min(steps.length - 1, i)); touch(); }
  function play() {
    touch();
    if (playing) { clearInterval(timer); playing = false; return; }
    if (idx >= steps.length - 1) idx = 0;
    playing = true;
    timer = setInterval(() => { if (idx >= steps.length - 1) { clearInterval(timer); playing = false; } else idx++; }, 1100);
  }
  function hoverChart(e) {
    const r = e.currentTarget.getBoundingClientRect();
    const sx = ((e.clientX - r.left) / r.width) * W;
    const s = ((sx - ML) / (W - ML - MR)) * 500;
    let best = 0; for (let i = 0; i < steps.length; i++) if (Math.abs(steps[i] - s) < Math.abs(steps[best] - s)) best = i;
    pick(best);
  }
</script>

<div class="widget wide">
  <div class="widget-title">A real training run · replay</div>
  {#if !d}
    <div class="muted">Loading the recording…</div>
  {:else}
    <div class="top">
      <button class="btn primary" onclick={play}>{playing ? '❚❚ Pause' : '▶ Play the training'}</button>
      <div class="stat"><span class="lab">training step</span><strong>{step}</strong><span class="lab">of 500</span></div>
      <div class="stat"><span class="lab">loss</span><strong>{cur.toFixed(2)}</strong><span class="lab">(lower is better)</span></div>
    </div>

    <input class="slider" type="range" min="0" max={steps.length - 1} value={idx} oninput={(e) => pick(+e.currentTarget.value)} aria-label="Training step" />

    <div class="grid">
      {#each names as n}
        <div class="name" class:known={docSet.has(n)}><span class="mono">{n || '(empty)'}</span>{#if docSet.has(n)}<small>in the data</small>{/if}</div>
      {/each}
    </div>
    <div class="cap">
      16 names the model invented at step {step}.
      {#if step > 0}{inData === 0 ? 'None of them are copied from the dataset.' : `${inData} of them happen to be real names from the dataset; the other ${16 - inData} are new.`}{:else}This is the untrained model: pure noise.{/if}
    </div>

    <svg viewBox="0 0 {W} {H}" class="chart" aria-label="Training loss over 500 steps, falling from about 3.3 to about 2.5"
      onmousemove={hoverChart} onclick={hoverChart} role="presentation">
      {#each yTicks as t}
        <line x1={ML} x2={W - MR} y1={y(t)} y2={y(t)} stroke="var(--grid)" stroke-width="1" />
        <text class="axis-text" x={ML - 8} y={y(t) + 4} text-anchor="end">{t.toFixed(1)}</text>
      {/each}
      {#each [0, 100, 200, 300, 400, 500] as s}
        <text class="axis-text" x={x(s)} y={H - 8} text-anchor="middle">{s}</text>
      {/each}
      <line x1={ML} x2={W - MR} y1={H - MB} y2={H - MB} stroke="var(--axis)" />
      <!-- pure-chance reference -->
      <line x1={ML} x2={W - MR} y1={y(chance)} y2={y(chance)} stroke="var(--ink-3)" stroke-width="1" opacity="0.55" />
      <text class="axis-text" x={W - MR} y={y(chance) - 6} text-anchor="end">pure chance: 3.30</text>
      <path d={path} fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />
      <line x1={x(step)} x2={x(step)} y1={MT} y2={H - MB} stroke="var(--ink-3)" stroke-width="1" opacity="0.6" />
      <circle cx={x(step)} cy={y(cur)} r="5" fill="var(--series-1)" stroke="var(--surface)" stroke-width="2" />
    </svg>
    <TableView caption="Loss chart as a table" cols={['Step', 'Loss (smoothed)']} rows={[0, 1, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500].map((s) => [s, lossAt(s).toFixed(2)])} />
    <div class="foot">Loss per step is noisy (each step studies one name), so the line is smoothed over 25 steps. The model has <strong>{d.num_params.toLocaleString()}</strong> numbers it can adjust. Sampling temperature {d.temperature}.</div>
  {/if}
</div>

<style>
  .top { display: flex; gap: 1.2rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.7rem; }
  .stat { display: flex; align-items: baseline; gap: 0.4rem; }
  .stat strong { font-size: 1.35rem; font-variant-numeric: tabular-nums; }
  .lab { color: var(--ink-3); font-size: 0.78rem; }
  .slider { width: 100%; accent-color: var(--accent); margin: 0.2rem 0 0.8rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(8.2rem, 1fr)); gap: 0.4rem; }
  .name { background: var(--code-bg); border-radius: 8px; padding: 0.4rem 0.6rem; font-size: 1rem; display: flex; justify-content: space-between; align-items: baseline; gap: 0.4rem; min-height: 2.2rem; }
  .name.known { box-shadow: inset 0 0 0 1px var(--series-3); }
  .name small { color: var(--ink-3); font-size: 0.66rem; }
  .cap { color: var(--ink-2); margin: 0.6rem 0 0.4rem; }
  .chart { width: 100%; height: auto; display: block; cursor: crosshair; margin-top: 0.4rem; }
  .foot { color: var(--ink-3); font-size: 0.8rem; margin-top: 0.3rem; }
</style>
