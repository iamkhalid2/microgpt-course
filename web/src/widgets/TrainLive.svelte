<script>
  import { getContext, onDestroy } from 'svelte';
  import TableView from '../components/TableView.svelte';
  import { py } from '../lib/py.svelte.js';
  import { patchSource, parseLine, loadSource } from '../lib/liveTrain.js';

  // Runs the ACTUAL microgpt.py in your browser (Pyodide). The only changes are a few print lines that report the loss
  // and sample some names while it trains. Nothing is simulated.
  let { defaultSteps = 100 } = $props();
  const beat = getContext('beat');
  let N = $state(defaultSteps);
  let phase = $state('idle');          // idle | running | done | stopped | error
  let step = $state(0);
  let losses = $state([]);
  let snaps = $state([]);              // { step, names[] }
  let finals = $state([]);
  let info = $state([]);
  let t0 = 0;
  let rate = $state(0);                // steps per second
  let err = $state('');
  onDestroy(() => { if (phase === 'running') py.stop(); });

  function parse(line) {
    parseLine(line, {
      loss: (st, l) => { step = st; losses = [...losses, l]; const el = (performance.now() - t0) / 1000; rate = step / Math.max(el, 0.001); if (step >= 50 || step >= N) beat?.complete(); },
      snap: (st, names) => { snaps = [...snaps, { step: st, names }]; },
      final: (n) => { finals = [...finals, n]; },
      info: (l) => { info = [...info, l]; },
    });
  }

  async function start() {
    err = ''; step = 0; losses = []; snaps = []; finals = []; info = []; rate = 0;
    phase = 'running';
    let src;
    try { src = patchSource(await loadSource(), { steps: N }); } catch (e) { phase = 'error'; err = String(e.message ?? e); return; }
    t0 = performance.now();
    const r = await py.run(src, (text) => parse(text.trimEnd()), { session: 'live' });
    if (phase === 'stopped') return;
    if (!r.ok) { phase = 'error'; err = r.error; return; }
    phase = 'done';
  }
  function stop() { py.stop(); phase = 'stopped'; }

  const smooth = $derived.by(() => { const W = 12, out = []; for (let i = 0; i < losses.length; i++) { const a = Math.max(0, i - W + 1); let s = 0; for (let j = a; j <= i; j++) s += losses[j]; out.push(s / (i - a + 1)); } return out; });
  const W = 520, H = 190, ML = 36, MR = 10, MT = 10, MB = 22;
  const X = (i) => ML + (i / Math.max(1, N - 1)) * (W - ML - MR);
  const Y = (v) => MT + (1 - (Math.min(4.2, Math.max(2, v)) - 2) / 2.2) * (H - MT - MB);
  const line = $derived(smooth.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' '));
  const eta = $derived(rate > 0 ? Math.max(0, Math.round((N - step) / rate)) : null);
  const last = $derived(snaps[snaps.length - 1]);
</script>

<div class="widget wide">
  <div class="widget-title">Train the real microgpt.py · in your browser</div>
  <div class="ctl">
    <label>Training steps <select bind:value={N} disabled={phase === 'running'}><option value={100}>100 steps (about 10 seconds)</option><option value={250}>250 (about 30 seconds)</option><option value={500}>500 (the file's own default, about a minute)</option><option value={1000}>1,000 (about 2 minutes)</option><option value={3000}>3,000 (about 6 minutes, the run behind the other chapters)</option></select></label>
    {#if phase !== 'running'}<button class="btn primary" onclick={start} disabled={py.status === 'running'}>▶ Start training</button>{:else}<button class="btn" onclick={stop}>■ Stop</button>{/if}
    {#if phase === 'running'}<span class="st">{py.status === 'loading' ? 'Starting Python (first run downloads it)…' : step === 0 ? 'Loading the names and building 4,064 dials…' : `step ${step} of ${N}${eta !== null ? ` · about ${eta}s left` : ''}`}</span>{/if}
    {#if phase === 'stopped'}<span class="st">Stopped.</span>{/if}
  </div>
  {#if info.length}<div class="info mono">{info.join('   ·   ')}</div>{/if}
  <div class="prog"><span style="width:{(step / N) * 100}%"></span></div>

  <div class="grid">
    <div>
      <div class="h">Loss, step by step (smoothed)</div>
      <svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Training loss over steps">
        {#each [2.5, 3, 3.5, 4] as t}<line x1={ML} x2={W - MR} y1={Y(t)} y2={Y(t)} stroke="var(--grid)" /><text class="axis-text" x={ML - 5} y={Y(t) + 4} text-anchor="end">{t}</text>{/each}
        <line x1={ML} x2={W - MR} y1={Y(3.296)} y2={Y(3.296)} stroke="var(--ink-3)" opacity="0.6" /><text class="axis-text" x={W - MR} y={Y(3.296) - 4} text-anchor="end">pure chance 3.30</text>
        <path d={line} fill="none" stroke="var(--series-1)" stroke-width="2.2" stroke-linejoin="round" />
        <text class="axis-text" x={ML} y={H - 6}>step 1</text><text class="axis-text" x={W - MR} y={H - 6} text-anchor="end">step {N}</text>
      </svg>
      {#if smooth.length}<TableView caption="Loss chart as a table" cols={['Step', 'Loss (smoothed)']} rows={smooth.map((v, i) => [i + 1, v.toFixed(2)]).filter((_, i, a) => i % 10 === 9 || i === a.length - 1)} />{/if}
    </div>
    <div>
      <div class="h">8 names the model writes right now (at step {last ? last.step : 0})</div>
      <div class="names">{#each last ? last.names : [] as n}<span class="mono n">{n || '·'}</span>{/each}{#if !last}<span class="muted">waiting for the model…</span>{/if}</div>
      {#if snaps.length > 1}
        <details class="hist"><summary>How the names changed</summary>
          {#each snaps as s}<div class="hr"><span class="sn">step {s.step}</span><span class="mono">{s.names.slice(0, 5).join('  ')}</span></div>{/each}
        </details>
      {/if}
    </div>
  </div>
  {#if phase === 'done'}<div class="done">Done. Final loss (smoothed): <strong>{smooth[smooth.length - 1].toFixed(2)}</strong>. {finals.length ? 'The file then wrote 20 names: ' : ''}<span class="mono">{finals.slice(0, 8).join(', ')}</span></div>{/if}
  {#if phase === 'error'}<div class="bad">Something went wrong: {err}</div>{/if}
  <div class="cap">This runs the real file from this repository (with print lines added to report progress). Python in a browser is slower than on your computer, and that is the only difference.</div>
</div>

<style>
  .ctl { display: flex; gap: 0.8rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.6rem; } label { max-width: 100%; flex-wrap: wrap; color: var(--ink-2); display: flex; gap: 0.5rem; align-items: center; font-size: 0.85rem; } select { max-width: 100%; padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); } .st { color: var(--ink-2); font-size: 0.85rem; }
  .info { font-size: 0.74rem; color: var(--ink-3); margin-bottom: 0.4rem; } .prog { height: 6px; background: var(--surface-2); border-radius: 3px; overflow: hidden; margin-bottom: 0.8rem; } .prog span { display: block; height: 100%; background: var(--accent); transition: width 0.3s; }
  .grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 1.2rem; } @media (max-width: 800px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .h { font-size: 0.78rem; color: var(--ink-3); font-weight: 600; margin-bottom: 0.3rem; } .chart { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 10px; }
  .names { display: flex; flex-wrap: wrap; gap: 0.35rem; } .n { background: var(--code-bg); padding: 0.25rem 0.6rem; border-radius: 8px; font-size: 0.95rem; } .hist { margin-top: 0.7rem; font-size: 0.8rem; } .hist summary { cursor: pointer; color: var(--ink-2); } .hr { display: flex; gap: 0.7rem; padding: 0.15rem 0; } .sn { color: var(--ink-3); min-width: 4.2rem; }
  .done { margin-top: 0.8rem; padding: 0.6rem 0.9rem; background: var(--good-wash); border-radius: 10px; } .bad { margin-top: 0.8rem; color: var(--pain); } .cap { margin-top: 0.7rem; color: var(--ink-3); font-size: 0.78rem; }
</style>
