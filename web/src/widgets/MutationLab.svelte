<script>
  import { getContext, onDestroy } from 'svelte';
  import TableView from '../components/TableView.svelte';
  import { py } from '../lib/py.svelte.js';
  import { patchSource, parseLine, loadSource } from '../lib/liveTrain.js';
  import { DATASETS, cleanList } from '../lib/datasets.js';
  import { data } from '../lib/data.svelte.js';
  import { countParams } from '../lib/transformer.js';

  // Change the model or the data, train the REAL file, and compare runs.
  const beat = getContext('beat');
  let dsKey = $state('names');
  let own = $state('');
  let nLayer = $state(1), nEmbd = $state(16), steps = $state(250), lr = $state('1e-2');
  let phase = $state('idle');
  let step = $state(0), losses = $state([]), snaps = $state([]), finals = $state([]), err = $state('');
  let runs = $state([]);
  let t0 = 0;
  onDestroy(() => { if (phase === 'running') py.stop(); });

  const docs = $derived(dsKey === 'own' ? cleanList(own) : DATASETS[dsKey].docs);
  const vocab = $derived(docs ? new Set(docs.join('')).size + 1 : 27);
  const dials = $derived(countParams({ vocab_size: vocab, n_embd: nEmbd, n_layer: nLayer, block_size: 8 }));
  const estSec = $derived(Math.round(steps * 0.11 * (dials / 4064) * (docs && docs.length < 300 ? 1 : 1)));
  const chance = $derived(Math.log(vocab));
  const smooth = $derived.by(() => { const W = 12, out = []; for (let i = 0; i < losses.length; i++) { const a = Math.max(0, i - W + 1); let s = 0; for (let j = a; j <= i; j++) s += losses[j]; out.push(s / (i - a + 1)); } return out; });
  const W = 460, H = 150, ML = 34, MR = 8, MT = 8, MB = 18;
  const X = (i) => ML + (i / Math.max(1, steps - 1)) * (W - ML - MR);
  const Y = (v) => MT + (1 - (Math.min(chance + 0.6, Math.max(1.5, v)) - 1.5) / (chance + 0.6 - 1.5)) * (H - MT - MB);
  const line = $derived(smooth.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' '));

  async function start() {
    err = '';
    if (docs && docs.length < 20) { err = 'Please give at least 20 entries (one per line).'; return; }
    step = 0; losses = []; snaps = []; finals = [];
    phase = 'running';
    const cfg = { label: `${DATASETS[dsKey].label} · ${nLayer} layer${nLayer > 1 ? 's' : ''} · width ${nEmbd} · ${steps} steps · lr ${lr}`, dials, vocab };
    try {
      let dataFile = null;
      if (docs) { dataFile = 'custom_data.txt'; await py.run(`open('${dataFile}', 'w').write(${JSON.stringify(docs.join('\n'))})`, () => {}, { session: 'live' }); }
      const src = patchSource(await loadSource(), { steps, overrides: { n_layer: nLayer, n_embd: nEmbd, lr }, dataFile });
      t0 = performance.now();
      const r = await py.run(src, (text) => parseLine(text.trimEnd(), {
        loss: (st, l) => { step = st; losses = [...losses, l]; },
        snap: (st, names) => { snaps = [...snaps, { step: st, names }]; },
        final: (n) => { finals = [...finals, n]; },
      }), { session: 'live' });
      if (phase === 'stopped') return;
      if (!r.ok) { phase = 'error'; err = r.error; return; }
      const last = losses.slice(-20);
      runs = [...runs, { ...cfg, loss: last.reduce((a, b) => a + b, 0) / Math.max(1, last.length), names: finals.slice(0, 5), secs: Math.round((performance.now() - t0) / 1000) }];
      phase = 'done';
      beat?.complete();
    } catch (e) { phase = 'error'; err = String(e.message ?? e); }
  }
  function stop() { py.stop(); phase = 'stopped'; }
  const lastSnap = $derived(snaps[snaps.length - 1]);
</script>

<div class="widget wide">
  <div class="widget-title">The mutation lab · change something, train the real file, compare</div>
  <div class="cfg">
    <label>Data <select bind:value={dsKey} disabled={phase === 'running'}>{#each Object.entries(DATASETS) as [k, d]}<option value={k}>{d.label}{d.docs ? ` (${d.docs.length})` : ''}</option>{/each}</select></label>
    <label>Layers <select bind:value={nLayer} disabled={phase === 'running'}><option value={1}>1 (the file)</option><option value={2}>2</option><option value={3}>3</option></select></label>
    <label>Width (numbers per letter) <select bind:value={nEmbd} disabled={phase === 'running'}><option value={16}>16 (the file)</option><option value={32}>32</option><option value={8}>8</option></select></label>
    <label>Steps <select bind:value={steps} disabled={phase === 'running'}><option value={100}>100</option><option value={250}>250</option><option value={500}>500 (the file)</option><option value={1000}>1,000</option></select></label>
    <label>Learning rate <select bind:value={lr} disabled={phase === 'running'}><option value="1e-3">0.001</option><option value="3e-3">0.003</option><option value="1e-2">0.01 (the file)</option><option value="3e-2">0.03</option><option value="1e-1">0.1</option></select></label>
  </div>
  {#if dsKey === 'own'}
    <textarea rows="4" bind:value={own} placeholder="Paste your own list: one name, word or place per line. Letters only (punctuation and spaces are removed). 50 or more entries works best." aria-label="Your own data"></textarea>
    <div class="hint">{docs.length} usable entries.</div>
  {/if}
  <div class="bar">
    {#if phase !== 'running'}<button class="btn primary" onclick={start} disabled={py.status === 'running'}>▶ Train this model</button>{:else}<button class="btn" onclick={stop}>■ Stop</button>{/if}
    <span class="est">This model has <strong>{dials.toLocaleString()}</strong> dials (alphabet of {vocab}). {phase === 'running' ? `Step ${step} of ${steps}…` : `Roughly ${estSec < 60 ? estSec + ' seconds' : Math.round(estSec / 60) + ' minute' + (estSec >= 120 ? 's' : '')} to train.`}</span>
  </div>
  <div class="prog"><span style="width:{(step / steps) * 100}%"></span></div>
  {#if err}<div class="bad">{err}</div>{/if}

  {#if losses.length}
    <div class="grid">
      <div><svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Loss over steps">
        <line x1={ML} x2={W - MR} y1={Y(chance)} y2={Y(chance)} stroke="var(--ink-3)" opacity="0.6" /><text class="axis-text" x={W - MR} y={Y(chance) - 4} text-anchor="end">pure chance {chance.toFixed(2)}</text>
        <path d={line} fill="none" stroke="var(--series-1)" stroke-width="2.2" stroke-linejoin="round" />
      </svg>
      <TableView caption="Loss chart as a table" cols={['Step', 'Loss (smoothed)']} rows={smooth.map((v, i) => [i + 1, v.toFixed(2)]).filter((_, i, a) => i % 10 === 9 || i === a.length - 1)} /></div>
      <div><div class="h">Names it writes now (step {lastSnap ? lastSnap.step : 0})</div><div class="names">{#each lastSnap ? lastSnap.names : [] as n}<span class="mono n">{n || '·'}</span>{/each}</div></div>
    </div>
  {/if}

  {#if runs.length}
    <div class="h" style="margin-top:1rem">Your runs</div>
    <div class="tbl">
      <div class="r head"><span>Setup</span><span>Dials</span><span>Final loss</span><span>Some names it wrote</span></div>
      {#each runs as r}<div class="r"><span>{r.label}</span><span>{r.dials.toLocaleString()}</span><span><strong>{r.loss.toFixed(2)}</strong> <small>(chance {Math.log(r.vocab).toFixed(2)})</small></span><span class="mono">{r.names.join(', ')}</span></div>{/each}
    </div>
    <div class="cap">Losses are only comparable between runs on the <em>same data</em>: a different alphabet has a different chance level.</div>
  {/if}
</div>

<style>
  .cfg { display: flex; gap: 0.6rem 1.2rem; flex-wrap: wrap; margin-bottom: 0.7rem; } label { display: flex; flex-direction: column; gap: 0.2rem; color: var(--ink-3); font-size: 0.74rem; }
  select, textarea { padding: 0.35rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-size: 0.88rem; } textarea { width: 100%; font-family: var(--font-mono); margin-bottom: 0.2rem; }
  .hint { color: var(--ink-3); font-size: 0.78rem; margin-bottom: 0.5rem; } .bar { display: flex; gap: 0.9rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.5rem; } .est { color: var(--ink-2); font-size: 0.85rem; }
  .prog { height: 6px; background: var(--surface-2); border-radius: 3px; overflow: hidden; margin-bottom: 0.7rem; } .prog span { display: block; height: 100%; background: var(--accent); transition: width 0.3s; } .bad { color: var(--pain); margin-bottom: 0.5rem; }
  .grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 1rem; } @media (max-width: 760px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .chart { width: 100%; height: auto; display: block; background: var(--surface-2); border-radius: 10px; } .h { font-size: 0.78rem; color: var(--ink-3); font-weight: 600; margin-bottom: 0.3rem; } .names { display: flex; flex-wrap: wrap; gap: 0.3rem; } .n { background: var(--code-bg); padding: 0.2rem 0.55rem; border-radius: 8px; font-size: 0.9rem; }
  .tbl { display: grid; gap: 3px; } .r { display: grid; grid-template-columns: 2.2fr 0.7fr 1fr 2fr; gap: 0.6rem; padding: 0.35rem 0.6rem; background: var(--surface-2); border-radius: 8px; font-size: 0.8rem; align-items: center; } @media (max-width: 760px) { .r { grid-template-columns: 1fr 1fr; } } .r.head { background: none; color: var(--ink-3); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; } small { color: var(--ink-3); } .cap { margin-top: 0.5rem; color: var(--ink-3); font-size: 0.78rem; }
</style>
