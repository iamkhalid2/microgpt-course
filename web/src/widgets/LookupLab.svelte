<script>
  import { getContext } from 'svelte';
  import { softmax } from '../lib/stats.js';

  // Attention by hand: a question (query) is matched against each memo's label (key); the answer blends the memos'
  // contents (values) in proportion to the match.
  const beat = getContext('beat');
  const memos = [
    { name: 'memo 1', key: [2, 0], value: [9, 1] },
    { name: 'memo 2', key: [0, 2], value: [1, 8] },
    { name: 'memo 3', key: [-2, 0], value: [-6, 2] },
    { name: 'memo 4', key: [1.4, 1.4], value: [5, 5] },
  ];
  let q = $state([1.5, 0.5]);
  let moves = 0;
  const touch = () => { if (++moves >= 4) beat?.complete(); };

  const scores = $derived(memos.map((m) => m.key[0] * q[0] + m.key[1] * q[1]));
  const weights = $derived(softmax(scores));
  const out = $derived([0, 1].map((j) => memos.reduce((s, m, i) => s + weights[i] * m.value[j], 0)));
  const f = (v) => v.toFixed(2);
</script>

<div class="widget wide">
  <div class="widget-title">Attention by hand · ask a question, get a blended answer</div>
  <div class="ask">
    <div class="h">Your question (the query), two numbers:</div>
    <label>first <input type="range" min="-3" max="3" step="0.1" bind:value={q[0]} oninput={touch} aria-label="Query, first number" /> <strong>{f(q[0])}</strong></label>
    <label>second <input type="range" min="-3" max="3" step="0.1" bind:value={q[1]} oninput={touch} aria-label="Query, second number" /> <strong>{f(q[1])}</strong></label>
  </div>
  <div class="tbl">
    <div class="r head"><span></span><span>label (key)</span><span>content (value)</span><span>match = query · key</span><span>weight (after softmax)</span></div>
    {#each memos as m, i}
      <div class="r" class:top={weights[i] === Math.max(...weights)}>
        <span class="nm">{m.name}</span>
        <span class="mono">({m.key[0]}, {m.key[1]})</span>
        <span class="mono">({m.value[0]}, {m.value[1]})</span>
        <span class="mono">{f(scores[i])}</span>
        <span class="bar"><span class="fill" style="width:{weights[i] * 100}%"></span><em>{(weights[i] * 100).toFixed(0)}%</em></span>
      </div>
    {/each}
  </div>
  <div class="res">
    <div>Answer = the contents, blended by weight:</div>
    <div class="mono big">{weights.map((w, i) => `${(w).toFixed(2)}×(${memos[i].value.join(', ')})`).join(' + ')}</div>
    <div class="mono big">= <strong>({f(out[0])}, {f(out[1])})</strong></div>
  </div>
  <div class="note">Try pointing the question toward one memo's label: that memo's weight takes over, and the answer approaches its content. Point between two labels and the answer is a blend.</div>
</div>

<style>
  .ask { display: grid; gap: 0.3rem; margin-bottom: 0.8rem; }
  .ask label { display: flex; gap: 0.6rem; align-items: center; color: var(--ink-2); } .ask input { flex: 1; max-width: 22rem; accent-color: var(--accent); }
  .h { font-weight: 600; }
  .tbl { display: grid; gap: 0.3rem; }
  .r { display: grid; grid-template-columns: 4.2rem 6rem 6rem 7rem 1fr; gap: 0.6rem; align-items: center; padding: 0.3rem 0.6rem; border-radius: 9px; background: var(--surface-2); font-size: 0.85rem; }
  .r.head { background: none; color: var(--ink-3); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; }
  .r.top { background: var(--accent-wash); }
  @media (max-width: 760px) { .r { grid-template-columns: 3rem 3.6rem 3.6rem 3rem minmax(0, 1fr); gap: 0.3rem; font-size: 0.72rem; padding: 0.3rem 0.4rem; } .r > :global(*) { min-width: 0; } }
  .nm { font-weight: 600; } .bar { position: relative; height: 18px; background: var(--surface); border-radius: 4px; overflow: hidden; }
  .fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--series-1); border-radius: 0 4px 4px 0; transition: width 0.15s; }
  .bar em { position: relative; font-style: normal; font-size: 0.74rem; padding-left: 0.4rem; line-height: 18px; color: var(--ink); }
  .res { margin-top: 0.8rem; background: var(--surface-2); border-radius: 10px; padding: 0.6rem 0.8rem; } .big { font-size: 0.82rem; overflow-x: auto; white-space: nowrap; }
  .note { margin-top: 0.7rem; color: var(--ink-2); font-size: 0.85rem; }
</style>
