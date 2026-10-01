<script>
  import { getContext } from 'svelte';
  import { countParams } from '../lib/gpt.js';

  // Where do the dials live? Change the design and watch the count. At the defaults you get microgpt's 4,064.
  const beat = getContext('beat');
  let vocab = $state(27), embd = $state(16), head = $state(4), layers = $state(1), block = $state(8);
  let acts = 0;
  const touch = () => { if (++acts >= 4) beat?.complete(); };
  const heads = $derived([1, 2, 4, 8, 12, 16, 32].filter((h) => embd % h === 0));
  $effect(() => { if (!heads.includes(head)) head = heads[heads.length > 2 ? 2 : heads.length - 1]; });
  const rows = $derived([
    { name: 'letter coordinates (wte)', n: vocab * embd, note: `${vocab} symbols × ${embd} numbers` },
    { name: 'position coordinates (wpe)', n: block * embd, note: `${block} positions × ${embd} numbers` },
    { name: 'output head (lm_head)', n: vocab * embd, note: `${vocab} scores × ${embd} numbers` },
    { name: 'attention: query, key, value, output (4 tables per layer)', n: layers * 4 * embd * embd, note: `${layers} × 4 × ${embd} × ${embd}` },
    { name: 'MLP: expand and shrink (2 tables per layer)', n: layers * 8 * embd * embd, note: `${layers} × (4·${embd} × ${embd} + ${embd} × 4·${embd})` },
  ]);
  const total = $derived(countParams({ vocab_size: vocab, n_embd: embd, n_layer: layers, block_size: block }));
  const mx = $derived(Math.max(...rows.map((r) => r.n), 1));
  const fmt = (n) => n.toLocaleString();
  function preset(v, e, h, l, b) { vocab = v; embd = e; head = h; layers = l; block = b; touch(); }
</script>

<div class="widget wide">
  <div class="widget-title">Count the dials · design your own model</div>
  <div class="pre">
    <button class="chip" onclick={() => preset(27, 16, 4, 1, 8)}>microgpt (the real file)</button>
    <button class="chip" onclick={() => preset(27, 32, 4, 1, 8)}>twice as wide</button>
    <button class="chip" onclick={() => preset(27, 16, 4, 4, 8)}>four layers deep</button>
    <button class="chip" onclick={() => preset(50257, 768, 12, 12, 1024)}>GPT-2-small-sized (this recipe)</button>
  </div>
  <div class="ctl">
    <label>Symbols (vocab) <input type="number" min="2" max="100000" bind:value={vocab} oninput={touch} /></label>
    <label>Numbers per letter (n_embd)
      <select bind:value={embd} onchange={touch}>{#each [4, 8, 16, 32, 64, 128, 256, 768] as e}<option value={e}>{e}</option>{/each}</select></label>
    <label>Heads <select bind:value={head} onchange={touch}>{#each heads as h}<option value={h}>{h}</option>{/each}</select></label>
    <label>Layers <input type="number" min="1" max="96" bind:value={layers} oninput={touch} /></label>
    <label>Longest name (block_size) <input type="number" min="1" max="8192" bind:value={block} oninput={touch} /></label>
  </div>
  {#each rows as r}
    <div class="row"><span class="nm">{r.name}<small>{r.note}</small></span><span class="bar"><span class="fill" style="width:{(r.n / mx) * 100}%"></span></span><span class="v">{fmt(r.n)}</span></div>
  {/each}
  <div class="tot"><span>Total dials</span><strong>{fmt(total)}</strong>{#if total === 4064}<em class="ok">= the 4,064 that microgpt.py prints</em>{/if}</div>
  <div class="note">Notice: <strong>number of heads doesn't change the count</strong> (Chapter 11: heads share out the same numbers). Doubling the width quadruples the layer tables (n_embd is squared). {total > 1e8 ? 'This recipe at that size would hold about ' + (total / 1e6).toFixed(0) + ' million dials. (The real GPT-2 small has 124 million: it reuses its letter table as the output head, which saves one big table.)' : ''}</div>
</div>

<style>
  .pre { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.7rem; } .chip { padding: 0.3rem 0.8rem; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); font-size: 0.8rem; } .chip:hover { border-color: var(--accent); }
  .ctl { display: flex; gap: 0.8rem 1.2rem; flex-wrap: wrap; margin-bottom: 0.9rem; } .ctl label { display: flex; flex-direction: column; gap: 0.2rem; color: var(--ink-3); font-size: 0.74rem; }
  .ctl input, .ctl select { padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); width: 8rem; font-size: 0.9rem; }
  .row { display: grid; grid-template-columns: 22rem 1fr 6.5rem; gap: 0.7rem; align-items: center; margin-bottom: 5px; } @media (max-width: 760px) { .row { grid-template-columns: 1fr 5.5rem; } .row .bar { grid-column: 1 / -1; } }
  .nm { font-size: 0.85rem; } .nm small { display: block; color: var(--ink-3); font-size: 0.72rem; } .bar { height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; } .fill { display: block; height: 100%; background: var(--series-1); border-radius: 0 4px 4px 0; } .v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .tot { display: flex; gap: 0.8rem; align-items: baseline; margin-top: 0.7rem; padding: 0.6rem 0.9rem; background: var(--surface-2); border-radius: 12px; flex-wrap: wrap; } .tot strong { font-size: 1.7rem; font-variant-numeric: tabular-nums; } .ok { color: var(--good); font-style: normal; font-weight: 600; font-size: 0.85rem; }
  .note { margin-top: 0.7rem; color: var(--ink-2); font-size: 0.85rem; }
</style>
