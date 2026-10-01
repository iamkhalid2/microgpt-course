<script>
  import { getContext } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { encode, tokenLabel, sum } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // "How likely did the model think this name was?" = multiply the probability of every step.
  let { need = 2 } = $props();
  const beat = getContext('beat');
  let name = $state('emma');
  let model = $state('bigram');
  let smooth = $state(false);
  let changes = 0;
  const MODELS = { chance: 'Pure chance', unigram: 'Letter frequencies', bigram: 'Bigram table' };

  const ids = $derived(data.vocab ? encode(name.trim().toLowerCase(), data.vocab) : null);
  const uniTot = $derived(sum(data.uni));
  const rowTot = $derived(data.bi.map((r) => sum(r)));
  const alpha = $derived(smooth ? 1 : 0);

  function pOf(m, prev, nxt) {
    const V = data.vocab.V;
    if (m === 'chance') return 1 / V;
    if (m === 'unigram') return data.uni[nxt] / uniTot;
    return (data.bi[prev][nxt] + alpha) / (rowTot[prev] + alpha * V);
  }
  const steps = $derived.by(() => {
    if (!ids) return [];
    const out = [];
    for (let i = 0; i < ids.length - 1; i++) out.push({ prev: ids[i], nxt: ids[i + 1], p: pOf(model, ids[i], ids[i + 1]) });
    return out;
  });
  const product = $derived(steps.reduce((a, s) => a * s.p, 1));
  const all = $derived(ids ? Object.keys(MODELS).map((m) => ({ m, prod: ids.slice(0, -1).reduce((a, _, i) => a * pOf(m, ids[i], ids[i + 1]), 1) })) : []);
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const sci = (v) => {
    if (v === 0) return '0';
    if (v >= 0.001) return v.toFixed(4);
    const [m, e] = v.toExponential(1).split('e');
    return `${m} × 10${[...String(parseInt(e))].map((c) => SUP[c]).join('')}`;
  };
  const oneIn = (v) => (v === 0 ? 'impossible' : v >= 0.5 ? 'likely' : `1 chance in ${Math.round(1 / v).toLocaleString()}`);
  function touch() { if (++changes >= need) beat?.complete(); }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">How likely did the model think this name was?</div>
    <div class="ctl">
      <label>Name <input bind:value={name} oninput={touch} spellcheck="false" maxlength="14" /></label>
      <span class="grp" role="group" aria-label="Model">
        {#each Object.entries(MODELS) as [k, label]}<button class:on={model === k} onclick={() => { model = k; touch(); }}>{label}</button>{/each}
      </span>
      {#if model === 'bigram'}<label class="chk"><input type="checkbox" bind:checked={smooth} /> never say never (+1 to every pair)</label>{/if}
    </div>

    {#if !ids}
      <div class="bad">That has a character outside the model's alphabet (only a–z). Try a lowercase name.</div>
    {:else}
      <div class="chain">
        {#each steps as s}
          <div class="st" class:zero={s.p === 0}>
            <div class="pair mono"><span>{tokenLabel(s.prev, data.vocab)}</span><span class="arr">→</span><span>{tokenLabel(s.nxt, data.vocab)}</span></div>
            <div class="mini"><span style="height:{Math.max(3, Math.sqrt(s.p) * 100)}%"></span></div>
            <div class="pv">{s.p === 0 ? '0' : s.p >= 0.01 ? s.p.toFixed(3) : s.p.toFixed(4)}</div>
          </div>
          <div class="x" aria-hidden="true">×</div>
        {/each}
        <div class="eq">=</div>
        <div class="res" class:zero={product === 0}><div class="v">{sci(product)}</div><div class="o">{oneIn(product)}</div></div>
      </div>
      <div class="cmp">
        {#each all as a}
          <div class="crow" class:cur={a.m === model}><span>{MODELS[a.m]}</span><span class="cv">{oneIn(a.prod)}</span></div>
        {/each}
      </div>
      {#if model === 'bigram' && product === 0}
        <div class="warn">The bigram table has never seen one of these pairs, so it says <strong>impossible</strong>: probability 0. That's a dangerous thing for a model to claim. Tick “never say never” to see the fix.</div>
      {/if}
    {/if}
  </div>
</DataGate>

<style>
  .ctl { display: flex; gap: 0.8rem 1.2rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem; }
  label { color: var(--ink-2); }
  .ctl input:not([type]) { font-family: var(--font-mono); font-size: 1rem; padding: 0.35rem 0.6rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); width: 9rem; margin-left: 0.3rem; }
  .grp { display: inline-flex; border: 1px solid var(--line-strong); border-radius: 999px; overflow: hidden; }
  .grp button { border: 0; background: var(--surface); color: var(--ink-2); padding: 0.35rem 0.8rem; }
  .grp button.on { background: var(--accent); color: var(--on-accent); }
  .chk { font-size: 0.82rem; display: flex; gap: 0.35rem; align-items: center; }
  .chain { display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.3rem; margin: 0.4rem 0 1rem; }
  .st { display: flex; flex-direction: column; align-items: center; gap: 0.25rem; background: var(--code-bg); border-radius: 10px; padding: 0.4rem 0.55rem 0.3rem; min-width: 4.3rem; }
  .st.zero { background: var(--pain-wash); box-shadow: inset 0 0 0 1px var(--pain); }
  .pair { font-size: 1.05rem; display: flex; gap: 0.25rem; }
  .arr { color: var(--ink-3); }
  .mini { width: 100%; height: 30px; display: flex; align-items: flex-end; }
  .mini span { width: 100%; background: var(--series-1); border-radius: 3px 3px 0 0; }
  .pv { font-size: 0.78rem; font-variant-numeric: tabular-nums; color: var(--ink-2); }
  .x, .eq { color: var(--ink-3); font-size: 1.2rem; }
  .res { background: var(--accent-wash); border-radius: 12px; padding: 0.5rem 0.9rem; }
  .res.zero { background: var(--pain-wash); }
  .v { font-size: 1.25rem; font-weight: 700; }
  .o { font-size: 0.78rem; color: var(--ink-2); }
  .cmp { display: grid; gap: 2px; max-width: 26rem; margin-top: 0.5rem; }
  .crow { display: flex; justify-content: space-between; gap: 1rem; padding: 0.25rem 0.6rem; border-radius: 8px; font-size: 0.85rem; color: var(--ink-2); }
  .crow.cur { background: var(--surface-2); color: var(--ink); font-weight: 600; }
  .bad, .warn { margin-top: 0.4rem; color: var(--pain); }
  .warn { background: var(--pain-wash); padding: 0.6rem 0.8rem; border-radius: 10px; color: var(--ink); margin-top: 0.8rem; }
</style>
