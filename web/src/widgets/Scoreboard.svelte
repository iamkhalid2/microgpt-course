<script>
  import { getContext, onMount } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import * as S from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // Every model so far, graded by the same ruler: average surprise per symbol. Lower is better.
  const beat = getContext('beat');
  let held = $state(false);
  let alpha = $state(0);
  let moved = 0;
  let long = $state(null);
  onMount(() => { fetch(new URL('data/long_run.json', document.baseURI)).then((r) => r.json()).then((d) => (long = d)).catch(() => {}); });

  const rows = $derived.by(() => {
    if (!data.losses) return [];
    const { vocab, docs, train, test } = data;
    const L = held
      ? { chance: S.lossUniform(vocab.V), unigram: S.lossUnigram(test, vocab, train), bigram: S.lossBigram(test, vocab, train, alpha) }
      : { chance: data.losses.chance, unigram: data.losses.unigram, bigram: data.losses.bigram };
    return [
      { k: 'chance', name: 'Pure chance', note: 'all 27 symbols equally likely', v: L.chance },
      { k: 'unigram', name: 'Letter frequencies', note: 'the wheel (Chapter 2)', v: L.unigram },
      { k: 'bigram', name: 'Bigram table', note: 'one letter of memory (Chapter 2)', v: L.bigram },
    ];
  });
  const target = $derived(long ? [
    { k: 't1', name: 'microgpt.py · 500 steps', note: 'preview: the real thing, barely trained', v: long.meanLoss[long.steps.indexOf(500)] },
    { k: 't2', name: 'microgpt.py · 3,000 steps', note: 'preview: trained 6× longer', v: long.meanLoss[long.steps.indexOf(3000)] },
  ] : []);
  const maxV = 3.4;
  const fmt = (v) => (Number.isFinite(v) ? v.toFixed(3) : '∞');
  const touch = () => { if (++moved >= 2) beat?.complete(); };
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">The scoreboard · average surprise per symbol (lower is better)</div>
    <div class="ctl">
      <span class="grp" role="group" aria-label="What to grade on">
        <button class:on={!held} onclick={() => { held = false; touch(); }}>Graded on all names</button>
        <button class:on={held} onclick={() => { held = true; touch(); }}>Graded on names it never counted</button>
      </span>
      {#if held}
        <label class="al">pretend every pair was seen <input type="range" min="0" max="3" step="0.05" bind:value={alpha} oninput={touch} aria-label="Smoothing" /> <strong>{alpha.toFixed(2)}</strong> extra times</label>
      {/if}
    </div>

    {#each rows as r}
      <div class="row">
        <div class="nm">{r.name}<span>{r.note}</span></div>
        <div class="bar"><div class="fill" style="width:{Number.isFinite(r.v) ? (r.v / maxV) * 100 : 100}%" class:inf={!Number.isFinite(r.v)}></div></div>
        <div class="val" class:inf={!Number.isFinite(r.v)}>{fmt(r.v)}</div>
      </div>
    {/each}
    {#if held && rows[2] && !Number.isFinite(rows[2].v)}
      <div class="warn">The bigram table is <strong>infinitely</strong> surprised: among the names it never counted, some contain a letter pair it never saw, so it called that pair <em>impossible</em>. Drag the slider to hand it a little humility.</div>
    {:else if held && rows[2]}
      <div class="ok">With a touch of humility, the bigram table scores {rows[2].v.toFixed(3)} on names it never counted. Basically the same as on the names it did. A table of 729 numbers has almost nothing to memorise with.</div>
    {/if}

    {#if target.length}
      <div class="sep">For a sense of where we're heading: the real 200-line model</div>
      {#each target as r}
        <div class="row t">
          <div class="nm">{r.name}<span>{r.note}</span></div>
          <div class="bar"><div class="fill" style="width:{(r.v / maxV) * 100}%"></div></div>
          <div class="val">{fmt(r.v)}</div>
        </div>
      {/each}
    {/if}
  </div>
</DataGate>

<style>
  .ctl { display: flex; gap: 0.8rem 1.2rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.9rem; }
  .grp { display: inline-flex; border: 1px solid var(--line-strong); border-radius: 999px; overflow: hidden; }
  .grp button { border: 0; background: var(--surface); color: var(--ink-2); padding: 0.35rem 0.85rem; }
  .grp button.on { background: var(--accent); color: var(--on-accent); }
  .al { display: flex; gap: 0.5rem; align-items: center; color: var(--ink-2); font-size: 0.85rem; }
  .al input { accent-color: var(--accent); width: 130px; }
  .row { display: grid; grid-template-columns: 15rem 1fr 4.6rem; gap: 0.9rem; align-items: center; padding: 0.5rem 0; border-top: 1px solid var(--line); }
  @media (max-width: 640px) { .row { grid-template-columns: 1fr 4.2rem; } .row .bar { grid-column: 1 / -1; order: 3; } }
  .nm { font-weight: 600; }
  .nm span { display: block; font-weight: 400; font-size: 0.76rem; color: var(--ink-3); }
  .bar { height: 14px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
  .fill { height: 100%; background: var(--series-1); border-radius: 0 4px 4px 0; transition: width 0.35s ease; }
  .fill.inf { background: repeating-linear-gradient(45deg, var(--pain), var(--pain) 6px, transparent 6px, transparent 11px); }
  .row.t .fill { background: var(--series-3); }
  .val { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; font-size: 1.05rem; }
  .val.inf { color: var(--pain); }
  .warn, .ok { margin: 0.6rem 0 0.2rem; padding: 0.6rem 0.8rem; border-radius: 10px; }
  .warn { background: var(--pain-wash); }
  .ok { background: var(--good-wash); }
  .sep { margin: 1.1rem 0 0.3rem; color: var(--ink-3); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
</style>
