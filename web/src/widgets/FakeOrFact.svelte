<script>
  import { getContext } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { progress } from '../lib/progress.svelte.js';
  import { makeRng } from '../lib/rng.js';
  import * as S from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // One real name hides among three impostors made by a model. Your accuracy falls as the models improve.
  let { levels = ['chance'], need = 4, title = 'Spot the real name' } = $props();
  const beat = getContext('beat');

  const LEVEL = {
    chance: { label: 'Random letters', note: 'every letter equally likely' },
    unigram: { label: 'Letter frequencies', note: 'common letters are common' },
    bigram: { label: 'Bigram table', note: 'each letter depends on the one before' },
  };
  let level = $state(levels[0]);
  let rng = makeRng(Date.now() % 2147483647);
  let options = $state([]);
  let answerIdx = $state(0);
  let picked = $state(null);
  let answered = $state(0);

  const docSet = $derived(new Set(data.docs));

  function fake(lv) {
    for (let tries = 0; tries < 50; tries++) {
      let n;
      if (lv === 'chance') n = S.sampleBabble(data.vocab, rng, 10);
      else if (lv === 'unigram') n = S.sampleUnigram(data.uni, data.vocab, rng, 12);
      else n = S.sampleBigram(data.bi, data.vocab, rng, 12);
      if (n.length >= 2 && !docSet.has(n)) return n;
    }
    return 'xq';
  }
  function deal() {
    const real = rng.pick(data.docs);
    const fakes = new Set();
    while (fakes.size < 3) fakes.add(fake(level));
    const all = [...fakes];
    answerIdx = rng.int(4);
    all.splice(answerIdx, 0, real);
    options = all; picked = null;
  }
  $effect(() => { if (data.status === 'ready' && options.length === 0) deal(); });
  function setLevel(lv) { level = lv; deal(); }

  const rec = $derived(progress.game[level] ?? { played: 0, correct: 0 });
  function choose(i) {
    if (picked !== null) return;
    picked = i;
    const r = progress.game[level] ?? { played: 0, correct: 0 };
    progress.game[level] = { played: r.played + 1, correct: r.correct + (i === answerIdx ? 1 : 0) };
    answered++;
    if (answered >= need) beat?.complete();
  }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">{title}</div>
    {#if levels.length > 1}
      <div class="lv">
        {#each levels as lv}
          <button class="lvb" class:on={level === lv} onclick={() => setLevel(lv)}>{LEVEL[lv].label}</button>
        {/each}
      </div>
    {/if}
    <p class="q">One of these four is a <strong>real</strong> name from the dataset. The other three were made by the <strong>{LEVEL[level].label.toLowerCase()}</strong> model ({LEVEL[level].note}). Click the real one.</p>
    <div class="opts">
      {#each options as n, i}
        <button class="opt mono" disabled={picked !== null} class:right={picked !== null && i === answerIdx} class:wrong={picked === i && i !== answerIdx} onclick={() => choose(i)}>{n}</button>
      {/each}
    </div>
    <div class="foot">
      {#if picked !== null}
        <span class="res" class:ok={picked === answerIdx}>{picked === answerIdx ? 'Got it.' : 'Fooled!'}</span>
        <button class="btn primary" onclick={deal}>Next →</button>
      {:else}<span class="muted">{answered < need ? `${need - answered} more to continue` : ''}</span>{/if}
      <span class="sp"></span>
      <span class="score">Your record here: <strong>{rec.correct}/{rec.played}</strong>{rec.played ? ` (${Math.round((100 * rec.correct) / rec.played)}%)` : ''} · guessing gets 25%</span>
    </div>
  </div>
</DataGate>

<style>
  .lv { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
  .lvb { padding: 0.3rem 0.8rem; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); }
  .lvb.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .q { margin: 0 0 0.8rem; color: var(--ink-2); }
  .opts { display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: 0.6rem; }
  .opt { padding: 0.9rem; font-size: 1.25rem; border-radius: 12px; border: 1px solid var(--line-strong); background: var(--code-bg); color: var(--ink); }
  .opt:hover:not(:disabled) { border-color: var(--accent); background: var(--accent-wash); }
  .opt.right { border-color: var(--good); background: var(--good-wash); }
  .opt.wrong { border-color: var(--pain); background: var(--pain-wash); }
  .foot { display: flex; align-items: center; gap: 0.8rem; margin-top: 0.9rem; flex-wrap: wrap; }
  .sp { flex: 1; }
  .res { font-weight: 700; color: var(--pain); }
  .res.ok { color: var(--good); }
  .score { color: var(--ink-3); font-size: 0.82rem; }
</style>
