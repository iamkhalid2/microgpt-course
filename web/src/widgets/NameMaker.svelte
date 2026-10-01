<script>
  import { getContext } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { makeRng } from '../lib/rng.js';
  import * as S from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // Generate a batch of names from a chosen model and look at them.
  let { model = 'unigram', count = 12, title = 'Names from this model', need = 1 } = $props();
  const beat = getContext('beat');
  let names = $state([]);
  let made = 0;
  const rng = makeRng((Date.now() >>> 0) % 2147483647);
  const docSet = $derived(new Set(data.docs));
  const gen = () => {
    const out = [];
    for (let i = 0; i < count; i++) {
      out.push(model === 'chance' ? S.sampleBabble(data.vocab, rng, 10) : model === 'unigram' ? S.sampleUnigram(data.uni, data.vocab, rng, 14) : S.sampleBigram(data.bi, data.vocab, rng, 14));
    }
    names = out;
    if (++made >= need) beat?.complete();
  };
  $effect(() => { if (data.status === 'ready' && names.length === 0) gen(); });
  const real = $derived(names.filter((n) => docSet.has(n)).length);
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">{title}</div>
    <div class="names">
      {#each names as n}<span class="mono n" class:real={docSet.has(n)}>{n || '(nothing)'}</span>{/each}
    </div>
    <div class="foot">
      <button class="btn primary" onclick={gen}>↻ Make {count} more</button>
      <span class="muted">{real} of these {count} happen to be real names from the dataset (outlined).</span>
    </div>
  </div>
</DataGate>

<style>
  .names { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-bottom: 0.9rem; }
  .n { background: var(--code-bg); padding: 0.35rem 0.75rem; border-radius: 9px; font-size: 1.05rem; border: 1px solid transparent; }
  .n.real { border-color: var(--series-3); }
  .foot { display: flex; gap: 0.9rem; align-items: center; flex-wrap: wrap; }
</style>
