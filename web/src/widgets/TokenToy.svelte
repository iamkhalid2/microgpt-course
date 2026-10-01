<script>
  import { getContext } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { encode, tokenLabel } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';

  // Type a name, watch it become numbers. The alphabet is whatever characters appear in the data, sorted.
  const beat = getContext('beat');
  let text = $state('emma');
  let hov = $state(null);
  let edits = 0;
  const chars = $derived([...text]);
  const unknown = $derived(chars.filter((c) => !data.vocab.uchars.includes(c)));
  const ids = $derived(unknown.length ? null : encode(text, data.vocab));
  function oninput() { if (++edits >= 3) beat?.complete(); }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">Tokenizer toy</div>
    <label class="in">Type any name: <input bind:value={text} {oninput} spellcheck="false" maxlength="18" /></label>

    <div class="chips">
      {#if ids}
        {#each ids as id, i}
          <div class="chip" class:bos={id === data.vocab.BOS} class:hl={hov === id} onmouseenter={() => (hov = id)} onmouseleave={() => (hov = null)} role="presentation">
            <span class="ch">{tokenLabel(id, data.vocab)}</span><span class="id">{id}</span>
          </div>
        {/each}
      {:else}
        <div class="bad">The model has never seen {unknown.map((c) => (c === ' ' ? 'a space' : `“${c}”`)).join(', ')}. Its whole alphabet is the 26 lowercase letters below, so anything else has no number.</div>
      {/if}
    </div>
    <div class="code mono">tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]</div>

    <div class="alpha">
      {#each data.vocab.uchars as c, i}
        <div class="cell" class:hl={hov === i}><span class="ch">{c}</span><span class="id">{i}</span></div>
      {/each}
      <div class="cell bos" class:hl={hov === data.vocab.BOS}><span class="ch">⏎</span><span class="id">{data.vocab.BOS}</span></div>
    </div>
    <div class="cap">⏎ is the special <strong>BOS</strong> token ("beginning of sequence"). It isn't a letter; it marks where a name starts and ends. The numbers are just labels: <code>z</code> is not "bigger" than <code>a</code> in any meaningful way.</div>
  </div>
</DataGate>

<style>
  .in { display: block; margin-bottom: 0.9rem; color: var(--ink-2); }
  input { font-family: var(--font-mono); font-size: 1.05rem; padding: 0.4rem 0.7rem; border-radius: 10px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); margin-left: 0.4rem; width: 12rem; }
  .chips { display: flex; flex-wrap: wrap; gap: 0.35rem; min-height: 4.2rem; align-items: flex-start; }
  .chip, .cell { display: flex; flex-direction: column; align-items: center; border-radius: 9px; background: var(--code-bg); padding: 0.3rem 0.2rem; min-width: 2.3rem; border: 1px solid transparent; }
  .chip .ch { font-family: var(--font-mono); font-size: 1.3rem; }
  .id { font-size: 0.75rem; color: var(--ink-3); font-variant-numeric: tabular-nums; }
  .chip .id { color: var(--accent-strong); font-weight: 700; font-size: 0.95rem; }
  .bos { background: var(--gold-wash); border-color: var(--gold); }
  .hl { border-color: var(--accent); background: var(--accent-wash); }
  .bad { color: var(--pain); padding: 0.4rem 0; }
  .code { margin: 0.6rem 0 1rem; font-size: 0.78rem; color: var(--ink-2); background: var(--code-bg); padding: 0.4rem 0.7rem; border-radius: 8px; overflow-x: auto; white-space: nowrap; }
  .alpha { display: grid; grid-template-columns: repeat(auto-fill, minmax(2.1rem, 1fr)); gap: 0.3rem; }
  .cell .ch { font-family: var(--font-mono); }
  .cap { margin-top: 0.8rem; color: var(--ink-2); font-size: 0.85rem; }
</style>
