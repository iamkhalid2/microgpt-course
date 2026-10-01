<script>
  import { progress, bank } from '../lib/progress.svelte.js';
  import { UNLOCKS } from '../lib/fileMap.js';
  import { chapters } from '../chapters/index.js';
  import { href } from '../lib/router.svelte.js';

  // The end of a lesson: say what you now own, then light up the matching lines of microgpt.py.
  let { chapter, children } = $props();
  const banked = $derived(!!progress.banked[chapter]);
  const nLines = (UNLOCKS[chapter] ?? []).reduce((s, [a, b]) => s + (b - a + 1), 0);
  const nextCh = $derived(chapters[chapters.findIndex((c) => c.id === chapter) + 1]);

  function claim() {
    bank(chapter);
    window.dispatchEvent(new CustomEvent('igpt:banked', { detail: { chapter } }));
  }
</script>

<div class="bank widget wide" class:banked>
  <div class="widget-title">Bank it</div>
  <div class="own">{@render children()}</div>
  {#if !banked}
    <button class="btn primary big" onclick={claim}>{nLines ? `Light up ${nLines} line${nLines === 1 ? '' : 's'} of microgpt.py` : 'Mark this chapter complete'}</button>
  {:else}
    <div class="done">
      <span class="check">✓</span> {nLines ? "Banked. Open The File (top right) to see what you've earned." : 'Done.'}
      {#if nextCh}
        <a class="btn primary" href={href(nextCh.ready ? `ch/${nextCh.num}` : '')}>{nextCh.ready ? `Next: ${nextCh.title} →` : 'Back to the map →'}</a>
      {/if}
    </div>
  {/if}
</div>

<style>
  .bank { border: 2px solid var(--accent); }
  .bank.banked { border-color: var(--good); }
  .own { font-family: var(--font-body); font-size: 1.05rem; margin-bottom: 1rem; }
  .own :global(ul) { padding-left: 1.2rem; margin: 0; }
  .own :global(li) { margin-bottom: 0.35rem; }
  .big { font-size: 0.95rem; padding: 0.7rem 1.2rem; }
  .done { display: flex; gap: 0.8rem; align-items: center; flex-wrap: wrap; }
  .check { color: var(--good); font-weight: 800; }
</style>
