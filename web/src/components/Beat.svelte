<script>
  import { getContext, setContext, tick } from 'svelte';
  import { progress } from '../lib/progress.svelte.js';

  // One "scene" of a lesson. Scenes drip out one at a time; a gated scene waits until you've actually done the thing.
  let { gate = false, last = false, children } = $props();

  const L = getContext('lesson');
  const idx = L.register();
  let done = $state(!gate || idx < L.current);
  setContext('beat', { complete() { done = true; } });

  const visible = $derived(progress.reader || idx <= L.current);
  const isCurrent = $derived(idx === L.current && !progress.reader);

  async function next() {
    L.advance(idx + 1);
    await tick();
    document.getElementById(`${L.id}-b${idx + 1}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
</script>

{#if visible}
  <section class="beat" id="{L.id}-b{idx}">
    {@render children()}
    {#if isCurrent && !last}
      <div class="cont">
        <button class="btn primary" disabled={gate && !done} onclick={next}>
          {gate && !done ? 'Do the thing above to continue' : 'Continue'} <span aria-hidden="true">↓</span>
        </button>
        {#if gate && !done}<button class="btn ghost" onclick={next}>skip this</button>{/if}
      </div>
    {/if}
  </section>
{/if}

<style>
  .beat { max-width: var(--wide); margin: 0 auto 2.6rem; scroll-margin-top: calc(var(--header-h) + 18px); animation: rise 0.5s ease both; }
  .beat > :global(*) { max-width: var(--col); margin-left: auto; margin-right: auto; }
  .beat > :global(.wide) { max-width: var(--wide); }
  .cont { display: flex; gap: 0.6rem; align-items: center; margin-top: 2rem; }
  @keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
</style>
