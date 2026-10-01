<script>
  import { setContext, onMount } from 'svelte';
  import { progress } from '../lib/progress.svelte.js';
  import { py } from '../lib/py.svelte.js';
  import { loadData } from '../lib/data.svelte.js';

  let { id, num, title, tagline, setup = '', children } = $props();

  const lesson = $state({ current: progress.beat[id] ?? 0, count: 0 });
  let counter = 0;

  setContext('lesson', {
    id,
    get current() { return lesson.current; },
    get count() { return lesson.count; },
    register() { return counter++; },
    advance(i) {
      lesson.current = Math.max(lesson.current, i);
      progress.beat[id] = lesson.current;
    },
  });

  py.startSession(id, setup);
  loadData();
  onMount(() => { lesson.count = counter; });

  const pct = $derived(progress.reader || lesson.count === 0 ? 100 : Math.min(100, ((lesson.current + 1) / lesson.count) * 100));
</script>

<div class="meter" aria-hidden="true"><div style="width:{pct}%"></div></div>

<article class="lesson">
  <header class="lhead">
    <div class="chip">Chapter {num}</div>
    <h1>{title}</h1>
    <p class="tag">{tagline}</p>
  </header>
  {@render children()}
</article>

<style>
  .meter { position: fixed; top: var(--header-h); left: 0; right: 0; height: 3px; z-index: 40; background: transparent; }
  .meter div { height: 100%; background: var(--accent); transition: width 0.5s ease; border-radius: 0 3px 3px 0; }
  .lesson { padding: 2.5rem 1.25rem 7rem; }
  .lhead { max-width: var(--col); margin: 0 auto 2.4rem; }
  .chip { font-family: var(--font-ui); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent-strong); margin-bottom: 0.8rem; }
  .tag { font-size: 1.3rem; color: var(--ink-2); font-style: italic; }
</style>
