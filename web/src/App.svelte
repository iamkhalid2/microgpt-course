<script>
  import { route, href } from './lib/router.svelte.js';
  import { byNum } from './chapters/index.js';
  import Header from './components/Header.svelte';
  import Home from './components/Home.svelte';
  import FilePanel from './components/FilePanel.svelte';
  import Glossary from './components/Glossary.svelte';
  import Museum from './components/Museum.svelte';
  import { ui } from './lib/ui.svelte.js';
  import { loadData } from './lib/data.svelte.js';

  loadData();
  const ch = $derived(route.name === 'chapter' ? byNum(route.id) : null);
  let Comp = $state(null);
  let failed = $state('');

  $effect(() => {
    const target = ch;
    Comp = null; failed = '';
    if (target?.ready) {
      target.load().then((m) => { if (byNum(route.id) === target) Comp = m.default; }).catch((e) => { failed = String(e); });
    }
  });
</script>

<Header />
<main>
  {#if route.name === 'home'}
    <Home />
  {:else if route.name === 'glossary'}
    <Glossary />
  {:else if !ch}
    <div class="note"><h2>No such chapter</h2><p><a href={href('')}>Back to the map</a></p></div>
  {:else if !ch.ready}
    <div class="note">
      <div class="chip">Chapter {ch.num}</div>
      <h2>{ch.title}</h2>
      <p><strong>The pain:</strong> {ch.pain}</p>
      <p><strong>What you'll invent:</strong> {ch.invent}</p>
      <p class="muted">This chapter is still being built. <a href={href('')}>Back to the map</a></p>
    </div>
  {:else if failed}
    <div class="note"><h2>Couldn't load this chapter</h2><p>{failed}</p></div>
  {:else if Comp}
    {#key ch.num}<Comp />{/key}
  {:else}
    <div class="note muted">Loading…</div>
  {/if}
</main>
<FilePanel />
<Museum bind:open={ui.museumOpen} />

<style>
  main { padding-top: var(--header-h); padding-right: 30px; min-height: 100vh; }
  .note { max-width: var(--col); margin: 4rem auto; padding: 0 1.25rem; }
  .chip { font-family: var(--font-ui); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent-strong); }
  @media (max-width: 720px) { main { padding-right: 16px; } }
</style>
