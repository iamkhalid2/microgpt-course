<script>
  import { GLOSSARY } from '../lib/glossary.js';
  import { href } from '../lib/router.svelte.js';
  let q = $state('');
  const shown = $derived(GLOSSARY.filter((g) => !q.trim() || (g.term + ' ' + g.def).toLowerCase().includes(q.trim().toLowerCase())).sort((a, b) => a.term.localeCompare(b.term)));
</script>

<div class="gl">
  <h1>Glossary</h1>
  <p class="muted">Every term the course uses, in plain words, with the chapter that builds it.</p>
  <input type="search" bind:value={q} placeholder="Search terms…" aria-label="Search the glossary" />
  <dl>
    {#each shown as g (g.term)}
      <div class="e"><dt>{g.term}</dt><dd>{g.def} <a href={href(`ch/${g.ch}`)}>Chapter {g.ch}</a></dd></div>
    {:else}
      <p class="muted">Nothing matches “{q}”.</p>
    {/each}
  </dl>
</div>

<style>
  .gl { max-width: var(--col); margin: 2.5rem auto 5rem; padding: 0 1.25rem; }
  h1 { font-family: var(--font-display); margin: 0 0 0.3rem; }
  input { width: 100%; box-sizing: border-box; margin: 1rem 0; padding: 0.6rem 0.9rem; border-radius: 10px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-family: var(--font-ui); font-size: 0.95rem; }
  dl { margin: 0; } .e { padding: 0.7rem 0; border-bottom: 1px solid var(--line); }
  dt { font-family: var(--font-ui); font-weight: 600; } dd { margin: 0.2rem 0 0; color: var(--ink-2); }
  dd a { font-family: var(--font-ui); font-size: 0.78rem; margin-left: 0.3rem; white-space: nowrap; }
</style>
