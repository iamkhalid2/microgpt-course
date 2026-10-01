<script>
  import { route, href } from '../lib/router.svelte.js';
  import { chapters, byNum } from '../chapters/index.js';
  import { progress, litCount } from '../lib/progress.svelte.js';
  import { CODE_LINES } from '../lib/fileMap.js';
  import { ui } from '../lib/ui.svelte.js';
  import { data } from '../lib/data.svelte.js';

  const current = $derived(route.name === 'chapter' ? byNum(route.id) : null);
  let theme = $state(document.documentElement.dataset.theme ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
  function toggleTheme() {
    theme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('igpt.theme', theme); } catch {}
  }
  const best = $derived(progress.banked.ch3 && data.losses ? data.losses.bigram : null);
  const lines = $derived(litCount());
</script>

<header class="hdr">
  <a class="brand" href={href('')}><span class="logo">g</span><span class="name">Invent a GPT</span></a>
  {#if current}
    <div class="here"><span class="sep">/</span> <span class="num">{current.num}</span> {current.title}</div>
  {/if}
  <div class="sp"></div>

  <a class="pill gl" href={href('glossary')} title="Every term, in plain words"><span class="lbl">Glossary</span><strong>A–Z</strong></a>
  <button class="pill" onclick={() => (ui.museumOpen = !ui.museumOpen)} title="Every model you've built, scored">
    <span class="lbl">Best loss</span><strong>{best ? best.toFixed(2) : '—'}</strong>
  </button>
  <button class="pill" onclick={() => (ui.fileOpen = !ui.fileOpen)} title="The real microgpt.py, lit up as you earn it">
    <span class="lbl">The File</span><strong>{lines}/{CODE_LINES.length}</strong>
  </button>
  <button class="icon" onclick={() => (progress.reader = !progress.reader)} aria-pressed={progress.reader} title="Reader mode: show every scene at once instead of one at a time">{progress.reader ? '☰' : '⋮'}</button>
  <button class="icon" onclick={toggleTheme} aria-label="Toggle dark mode" title="Light / dark">{theme === 'dark' ? '☀' : '☾'}</button>
</header>

<style>
  .hdr { position: fixed; z-index: 50; top: 0; left: 0; right: 0; height: var(--header-h); display: flex; align-items: center; gap: 0.7rem; padding: 0 1rem 0 1.1rem; background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); font-family: var(--font-ui); }
  .brand { display: flex; align-items: center; gap: 0.6rem; text-decoration: none; color: var(--ink); font-weight: 600; }
  .logo { width: 28px; height: 28px; border-radius: 8px; background: var(--accent-strong); color: #fff; display: grid; place-items: center; font-family: var(--font-mono); font-weight: 700; position: relative; }
  .logo::after { content: ''; position: absolute; top: -3px; right: -3px; width: 8px; height: 8px; border-radius: 50%; background: var(--gold); border: 2px solid var(--bg); }
  .name { font-family: var(--font-display); font-size: 1.05rem; }
  .here { color: var(--ink-2); font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .here .num { display: inline-grid; place-items: center; width: 1.3rem; height: 1.3rem; border-radius: 50%; background: var(--accent-wash); color: var(--accent-strong); font-weight: 700; font-size: 0.75rem; margin-right: 0.2rem; }
  .sep { color: var(--line-strong); }
  .sp { flex: 1; }
  .pill { display: flex; gap: 0.5rem; align-items: baseline; padding: 0.35rem 0.8rem; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-size: 0.8rem; }
  a.pill { text-decoration: none; }
  .pill:hover { border-color: var(--ink-3); }
  .pill .lbl { color: var(--ink-3); font-size: 0.72rem; }
  .pill strong { font-variant-numeric: tabular-nums; }
  .icon { width: 2rem; height: 2rem; border-radius: 50%; border: 1px solid transparent; background: transparent; color: var(--ink-2); font-size: 1rem; }
  .icon:hover, .icon[aria-pressed='true'] { background: var(--surface-2); }
  @media (max-width: 760px) { .name, .here, .pill .lbl { display: none; } .pill.gl .lbl { display: inline; } .pill.gl strong { display: none; } }
</style>
