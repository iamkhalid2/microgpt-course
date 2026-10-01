<script>
  import { onMount } from 'svelte';
  import { LINES, CODE_LINES, UNLOCKS, FOCUS } from '../lib/fileMap.js';
  import { progress, litLines, litCount } from '../lib/progress.svelte.js';
  import { highlight } from '../lib/highlight.js';
  import { route, go } from '../lib/router.svelte.js';
  import { ui } from '../lib/ui.svelte.js';

  // "The File": the real microgpt.py. Lines you haven't earned are blurred; earning a chapter lights them up.
  const lit = $derived(litLines());
  const chId = $derived(route.name === 'chapter' ? 'ch' + route.id : null);
  const focus = $derived.by(() => {
    const s = new Set();
    for (const [a, b] of FOCUS[chId] ?? []) for (let n = a; n <= b; n++) s.add(n);
    return s;
  });
  let flash = $state(new Set());
  let panelEl = $state();

  const chapterOf = (n) => { for (const [ch, rs] of Object.entries(UNLOCKS)) if (rs.some(([a, b]) => n >= a && n <= b)) return ch; return null; };
  const isCode = new Set(CODE_LINES);
  const html = LINES.map((l) => highlight(l));

  onMount(() => {
    const onBanked = (e) => {
      const ch = e.detail.chapter;
      const s = new Set();
      for (const [a, b] of UNLOCKS[ch] ?? []) for (let n = a; n <= b; n++) s.add(n);
      flash = s; if (s.size) ui.fileOpen = true;
      setTimeout(() => { flash = new Set(); }, 3200);
      if (s.size) setTimeout(() => { const first = Math.min(...s); panelEl?.querySelector(`[data-n="${first}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' }); }, 350);
    };
    const onKey = (e) => { if (e.key === 'Escape') { ui.fileOpen = false; ui.museumOpen = false; } };
    window.addEventListener('igpt:banked', onBanked);
    window.addEventListener('keydown', onKey);
    return () => { window.removeEventListener('igpt:banked', onBanked); window.removeEventListener('keydown', onKey); };
  });

  function clickLine(n) {
    const ch = lit.get(n);
    if (ch) { ui.fileOpen = false; go('ch/' + ch.slice(2)); }
  }
  function mini(n) { ui.fileOpen = true; setTimeout(() => panelEl?.querySelector(`[data-n="${n}"]`)?.scrollIntoView({ block: 'center' }), 60); }
</script>

<!-- minimap: one tiny bar per line of the file -->
<nav class="mini" aria-label="Code minimap">
  {#each LINES as l, i}
    {@const n = i + 1}
    <button class="m" class:lit={lit.has(n)} class:focus={focus.has(n)} class:blank={!l.trim()} tabindex="-1" aria-hidden="true" onclick={() => mini(n)}>
      <span style="width:{Math.min(100, (l.trim().length / 70) * 100)}%"></span>
    </button>
  {/each}
</nav>

{#if ui.fileOpen}
  <div class="scrim" onclick={() => (ui.fileOpen = false)} role="presentation"></div>
  <aside class="panel" aria-label="microgpt.py">
    <div class="ph">
      <div>
        <div class="ttl">microgpt.py</div>
        <div class="sub">{litCount()} of {CODE_LINES.length} lines earned · blurred lines are waiting for their chapter</div>
      </div>
      <button class="btn ghost" onclick={() => (ui.fileOpen = false)} aria-label="Close">✕</button>
    </div>
    <div class="code" bind:this={panelEl}>
      {#each LINES as l, i}
        {@const n = i + 1}
        {@const isLit = lit.has(n)}
        <div class="ln" data-n={n} class:lit={isLit} class:focus={focus.has(n)} class:flash={flash.has(n)}
          class:locked={!isLit && isCode.has(n)} class:clickable={isLit}
          title={isLit ? `Earned in chapter ${lit.get(n).slice(2)}. Click to revisit.` : isCode.has(n) ? (chapterOf(n) ? `Unlocks in chapter ${chapterOf(n).slice(2)}` : 'Unlocks in a later chapter') : ''}
          onclick={() => clickLine(n)} role={isLit ? 'button' : undefined}>
          <span class="num">{n}</span><span class="txt">{@html html[i] || '&nbsp;'}</span>
        </div>
      {/each}
    </div>
  </aside>
{/if}

<style>
  .mini { position: fixed; right: 0; top: var(--header-h); bottom: 0; width: 30px; z-index: 30; display: flex; flex-direction: column; padding: 8px 5px; gap: 0; background: linear-gradient(to left, var(--bg) 60%, transparent); }
  .m { all: unset; flex: 1; min-height: 1px; display: flex; align-items: center; cursor: pointer; }
  .m span { display: block; height: 100%; max-height: 2.2px; min-height: 1px; background: var(--line-strong); border-radius: 1px; }
  .m.lit span { background: var(--accent); }
  .m.focus span { background: var(--pain); }
  .m.focus.lit span { background: var(--pain); }
  .m.blank span { visibility: hidden; }
  @media (max-width: 720px) { .mini { width: 16px; padding: 6px 3px; } }

  .scrim { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.35); z-index: 90; }
  .panel { position: fixed; top: 0; right: 0; bottom: 0; width: min(680px, 100vw); z-index: 100; background: var(--surface); border-left: 1px solid var(--line-strong); box-shadow: var(--shadow); display: flex; flex-direction: column; animation: slide 0.25s ease both; }
  @keyframes slide { from { transform: translateX(40px); opacity: 0; } to { transform: none; opacity: 1; } }
  .ph { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: 0.9rem 1.1rem; border-bottom: 1px solid var(--line); font-family: var(--font-ui); }
  .ttl { font-family: var(--font-mono); font-weight: 600; }
  .sub { font-size: 0.76rem; color: var(--ink-3); }
  .code { overflow: auto; padding: 0.6rem 0 2rem; font-family: var(--font-mono); font-size: 0.78rem; line-height: 1.55; }
  .ln { display: flex; gap: 0.9rem; padding: 0 1rem 0 0; white-space: pre; border-left: 3px solid transparent; }
  .num { flex: none; width: 3rem; text-align: right; color: var(--ink-3); opacity: 0.6; user-select: none; }
  .txt { flex: 1; }
  .ln.locked .txt { filter: blur(4px); opacity: 0.45; user-select: none; }
  .ln:not(.locked):not(.lit) { opacity: 0.55; }
  .ln.lit { border-left-color: var(--accent); background: var(--accent-wash); }
  .ln.clickable { cursor: pointer; }
  .ln.clickable:hover { background: color-mix(in srgb, var(--accent) 22%, transparent); }
  .ln.focus { border-left-color: var(--pain); }
  .ln.flash { animation: flash 3s ease both; }
  @keyframes flash { 0% { background: var(--gold); } 100% { background: var(--accent-wash); } }
  .txt :global(i) { font-style: normal; }
  .txt :global(.k) { color: var(--syn-keyword); }
  .txt :global(.s) { color: var(--syn-string); }
  .txt :global(.n) { color: var(--syn-number); }
  .txt :global(.c) { color: var(--syn-comment); font-style: italic; }
</style>
