<script>
  import { getContext } from 'svelte';
  import { PRESETS, nodesOf, topoTrace } from '../lib/graph.js';

  // Step through the recursive "visit" that lines cards up. Watch the stack of unfinished visits grow and shrink.
  const beat = getContext('beat');
  let pick = $state('chain');
  let i = $state(0);
  const nodes = $derived(nodesOf(PRESETS[pick]));
  const root = $derived(nodes[nodes.length - 1].id);
  const tr = $derived(topoTrace(nodes, root));
  const ev = $derived(tr.events[Math.min(i, tr.events.length - 1)]);
  const byId = $derived(Object.fromEntries(nodes.map((n) => [n.id, n])));
  const lineOn = $derived(ev.kind === 'skip' ? 0 : ev.kind === 'enter' ? 1 : 4);
  const code = ['if this card has been seen: stop', 'mark it as seen', 'for each ingredient: visit(ingredient)', '   (come back here when each is done)', 'add this card to the list'];
  const words = $derived.by(() => {
    const ing = (byId[ev.id].inputs ?? []);
    if (ev.kind === 'skip') return `Visit ${ev.id}: already seen. Nothing to do. (Cards used twice are only listed once.)`;
    if (ev.kind === 'enter') return ing.length ? `Visit ${ev.id}. Not seen yet, so mark it. Its ingredients are ${ing.join(' and ')}; visit those first.` : `Visit ${ev.id}. It has no ingredients, so there is nothing to wait for.`;
    return `Every ingredient of ${ev.id} is already on the list, so add ${ev.id}.`;
  });
  function go(d) { i = Math.max(0, Math.min(tr.events.length - 1, i + d)); if (i === tr.events.length - 1) beat?.complete(); }
  function choose(p) { pick = p; i = 0; }
</script>

<div class="widget wide">
  <div class="widget-title">Recursion, step by step</div>
  <div class="top">
    <button class="chip" class:on={pick === 'chain'} onclick={() => choose('chain')}>The chain</button>
    <button class="chip" class:on={pick === 'twice'} onclick={() => choose('twice')}>One input used twice</button>
  </div>
  <div class="grid">
    <div>
      <div class="h">The recipe, called <span class="mono">visit(card)</span></div>
      <div class="code mono">{#each code as l, n}<div class="ln" class:on={n === lineOn || (ev.kind === 'enter' && n === 2 && (byId[ev.id].inputs ?? []).length)}>{l}</div>{/each}</div>
    </div>
    <div>
      <div class="h">Unfinished visits (the stack: newest on top)</div>
      <div class="stack">{#each [...ev.stack].reverse() as s, n}<div class="fr mono" class:top={n === 0}>visit({s})</div>{/each}{#if !ev.stack.length}<div class="muted">(none)</div>{/if}</div>
      <div class="h">The list so far</div>
      <div class="order mono">{ev.order.length ? ev.order.join(' → ') : '(empty)'}</div>
    </div>
  </div>
  <div class="says">{words}</div>
  <div class="ctl">
    <button class="btn" onclick={() => go(-1)} disabled={i === 0}>◀ Back</button>
    <button class="btn primary" onclick={() => go(1)} disabled={i >= tr.events.length - 1}>Next ▶</button>
    <span class="muted">step {i + 1} of {tr.events.length}</span>
  </div>
</div>

<style>
  .top { display: flex; gap: 0.4rem; margin-bottom: 0.7rem; }
  .chip { padding: 0.3rem 0.8rem; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); font-size: 0.8rem; }
  .chip.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
  @media (max-width: 760px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .h { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin: 0.2rem 0 0.3rem; }
  .code { background: var(--code-bg); border-radius: 10px; padding: 0.4rem 0; font-size: 0.8rem; }
  .ln { padding: 0.15rem 0.8rem; border-left: 3px solid transparent; white-space: pre; }
  .ln.on { background: var(--accent-wash); border-left-color: var(--accent); }
  .stack { display: grid; gap: 3px; margin-bottom: 0.8rem; min-height: 2rem; }
  .fr { background: var(--surface-2); border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.82rem; }
  .fr.top { background: var(--accent-wash); font-weight: 700; }
  .order { background: var(--surface-2); border-radius: 8px; padding: 0.4rem 0.7rem; font-size: 0.85rem; min-height: 2rem; }
  .says { margin-top: 0.8rem; padding: 0.6rem 0.8rem; background: var(--accent-wash); border-radius: 10px; min-height: 3rem; }
  .ctl { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.7rem; }
</style>
