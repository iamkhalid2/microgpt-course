<script>
  import { getContext } from 'svelte';
  import { PRESETS, nodesOf } from '../lib/graph.js';

  // A card may only join the list AFTER every card it was made from. Click the cards in a valid order.
  const beat = getContext('beat');
  const nodes = nodesOf(PRESETS.chain);
  const label = { a: 'a (input 2)', b: 'b (input 3)', one: 'the number 1', c: 'c = a × b', d: 'd = c + 1', loss: 'loss = d²' };
  const short = { a: 'a', b: 'b', one: '1', c: 'c', d: 'd', loss: 'loss' };
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const shuffled = ['loss', 'c', 'one', 'b', 'd', 'a'];
  let list = $state([]);
  let msg = $state('');
  let shake = $state('');

  const left = $derived(shuffled.filter((id) => !list.includes(id)));
  const finished = $derived(list.length === nodes.length);

  function pick(id) {
    const need = (byId[id].inputs ?? []).filter((i) => !list.includes(i));
    if (need.length) {
      msg = `${label[id]} is made from ${need.map((n) => label[n]).join(' and ')}. ${need.length > 1 ? 'Those' : 'That'} must come first.`;
      shake = id; setTimeout(() => (shake = ''), 400);
      return;
    }
    msg = ''; list = [...list, id];
    if (list.length === nodes.length) beat?.complete();
  }
  function undo() { list = list.slice(0, -1); msg = ''; }
</script>

<div class="widget wide">
  <div class="widget-title">The ordering puzzle</div>
  <p class="q">Line these six cards up in a list so that <strong>every card appears after the cards it was made from</strong>. Click the cards one at a time, in order.</p>
  <div class="pool">
    {#each left as id}
      <button class="chip mono" class:shake={shake === id} onclick={() => pick(id)}>{label[id]}</button>
    {/each}
  </div>
  <div class="listlab">Your list, first to last:</div>
  <ol class="list">
    {#each list as id, i}<li class="mono"><span class="ix">{i + 1}</span>{label[id]}</li>{/each}
    {#if !list.length}<li class="muted">(empty)</li>{/if}
  </ol>
  {#if msg}<div class="msg" role="alert">{msg}</div>{/if}
  {#if finished}
    <div class="win">
      <strong>That's a valid order.</strong> There are several, and any of them works. Now read your list <em>backwards</em>:
      <div class="rev mono">{[...list].reverse().map((i) => short[i]).join(' → ')}</div>
      That's the order in which slopes must flow back: each card hands its slope to its ingredients only after <em>it</em> has received all of its own.
    </div>
  {/if}
  <div class="foot"><button class="btn ghost" onclick={undo} disabled={!list.length}>Undo</button><button class="btn ghost" onclick={() => { list = []; msg = ''; }} disabled={!list.length}>Start over</button></div>
</div>

<style>
  .q { margin: 0 0 0.7rem; color: var(--ink-2); }
  .pool { display: flex; gap: 0.5rem; flex-wrap: wrap; min-height: 2.8rem; margin-bottom: 0.8rem; }
  .chip { padding: 0.5rem 0.8rem; border-radius: 10px; border: 1px solid var(--line-strong); background: var(--code-bg); color: var(--ink); font-size: 0.85rem; }
  .chip:hover { border-color: var(--accent); background: var(--accent-wash); }
  .chip.shake { animation: shake 0.4s; border-color: var(--pain); background: var(--pain-wash); }
  @keyframes shake { 25% { transform: translateX(-5px); } 75% { transform: translateX(5px); } }
  .listlab { color: var(--ink-3); font-size: 0.78rem; margin-bottom: 0.2rem; }
  .list { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.25rem; }
  .list li { background: var(--surface-2); border-radius: 8px; padding: 0.3rem 0.7rem; font-size: 0.85rem; }
  .ix { display: inline-block; width: 1.3rem; color: var(--ink-3); }
  .msg { margin-top: 0.6rem; padding: 0.5rem 0.8rem; background: var(--pain-wash); border-radius: 9px; }
  .win { margin-top: 0.7rem; padding: 0.7rem 0.9rem; background: var(--good-wash); border-radius: 10px; }
  .rev { margin: 0.4rem 0; font-size: 0.9rem; }
  .foot { display: flex; gap: 0.5rem; margin-top: 0.7rem; }
</style>
