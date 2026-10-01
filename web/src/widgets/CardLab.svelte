<script>
  import { getContext } from 'svelte';
  import { backward } from '../lib/graph.js';
  import GraphView from './GraphView.svelte';

  // Build a computation graph the way an engine does: every piece of arithmetic makes a new memory card.
  const beat = getContext('beat');
  let a = $state(2), b = $state(3);
  let steps = $state([]);        // { id, op, src: [idOrConst], k?, label }
  let op = $state('mul');
  let s1 = $state('a'), s2 = $state('b'), num = $state(1), k = $state(2);
  let ran = $state(false);
  let msg = $state('');
  const names = 'cdefghijk';

  const OPS = {
    add: { text: 'add two', n: 2 }, mul: { text: 'multiply two', n: 2 },
    pow: { text: 'raise to a power', n: 1 }, log: { text: 'take the log of', n: 1 }, neg: { text: 'negate', n: 1 },
  };
  const ids = $derived(['a', 'b', ...steps.map((s) => s.id)]);
  const src = (v) => (v === '#' ? { id: null } : { id: v });

  // turn the learner's steps into a graph for the engine
  const model = $derived.by(() => {
    const nodes = [{ id: 'a', op: 'input', value: a }, { id: 'b', op: 'input', value: b }];
    for (const s of steps) {
      const inputs = s.src.map((x) => {
        if (typeof x === 'object') { const cid = `n${s.id}${nodes.length}`; nodes.push({ id: cid, op: 'const', value: x.v }); return cid; }
        return x;
      });
      nodes.push({ id: s.id, op: s.op, inputs, k: s.k, label: s.label });
    }
    const r = backward(nodes, {});
    const gnodes = nodes.map((n) => ({ id: n.id, label: n.op === 'input' || n.op === 'const' ? (n.op === 'const' ? 'number' : n.id) : n.label, data: r.val[n.id], grad: ran ? r.grad[n.id] : 0 }));
    const edges = nodes.flatMap((n) => (n.inputs ?? []).map((i) => [i, n.id]));
    return { r, graph: { nodes: gnodes, edges }, last: steps[steps.length - 1]?.id };
  });

  function value(id) { return model.r.val[id] ?? (id === 'a' ? a : id === 'b' ? b : NaN); }
  function make() {
    msg = '';
    const id = names[steps.length];
    if (!id) { msg = 'That is plenty of cards. Try “Run backward”.'; return; }
    const O = OPS[op];
    const x = s1 === '#' ? { v: num } : s1;
    const y = op === 'add' || op === 'mul' ? (s2 === '#' ? { v: num } : s2) : null;
    const xv = typeof x === 'object' ? x.v : value(x);
    if (op === 'log' && !(xv > 0)) { msg = 'The log only works on positive numbers. Pick another card.'; return; }
    const srcs = y === null ? [x] : [x, y];
    const sym = { add: '+', mul: '×' }[op];
    const nm = (v) => (typeof v === 'object' ? String(v.v) : v);
    const label = op === 'pow' ? `${id} = ${nm(x)}^${k}` : op === 'log' ? `${id} = log(${nm(x)})` : op === 'neg' ? `${id} = −${nm(x)}` : `${id} = ${nm(x)} ${sym} ${nm(y)}`;
    steps = [...steps, { id, op, src: srcs, k: op === 'pow' ? k : undefined, label }];
    ran = false; s1 = id;
  }
  function undo() { steps = steps.slice(0, -1); ran = false; }
  function run() { ran = true; if (steps.length >= 3) beat?.complete(); }
</script>

<div class="widget wide">
  <div class="widget-title">The card builder · every operation makes a card</div>
  <div class="inputs">
    <label>a = <input type="number" step="any" bind:value={a} oninput={() => (ran = false)} /></label>
    <label>b = <input type="number" step="any" bind:value={b} oninput={() => (ran = false)} /></label>
    <span class="muted">Try: multiply a and b, add 1, then raise the result to the power 2.</span>
  </div>
  <div class="form">
    <span>Make a new card:</span>
    <select bind:value={op}>{#each Object.entries(OPS) as [key, o]}<option value={key}>{o.text}</option>{/each}</select>
    <select bind:value={s1} aria-label="First ingredient">{#each ids as i}<option value={i}>{i}</option>{/each}<option value="#">a plain number…</option></select>
    {#if op === 'add' || op === 'mul'}
      <span>and</span>
      <select bind:value={s2} aria-label="Second ingredient">{#each ids as i}<option value={i}>{i}</option>{/each}<option value="#">a plain number…</option></select>
    {/if}
    {#if s1 === '#' || ((op === 'add' || op === 'mul') && s2 === '#')}<input class="num" type="number" step="any" bind:value={num} aria-label="The plain number" />{/if}
    {#if op === 'pow'}<span>power</span><input class="num" type="number" step="1" bind:value={k} aria-label="The power" />{/if}
    <button class="btn primary" onclick={make}>Make the card</button>
    <button class="btn ghost" onclick={undo} disabled={!steps.length}>Undo</button>
  </div>
  {#if msg}<div class="msg">{msg}</div>{/if}

  {#if steps.length}
    <GraphView graph={model.graph} title={ran ? 'Slopes flowed backwards through the cards' : 'Your cards so far'} />
    <div class="foot">
      <button class="btn primary" onclick={run} disabled={ran}>Run backward</button>
      <span class="muted">{steps.length < 3 ? `Make at least 3 cards (${steps.length} so far), then run backward.` : ran ? `The last card's value is ${+value(model.last).toFixed(4)}. Each card now shows its slope.` : 'Ready.'}</span>
    </div>
  {:else}
    <div class="empty">No cards yet except your two inputs. Make one above.</div>
  {/if}
</div>

<style>
  .inputs { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.7rem; font-family: var(--font-mono); }
  .inputs input, .num { width: 5rem; padding: 0.3rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-family: var(--font-mono); }
  .form { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
  select { padding: 0.35rem 0.5rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-family: var(--font-ui); }
  .msg { margin-top: 0.5rem; color: var(--pain); }
  .foot { display: flex; gap: 0.8rem; align-items: center; margin-top: 0.7rem; flex-wrap: wrap; }
  .empty { margin-top: 0.8rem; color: var(--ink-3); }
</style>
