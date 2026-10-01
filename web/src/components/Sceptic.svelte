<script>
  // A question a beginner is too polite to ask out loud. Click to hear the honest answer.
  let { q, children } = $props();
  let open = $state(false);
</script>
<div class="sceptic" class:open>
  <button class="ask" onclick={() => (open = !open)} aria-expanded={open}>
    <span class="avatar" aria-hidden="true">?</span>
    <span class="txt"><span class="who">The Sceptic asks</span>{q}</span>
    <span class="chev" aria-hidden="true">{open ? '−' : '+'}</span>
  </button>
  {#if open}<div class="ans">{@render children()}</div>{/if}
</div>
<style>
  .sceptic { border: 1px dashed var(--line-strong); border-radius: var(--radius); margin-bottom: 1.4rem; background: var(--surface); }
  .ask { all: unset; box-sizing: border-box; display: flex; gap: 0.8rem; align-items: center; width: 100%; padding: 0.8rem 1rem; cursor: pointer; font-family: var(--font-body); font-style: italic; }
  .ask:focus-visible { outline: 2px solid var(--accent); border-radius: var(--radius); }
  .avatar { flex: none; width: 2rem; height: 2rem; border-radius: 50%; background: var(--gold-wash); color: var(--gold); display: grid; place-items: center; font-family: var(--font-display); font-weight: 700; font-style: normal; border: 1px solid var(--gold); }
  .txt { flex: 1; }
  .who { display: block; font-family: var(--font-ui); font-style: normal; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-3); font-weight: 600; }
  .chev { font-family: var(--font-ui); color: var(--ink-3); font-size: 1.2rem; }
  .ans { padding: 0 1.1rem 0.6rem 3.8rem; animation: rise 0.3s ease both; }
  @keyframes rise { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }
  .ans :global(p:last-child) { margin-bottom: 0.4rem; }
</style>
