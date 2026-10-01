<script>
  import { getContext, onMount } from 'svelte';
  import { progress } from '../lib/progress.svelte.js';

  // Commit a guess BEFORE the answer is shown. Being wrong first is what makes the right answer stick.
  // kind: 'choice' (options[], answer=index) | 'number' (answer, unit, tolerance) | 'text' (free thought, no grading)
  let { id, kind = 'choice', options = [], answer = null, unit = '', tolerance = 0, min = 10, placeholder = '', q, children } = $props();

  const beat = getContext('beat');
  const saved = progress.answers[id];
  let value = $state(saved?.value ?? '');
  let locked = $state(saved !== undefined);

  onMount(() => { if (locked) beat?.complete(); });

  const correct = $derived(
    kind === 'choice' ? value === answer
    : kind === 'number' ? Math.abs(Number(value) - answer) <= tolerance
    : true
  );
  const canLock = $derived(kind === 'choice' ? value !== '' : kind === 'number' ? value !== '' && !isNaN(Number(value)) : String(value).trim().length >= min);

  function lock(v) {
    if (kind === 'choice') value = v;
    if (kind === 'choice' || canLock) {
      locked = true;
      progress.answers[id] = { value: kind === 'choice' ? v : value };
      beat?.complete();
    }
  }
</script>

<div class="predict widget wide" class:locked>
  <div class="widget-title">Predict first</div>
  <div class="q">{@render q()}</div>

  {#if kind === 'choice'}
    <div class="opts">
      {#each options as o, i}
        <button class="opt" class:picked={locked && value === i} class:right={locked && i === answer} class:wrong={locked && value === i && i !== answer}
          disabled={locked} onclick={() => lock(i)}>
          <span class="k">{String.fromCharCode(65 + i)}</span>{o}
        </button>
      {/each}
    </div>
  {:else if kind === 'number'}
    <div class="row">
      <input type="number" step="any" bind:value disabled={locked} placeholder="your guess" onkeydown={(e) => e.key === 'Enter' && canLock && lock()} />
      {#if unit}<span class="unit">{unit}</span>{/if}
      {#if !locked}<button class="btn primary" disabled={!canLock} onclick={() => lock()}>Lock in guess</button>{/if}
    </div>
  {:else}
    <textarea rows="3" bind:value disabled={locked} {placeholder}></textarea>
    {#if !locked}
      <div class="row"><button class="btn primary" disabled={!canLock} onclick={() => lock()}>Lock in my idea</button>
        <span class="hint">{String(value).trim().length < min ? 'A sentence or two. Wrong is fine.' : ''}</span></div>
    {/if}
  {/if}

  {#if locked}
    <div class="reveal">
      {#if kind !== 'text'}
        <div class="verdict" class:ok={correct}>
          {#if kind === 'choice'}{correct ? 'Yes.' : 'Not quite.'}
          {:else}You guessed {saved?.value ?? value}{unit ? ' ' + unit : ''}. {Number(saved?.value ?? value) === answer ? 'Spot on!' : correct ? 'Close!' : ''}{/if}
        </div>
      {:else}
        <div class="verdict ok">Saved. Here's how it plays out:</div>
      {/if}
      <div class="rbody">{@render children()}</div>
    </div>
  {/if}
</div>

<style>
  .q { font-family: var(--font-body); font-size: 1.08rem; margin-bottom: 0.9rem; }
  .q :global(p:last-child) { margin-bottom: 0; }
  .opts { display: grid; gap: 0.5rem; }
  .opt { text-align: left; display: flex; gap: 0.7rem; align-items: baseline; padding: 0.65rem 0.85rem; border-radius: 10px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-size: 0.95rem; }
  .opt:hover:not(:disabled) { border-color: var(--accent); background: var(--accent-wash); }
  .opt:disabled { cursor: default; }
  .opt .k { font-weight: 700; color: var(--ink-3); font-size: 0.8rem; }
  .opt.right { border-color: var(--good); background: var(--good-wash); }
  .opt.wrong { border-color: var(--pain); background: var(--pain-wash); }
  .row { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; margin-top: 0.3rem; }
  input[type='number'] { width: 9rem; padding: 0.5rem 0.7rem; border-radius: 10px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-size: 1rem; }
  textarea { width: 100%; padding: 0.6rem 0.8rem; border-radius: 10px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-size: 0.95rem; resize: vertical; margin-bottom: 0.4rem; }
  .unit, .hint { color: var(--ink-3); font-size: 0.85rem; }
  .reveal { margin-top: 1rem; padding-top: 0.9rem; border-top: 1px dashed var(--line-strong); animation: rise 0.35s ease both; }
  .verdict { font-weight: 700; margin-bottom: 0.4rem; color: var(--pain); }
  .verdict.ok { color: var(--good); }
  .rbody { font-family: var(--font-body); font-size: 1.05rem; }
  .rbody :global(p:last-child) { margin-bottom: 0; }
  @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
</style>
