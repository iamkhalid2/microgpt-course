<script>
  import { progress } from '../lib/progress.svelte.js';

  // The plain-English route through an exercise. The learner builds the program as a recipe in English by
  // choosing the right step wherever one is missing. The real Python sits next to each step, so reading it
  // becomes familiar before anyone has to write it. The assembled Python is what actually runs and gets checked.
  let { id, plain, running = false, passed = false, onrun, children } = $props();

  const saved = progress.code[id + ':plain'] ?? {};
  let picks = $state({ ...saved });   // line index -> true once that blank is answered correctly
  let tried = $state({});             // line index -> last wrong option the learner tried
  let showPy = $state(true);

  const blankIdx = plain.lines.map((l, i) => (l.blank ? i : -1)).filter((i) => i >= 0);
  const solved = $derived(blankIdx.every((i) => picks[i]));
  const left = $derived(blankIdx.filter((i) => !picks[i]).length);
  const code = plain.lines.map((l) => '    '.repeat(l.depth ?? 0) + l.py).join('\n');

  function choose(i, oi) {
    if (oi === plain.lines[i].answer) {
      picks[i] = true;
      delete tried[i];
      progress.code[id + ':plain'] = { ...$state.snapshot(picks) };
    } else tried[i] = oi;
  }
</script>

<div class="recipe">
  <div class="rhead">
    <div class="rt">{plain.intro ?? 'Read the recipe. Where a step is missing, choose the one that makes it work.'}</div>
    <label class="tog"><input type="checkbox" bind:checked={showPy} /> show the Python beside each step</label>
  </div>

  <ol class="steps">
    {#each plain.lines as l, i}
      <li style="--d:{l.depth ?? 0}" class:nested={(l.depth ?? 0) > 0}>
        {#if l.blank && !picks[i]}
          <div class="ask"><span class="q">?</span>{l.ask}</div>
          <div class="opts">
            {#each l.options as o, oi}
              <button class="po" class:bad={tried[i] === oi} onclick={() => choose(i, oi)}>{o}</button>
            {/each}
          </div>
          {#if tried[i] !== undefined}<div class="why">{l.why?.[tried[i]] ?? 'Not quite. Think about what we want at this step and try another.'}</div>{/if}
        {:else}
          <div class="row">
            <div class="en">{l.en}{#if l.blank}<span class="tick" aria-label="correct">✓</span>{/if}</div>
            {#if showPy}<code class="py">{l.py}</code>{/if}
          </div>
        {/if}
      </li>
    {/each}
  </ol>

  <div class="bar">
    <button class="btn primary" disabled={!solved || running} onclick={() => onrun(code)}>
      {running ? 'Running…' : solved ? '▶ Run this recipe' : `Choose the ${left} missing step${left === 1 ? '' : 's'} first`}
    </button>
    {#if passed}<span class="note">This is exactly the program the Python editor would run. Switch to “Python” to try writing it yourself.</span>{/if}
  </div>
</div>

<style>
  .rhead { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.6rem; }
  .rt { color: var(--ink-2); font-size: 0.88rem; }
  .tog { font-size: 0.78rem; color: var(--ink-3); display: flex; gap: 0.35rem; align-items: center; }
  .steps { list-style: none; margin: 0; padding: 0; counter-reset: s; }
  li { position: relative; padding: 0.45rem 0.7rem 0.45rem 2.2rem; margin-bottom: 0.3rem; border-radius: 10px; background: var(--code-bg); counter-increment: s; }
  li::before { content: counter(s); position: absolute; left: 0.7rem; top: 0.55rem; width: 1.15rem; height: 1.15rem; border-radius: 50%; background: var(--surface); color: var(--ink-3); font-size: 0.68rem; font-weight: 700; display: grid; place-items: center; }
  li { margin-left: calc(var(--d, 0) * 1.5rem); }
  @media (max-width: 560px) { li { margin-left: calc(var(--d, 0) * 0.7rem); padding-right: 0.5rem; } }
  li.nested { border-left: 3px solid var(--line-strong); }
  .row { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; flex-wrap: wrap; }
  .en { font-family: var(--font-body); font-size: 1.02rem; }
  .py { font-family: var(--font-mono); font-size: 0.74rem; color: var(--ink-3); background: none; padding: 0; white-space: pre-wrap; word-break: break-word; }
  .tick { color: var(--good); font-weight: 800; margin-left: 0.4rem; }
  .ask { font-family: var(--font-body); font-size: 1.02rem; margin-bottom: 0.45rem; font-weight: 600; }
  .q { display: inline-grid; place-items: center; width: 1.2rem; height: 1.2rem; border-radius: 50%; background: var(--gold-wash); color: var(--gold); border: 1px solid var(--gold); font-size: 0.72rem; margin-right: 0.5rem; font-family: var(--font-ui); }
  .opts { display: grid; gap: 0.35rem; }
  .po { text-align: left; padding: 0.5rem 0.75rem; border-radius: 9px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); font-family: var(--font-body); font-size: 0.98rem; line-height: 1.35; }
  .po:hover { border-color: var(--accent); background: var(--accent-wash); }
  .po.bad { border-color: var(--pain); background: var(--pain-wash); }
  .why { margin-top: 0.4rem; padding: 0.45rem 0.7rem; border-radius: 8px; background: var(--pain-wash); font-family: var(--font-body); font-size: 0.95rem; }
  .bar { margin-top: 0.8rem; display: flex; gap: 0.8rem; align-items: center; flex-wrap: wrap; }
  .note { color: var(--ink-3); font-size: 0.8rem; }
</style>
