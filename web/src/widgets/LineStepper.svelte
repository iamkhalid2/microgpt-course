<script>
  import { getContext } from 'svelte';

  // Step through a tiny program one line at a time, with the plain-English meaning of each line
  // and the "boxes" (variables) it changes. Authored by hand, small enough to verify by eye.
  let { program, steps, title = 'Step through a program' } = $props();
  const beat = getContext('beat');
  let i = $state(0);
  const s = $derived(steps[i]);
  const names = $derived([...new Set(steps.flatMap((x) => Object.keys(x.vars ?? {})))]);
  function go(d) {
    i = Math.max(0, Math.min(steps.length - 1, i + d));
    if (i === steps.length - 1) beat?.complete();
  }
</script>

<div class="widget wide">
  <div class="widget-title">{title}</div>
  <div class="grid">
    <div>
      <div class="code mono" role="img" aria-label="The program">
        {#each program as l, n}
          <div class="ln" class:on={s.line === n}><span class="num">{n + 1}</span><span class="txt" style="padding-left:{(l.depth ?? 0) * 1.6}rem">{l.code}</span></div>
        {/each}
      </div>
      <div class="ctl">
        <button class="btn" onclick={() => go(-1)} disabled={i === 0}>◀ Back</button>
        <button class="btn primary" onclick={() => go(1)} disabled={i === steps.length - 1}>Next line ▶</button>
        <span class="muted">step {i + 1} of {steps.length}</span>
      </div>
    </div>
    <div class="side">
      <div class="says">{s.en}</div>
      <div class="boxes">
        <div class="bt">The boxes (variables) right now</div>
        {#each names as n}
          <div class="box" class:has={s.vars && n in s.vars} class:changed={s.changed === n}>
            <span class="bn mono">{n}</span><span class="bv mono">{s.vars && n in s.vars ? s.vars[n] : '(empty)'}</span>
          </div>
        {/each}
      </div>
      {#if s.out}<div class="out"><span class="bt">Printed</span> <span class="mono">{s.out}</span></div>{/if}
    </div>
  </div>
</div>

<style>
  .grid { display: grid; grid-template-columns: 1.1fr 1fr; gap: 1.3rem; }
  @media (max-width: 820px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .code { background: var(--code-bg); border-radius: 10px; padding: 0.5rem 0; font-size: 0.88rem; overflow-x: auto; }
  .ln { display: flex; gap: 0.8rem; padding: 0.15rem 0.8rem; border-left: 3px solid transparent; white-space: pre; }
  .ln.on { background: var(--accent-wash); border-left-color: var(--accent); }
  .num { color: var(--ink-3); width: 1.2rem; text-align: right; flex: none; }
  .ctl { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.8rem; flex-wrap: wrap; }
  .says { font-family: var(--font-body); font-size: 1.05rem; min-height: 4.6rem; margin-bottom: 0.6rem; }
  .bt { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-3); font-weight: 600; }
  .boxes { display: grid; gap: 0.35rem; margin-top: 0.3rem; }
  .box { display: flex; justify-content: space-between; gap: 1rem; padding: 0.4rem 0.7rem; border-radius: 9px; border: 1px dashed var(--line-strong); color: var(--ink-3); }
  .box.has { border-style: solid; background: var(--surface-2); color: var(--ink); }
  .box.changed { border-color: var(--accent); background: var(--accent-wash); }
  .bn { font-weight: 600; }
  .out { margin-top: 0.7rem; padding: 0.4rem 0.7rem; background: var(--good-wash); border-radius: 9px; display: flex; gap: 0.6rem; align-items: baseline; }
</style>
