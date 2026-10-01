<script>
  import { onMount } from 'svelte';
  import { mdl, M, loadModel } from '../lib/model.svelte.js';
  import { href } from '../lib/router.svelte.js';
  import { byNum } from '../chapters/index.js';
  import { LAYERS, NAMES, buildSnapshot, captionFor } from '../lib/xfmr3d-data.js';
  import { XfmrScene } from '../lib/xfmr3d.js';

  // The real, trained microgpt as a stack you can turn. Which stage is lit comes from a slow tour, from hovering,
  // or from tapping a stage; the numbers in the picture and in the caption are measured from an actual run.
  let canvas = $state(null);
  let scene = null;
  let ready = $state(false);
  let failed = $state(false);
  let stepIdx = $state(0);
  let nameIdx = $state(0);
  let active = $state(0);
  let pinned = $state(false);
  let steps = $state([]);
  let snap = $state.raw(null);

  const reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fmt = (n) => n.toLocaleString('en-US');

  onMount(() => {
    let dead = false;
    loadModel().then(() => {
      if (dead) return;
      steps = M.steps; stepIdx = M.steps.length - 1;
      ready = true;
    }).catch(() => { failed = true; });
    return () => { dead = true; scene?.destroy(); scene = null; };
  });

  // Build the scene once the canvas and the model exist, then re-run the real model whenever the training step or
  // the name changes and let the picture glide to the new numbers.
  $effect(() => {
    if (!ready || !canvas || mdl.status !== 'ready') return;
    scene ??= new XfmrScene(canvas, { reduced, onActive: (i, p) => { active = i; pinned = p; } });
    const m = M.models[M.steps[stepIdx]];
    let s;
    try { s = buildSnapshot(m, NAMES[nameIdx]); } catch { failed = true; return; }
    snap = s;
    scene.setSnapshot(s);
  });

  const layer = $derived(LAYERS[active]);
  const chapter = $derived(byNum(layer.chapter));
  const caption = $derived(snap ? captionFor(layer.id, snap) : '');
</script>

{#if !failed}
  <div class="viz" role="group" aria-label="Interactive 3D model of the real trained transformer"
    onpointerenter={() => scene?.pauseTour(true)} onpointerleave={() => scene?.pauseTour(false)}
    onfocusin={() => scene?.pauseTour(true)} onfocusout={() => scene?.pauseTour(false)}>
    <div class="stage">
      <canvas bind:this={canvas} tabindex="0" aria-label="3D model of microgpt, from letters in at the bottom to next-letter odds at the top. Use the arrow keys to step through its stages."></canvas>
      {#if !ready}<div class="wait ui muted">Loading the real model…</div>{/if}
      <button class="chip name ui" onclick={() => (nameIdx = (nameIdx + 1) % NAMES.length)} aria-label="Try a different name" title="Try a different name">“{NAMES[nameIdx]}” ↻</button>
    </div>

    <div class="cap ui" aria-live="off">
      <div class="ct"><span class="num">{active + 1}</span> {layer.name}{#if pinned}<button class="resume" onclick={() => scene?.pin(-1)}>▶ resume tour</button>{/if}</div>
      <p>{caption}</p>
      <a class="more" href={href(`ch/${chapter.num}`)}>microgpt.py lines {layer.lines} · Chapter {chapter.num}, {chapter.title} →</a>
    </div>

    <div class="ctl ui">
      <div class="grp" role="group" aria-label="Which point in training to show">
        <span class="lbl">Training step</span>
        {#each steps as st, i}
          <button class="chip" class:on={stepIdx === i} aria-pressed={stepIdx === i} onclick={() => (stepIdx = i)}>{fmt(st)}</button>
        {/each}
      </div>
    </div>
    <div class="hint ui muted">Drag to turn it · hover or tap a stage · the dot is the letter being predicted</div>
  </div>
{/if}

<style>
  .viz { display: flex; flex-direction: column; gap: 0.7rem; min-width: 0; }
  .stage { position: relative; border: 1px solid var(--line); border-radius: var(--radius); background: color-mix(in srgb, var(--surface) 55%, transparent); overflow: hidden; }
  canvas { display: block; width: 100%; height: clamp(420px, 120vw, 500px); touch-action: pan-y; cursor: grab; }
  canvas:active { cursor: grabbing; }
  canvas:focus-visible { outline-offset: -3px; border-radius: var(--radius); }
  .wait { position: absolute; inset: 0; display: grid; place-items: center; font-size: 0.85rem; }
  .cap { min-height: 8.6rem; }
  .ct { font-weight: 700; font-size: 0.95rem; display: flex; align-items: center; gap: 0.5rem; }
  .num { display: inline-grid; place-items: center; width: 1.35rem; height: 1.35rem; border-radius: 50%; background: var(--accent-wash); color: var(--accent-strong); font-size: 0.74rem; }
  .cap p { margin: 0.35rem 0 0.4rem; font-family: var(--font-body); font-size: 0.97rem; line-height: 1.45; color: var(--ink-2); }
  .resume { margin-left: auto; border: 0; background: none; color: var(--accent-strong); font-size: 0.74rem; font-weight: 500; padding: 0.1rem 0.3rem; }
  .resume:hover { text-decoration: underline; }
  .more { font-size: 0.78rem; text-decoration: none; font-weight: 500; }
  .more:hover { text-decoration: underline; }
  .ctl { display: flex; flex-wrap: wrap; gap: 0.5rem 0.8rem; align-items: center; }
  .grp { display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center; }
  .lbl { font-size: 0.74rem; color: var(--ink-3); margin-right: 0.15rem; }
  .chip { border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); border-radius: 999px; padding: 0.22rem 0.65rem; font-size: 0.76rem; white-space: nowrap; }
  .chip:hover { border-color: var(--ink-3); }
  .chip.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .chip.name { position: absolute; top: 0.6rem; left: 0.6rem; font-family: var(--font-mono); color: var(--ink); background: color-mix(in srgb, var(--surface) 85%, transparent); }
  .hint { font-size: 0.72rem; }
</style>
