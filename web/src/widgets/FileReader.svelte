<script>
  import { getContext } from 'svelte';
  import { LINES } from '../lib/fileMap.js';
  import { NOTES, noteFor } from '../lib/lineNotes.js';
  import { highlight } from '../lib/highlight.js';
  import { chapters } from '../chapters/index.js';
  import { href } from '../lib/router.svelte.js';

  // The whole of microgpt.py, nothing blurred. mode "read": click any line for a plain-English note and the chapter that built it.
  // mode "quiz": you are given a description and must click the line that does it.
  let { mode = 'read' } = $props();
  const beat = getContext('beat');
  const html = LINES.map((l) => highlight(l));
  let sel = $state(mode === 'read' ? 29 : null);
  let filter = $state(null);
  let seen = new Set();

  const QUIZ = [
    { q: 'Which line turns the model\'s 27 scores into probabilities?', ok: [166, [97, 101]], ch: 8 },
    { q: 'Which line adds an attention block\'s answer back onto the lane?', ok: [134], ch: 13 },
    { q: 'Which line steps a dial against its slope?', ok: [181], ch: 15 },
    { q: 'Which line measures how surprised the model was by the right answer?', ok: [168], ch: 3 },
    { q: 'Which line wipes a dial\'s slope, ready for the next step?', ok: [182], ch: 7 },
    { q: 'Which line applies the temperature while writing a name?', ok: [195], ch: 17 },
    { q: 'Which line does the chain rule: adds (local slope × upstream slope) into an ingredient?', ok: [72], ch: 6 },
    { q: 'Which line works out the set of distinct letters in the names?', ok: [24], ch: 1 },
    { q: 'Which line decides that a name being written has ended?', ok: [197, 198], ch: 17 },
    { q: 'Which line creates the table of coordinates for the 27 symbols (and the position and output tables)?', ok: [81], ch: 9 },
  ];
  let qi = $state(0);
  let right = $state(0);
  let feedback = $state(null);
  let tries = $state(0);

  const matches = (ok, n) => ok.some((o) => (Array.isArray(o) ? n >= o[0] && n <= o[1] : o === n));
  const note = $derived(sel === null ? null : noteFor(sel));
  const inRange = (n) => note && n >= note.from && n <= note.to;
  const chOf = (n) => noteFor(n)?.ch;
  const startOf = new Set(NOTES.map((e) => e.from));
  const chTitle = (n) => chapters.find((c) => c.num === n)?.title;

  function pick(n) {
    if (!LINES[n - 1].trim()) return;
    if (mode === 'read') { sel = n; seen.add(noteFor(n)?.title); if (seen.size >= 5) beat?.complete(); return; }
    if (qi >= QUIZ.length) return;
    const Q = QUIZ[qi]; tries++;
    if (matches(Q.ok, n)) { right++; feedback = { good: true, n, text: `Yes: line ${n}. ${noteFor(n)?.title}.` }; }
    else feedback = { good: false, n, text: `That's line ${n}: "${noteFor(n)?.title ?? 'not the one'}". Try again. (Hint: Chapter ${Q.ch}, ${chTitle(Q.ch)}.)` };
  }
  function next() { feedback = null; qi++; if (qi >= QUIZ.length) beat?.complete(); }
</script>

<div class="widget wide fr">
  <div class="widget-title">{mode === 'read' ? 'The whole file · click any line' : 'Find the line'}</div>
  {#if mode === 'quiz'}
    <div class="quiz">
      {#if qi < QUIZ.length}
        <div class="qn">Question {qi + 1} of {QUIZ.length} · {right} found first try or after retries</div>
        <div class="qq">{QUIZ[qi].q}</div>
        {#if feedback}<div class="fb" class:good={feedback.good}>{feedback.text} {#if feedback.good}<button class="btn primary" onclick={next}>{qi + 1 === QUIZ.length ? 'Finish' : 'Next question'} →</button>{/if}</div>{:else}<div class="hint">Click a line in the file below.</div>{/if}
      {:else}
        <div class="fb good">All {QUIZ.length} found. You can navigate the whole file.</div>
      {/if}
    </div>
  {:else}
    <div class="chips">
      <button class="chip" class:on={filter === null} onclick={() => (filter = null)}>all</button>
      {#each [...new Set(NOTES.map((n) => n.ch))].sort((a, b) => a - b) as c}<button class="chip" class:on={filter === c} onclick={() => (filter = filter === c ? null : c)}>Ch {c}</button>{/each}
    </div>
  {/if}

  <div class="grid">
    <div class="code" role="list">
      {#each LINES as l, i}
        {@const n = i + 1}
        <button class="ln" class:sel={mode === 'read' && inRange(n)} class:dim={mode === 'read' && filter !== null && chOf(n) !== filter} class:hit={mode === 'quiz' && feedback && feedback.n === n} class:bad={mode === 'quiz' && feedback && feedback.n === n && !feedback.good} onclick={() => pick(n)} tabindex="-1">
          <span class="num">{n}</span><span class="txt">{@html html[i] || '&nbsp;'}</span>
          {#if mode === 'read' && startOf.has(n)}<span class="tag">Ch {chOf(n)}</span>{/if}
        </button>
      {/each}
    </div>
    {#if mode === 'read'}
      <aside class="note">
        {#if note}
          <div class="nl">lines {note.from}{note.to !== note.from ? `–${note.to}` : ''}</div>
          <h3>{note.title}</h3>
          <p>{note.text}</p>
          <a class="btn" href={href(`ch/${note.ch}`)}>Built in Chapter {note.ch}: {chTitle(note.ch)}</a>
        {:else}<p class="muted">Click a line of code.</p>{/if}
      </aside>
    {/if}
  </div>
</div>

<style>
  .fr { padding-bottom: 0.8rem; }
  .chips { display: flex; gap: 0.3rem; flex-wrap: wrap; margin-bottom: 0.7rem; } .chip { padding: 0.2rem 0.65rem; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); font-size: 0.74rem; } .chip.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .quiz { margin-bottom: 0.8rem; } .qn { color: var(--ink-3); font-size: 0.78rem; } .qq { font-family: var(--font-body); font-size: 1.1rem; margin: 0.3rem 0; } .hint { color: var(--ink-3); font-size: 0.85rem; } .fb { padding: 0.5rem 0.8rem; border-radius: 10px; background: var(--pain-wash); display: flex; gap: 0.8rem; align-items: center; flex-wrap: wrap; } .fb.good { background: var(--good-wash); }
  .grid { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(220px, 1fr); gap: 1rem; align-items: start; } @media (max-width: 900px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .code { max-height: 560px; overflow: auto; background: var(--code-bg); border-radius: 10px; padding: 0.4rem 0; font-family: var(--font-mono); font-size: 0.74rem; line-height: 1.55; }
  .ln { all: unset; box-sizing: border-box; display: flex; gap: 0.7rem; width: 100%; padding: 0 0.7rem 0 0; white-space: pre; cursor: pointer; border-left: 3px solid transparent; position: relative; }
  .ln:hover { background: color-mix(in srgb, var(--accent) 10%, transparent); } .ln.sel { background: var(--accent-wash); border-left-color: var(--accent); } .ln.dim { opacity: 0.25; } .ln.hit { background: var(--good-wash); border-left-color: var(--good); } .ln.hit.bad { background: var(--pain-wash); border-left-color: var(--pain); }
  .num { flex: none; width: 2.6rem; text-align: right; color: var(--ink-3); opacity: 0.7; user-select: none; } .txt { flex: 1; } .tag { position: absolute; right: 0.5rem; top: 0; font-family: var(--font-ui); font-size: 0.62rem; color: var(--accent-strong); font-weight: 700; background: var(--code-bg); padding: 0 0.3rem; border-radius: 4px; }
  .txt :global(i) { font-style: normal; } .txt :global(.k) { color: var(--syn-keyword); } .txt :global(.s) { color: var(--syn-string); } .txt :global(.n) { color: var(--syn-number); } .txt :global(.c) { color: var(--syn-comment); font-style: italic; }
  .note { position: sticky; top: calc(var(--header-h) + 12px); background: var(--surface-2); border-radius: 12px; padding: 0.9rem 1rem; } .nl { font-family: var(--font-mono); font-size: 0.72rem; color: var(--ink-3); } .note h3 { margin: 0.2rem 0 0.4rem; font-size: 1.1rem; } .note p { font-family: var(--font-body); font-size: 0.98rem; }
</style>
