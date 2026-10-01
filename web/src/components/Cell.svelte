<script>
  import { onMount, onDestroy, getContext } from 'svelte';
  import { EditorView, keymap, lineNumbers } from '@codemirror/view';
  import { EditorState } from '@codemirror/state';
  import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
  import { python } from '@codemirror/lang-python';
  import { HighlightStyle, syntaxHighlighting, indentOnInput, bracketMatching } from '@codemirror/language';
  import { tags as t } from '@lezer/highlight';
  import { py } from '../lib/py.svelte.js';
  import { progress } from '../lib/progress.svelte.js';
  import Recipe from './Recipe.svelte';
  import GraphView from '../widgets/GraphView.svelte';

  // A real Python editor that runs in the page. With `check`, it also grades itself:
  // the check is Python that runs in the same namespace and raises AssertionError("helpful message") if you're off.
  let { id, code, plain = null, check = '', pre = '', allowError = false, hint = '', solution = '', title = 'Your turn', rows = 6, children } = $props();

  const beat = getContext('beat');
  let host;
  let view;
  let out = $state('');
  let error = $state('');
  let running = $state(false);
  let passed = $state(!!progress.solved[id]);
  let checkMsg = $state('');
  let attempts = $state(0);
  let showHint = $state(false);
  // Two routes through every exercise: an English recipe (default) or real Python. Both run the same program and the same check.
  let mode = $state(plain ? progress.mode : 'code');
  function setMode(m) { mode = m; progress.mode = m; if (m === 'code') setTimeout(() => view?.requestMeasure(), 0); }

  onMount(() => {
    view = new EditorView({
      parent: host,
      state: EditorState.create({
        doc: progress.code[id] ?? code,
        extensions: [
          lineNumbers(), history(), indentOnInput(), bracketMatching(), python(),
          syntaxHighlighting(hl), theme,
          keymap.of([{ key: 'Mod-Enter', run: () => { run(); return true; } }, { key: 'Shift-Enter', run: () => { run(); return true; } }, indentWithTab, ...defaultKeymap, ...historyKeymap]),
          EditorView.contentAttributes.of({ 'aria-label': 'Python code editor' }),
        ],
      }),
    });
    if (passed) beat?.complete();
    setTimeout(() => py.warm(), 600);
  });
  onDestroy(() => view?.destroy());

  const hl = HighlightStyle.define([
    { tag: [t.keyword, t.controlKeyword, t.operatorKeyword, t.definitionKeyword, t.moduleKeyword], color: 'var(--syn-keyword)' },
    { tag: [t.string, t.special(t.string)], color: 'var(--syn-string)' },
    { tag: [t.number, t.bool, t.null], color: 'var(--syn-number)' },
    { tag: t.comment, color: 'var(--syn-comment)', fontStyle: 'italic' },
    { tag: [t.function(t.variableName), t.function(t.propertyName), t.className], color: 'var(--syn-func)' },
  ]);

  const theme = EditorView.theme({
    '&': { backgroundColor: 'var(--code-bg)', color: 'var(--ink)', fontSize: '0.86rem', borderRadius: '10px', maxWidth: '100%' },
    '.cm-scroller': { overflow: 'auto' },
    '.cm-content': { fontFamily: 'var(--font-mono)', padding: '10px 0', caretColor: 'var(--ink)', minHeight: `calc(${rows} * 1.55em)` },
    '.cm-line': { padding: '0 12px' },
    '.cm-gutters': { backgroundColor: 'transparent', color: 'var(--ink-3)', border: 'none', fontFamily: 'var(--font-mono)' },
    '.cm-activeLine': { backgroundColor: 'transparent' },
    '&.cm-focused': { outline: '2px solid var(--accent)', outlineOffset: '1px' },
    '&.cm-focused .cm-cursor': { borderLeftColor: 'var(--ink)' },
    '.cm-selectionBackground, &.cm-focused .cm-selectionBackground': { backgroundColor: 'var(--accent-wash) !important' },
    '.cm-matchingBracket': { backgroundColor: 'var(--accent-wash)', outline: '1px solid var(--accent)' },
  });

  // Lines printed as "@@GRAPH {json}" are turned into pictures (see show(...) in the engine exercises).
  const parts = $derived.by(() => {
    const text = [], graphs = [];
    for (const l of out.split('\n')) {
      if (l.startsWith('@@GRAPH ')) { try { graphs.push(JSON.parse(l.slice(8))); } catch { text.push(l); } } else text.push(l);
    }
    return { text: text.join('\n').replace(/\n+$/, ''), graphs };
  });

  const lastLine = (e) => String(e).trim().split('\n').pop();

  // One execution path for both routes: hidden prerequisites, then the program, then the (hidden) check.
  async function execute(src) {
    if (running) return;
    running = true; out = ''; error = ''; checkMsg = '';
    if (pre) { // hidden, idempotent prerequisites, so this works even after a page reload
      const r0 = await py.run(pre, () => {});
      if (!r0.ok) { error = 'Setup failed:\n' + r0.error; running = false; return; }
    }
    const r = await py.run(src, (text) => { out = (out + text).slice(-6000); });
    if (!r.ok) {
      error = r.error; attempts++;
      if (allowError) { passed = true; progress.solved[id] = true; beat?.complete(); }   // the point of this cell is to watch it fail
    } else if (check) {
      let cout = '';
      const c = await py.run(check, (text) => { cout += text; });
      if (c.ok) { passed = true; checkMsg = cout.trim(); progress.solved[id] = true; beat?.complete(); }
      else { attempts++; checkMsg = ''; error = '✗ ' + lastLine(c.error).replace(/^AssertionError:\s*/, ''); }
    } else {
      passed = true; progress.solved[id] = true; beat?.complete();
    }
    running = false;
  }
  async function run() {
    const src = view.state.doc.toString();
    progress.code[id] = src;
    await execute(src);
  }

  function reset() {
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: code } });
    delete progress.code[id];
    out = ''; error = ''; checkMsg = '';
  }
  function stop() { py.stop(); running = false; error = 'Stopped. (Variables from earlier cells are forgotten; run them again.)'; }
</script>

<div class="cell widget wide" class:passed>
  <div class="widget-title">
    {title}
    {#if passed}<span class="ok">✓ done</span>{/if}
    <span class="sp"></span>
    {#if plain}
      <span class="mode" role="group" aria-label="How do you want to do this exercise?">
        <button class:on={mode === 'plain'} onclick={() => setMode('plain')}>Plain English</button>
        <button class:on={mode === 'code'} onclick={() => setMode('code')}>Python</button>
      </span>
    {:else}<span class="kbd">⌘/Ctrl + Enter to run</span>{/if}
  </div>
  {#if children}<div class="task">{@render children()}</div>{/if}

  {#if plain && mode === 'plain'}
    <Recipe {id} {plain} {running} {passed} onrun={execute} />
  {/if}

  <div class="ed" class:hidden={mode !== 'code'} bind:this={host}></div>

  <div class="bar" class:hidden={mode !== 'code'}>
    {#if !running}
      <button class="btn primary" onclick={run}>▶ Run</button>
    {:else}
      <button class="btn" onclick={stop}>■ Stop</button>
      <span class="status">{py.status === 'loading' ? 'Starting Python… (first run takes a few seconds)' : 'Running…'}</span>
    {/if}
    <button class="btn ghost" onclick={reset}>Reset code</button>
    {#if hint}<button class="btn ghost" onclick={() => (showHint = !showHint)}>{showHint ? 'Hide hint' : 'Hint'}</button>{/if}
  </div>

  {#if showHint && hint && mode === 'code'}<div class="hint">{hint}</div>{/if}

  {#if plain && mode === 'code'}
    <details class="read">
      <summary>Not sure what the code is saying? Read it in plain English</summary>
      <ol>
        {#each plain.lines as l}
          <li style="margin-left:{(l.depth ?? 0) * 1.2}rem">
            {#if l.blank}<span class="you">(this is the line you write: {l.en.charAt(0).toLowerCase() + l.en.slice(1)})</span>{:else}<span>{l.en}</span> <code>{l.py}</code>{/if}
          </li>
        {/each}
      </ol>
    </details>
  {/if}

  {#if parts.text || error || checkMsg}
    <pre class="out" class:err={!!error && !parts.text}>{parts.text}{#if error}<span class="errtxt">{parts.text ? '\n' : ''}{error}</span>{/if}{#if checkMsg}<span class="oktxt">{parts.text ? '\n' : ''}✓ {checkMsg}</span>{/if}</pre>
  {/if}
  {#each parts.graphs.slice(-1) as g}<GraphView graph={g} />{/each}

  {#if solution && attempts >= 2 && !passed && mode === 'code'}
    <details class="sol"><summary>Stuck? Peek at one solution</summary><pre>{solution}</pre></details>
  {/if}
</div>

<style>
  .cell.passed { border-color: var(--good); }
  .ok { color: var(--good); text-transform: none; letter-spacing: 0; font-weight: 700; }
  .sp { flex: 1; }
  .ed.hidden, .bar.hidden { display: none; }
  .mode { display: inline-flex; border: 1px solid var(--line-strong); border-radius: 999px; overflow: hidden; text-transform: none; letter-spacing: 0; }
  .mode button { border: 0; background: var(--surface); color: var(--ink-2); padding: 0.25rem 0.75rem; font-size: 0.78rem; }
  .mode button.on { background: var(--accent); color: var(--on-accent); }
  .read { margin-top: 0.8rem; font-family: var(--font-body); font-size: 0.98rem; }
  .read summary { cursor: pointer; color: var(--ink-2); font-family: var(--font-ui); font-size: 0.85rem; }
  .read ol { margin: 0.6rem 0 0; padding-left: 1.4rem; }
  .read li { margin-bottom: 0.3rem; }
  .read code { font-size: 0.74rem; color: var(--ink-3); background: none; padding: 0 0 0 0.4rem; }
  .you { color: var(--ink-3); font-style: italic; }
  .kbd { text-transform: none; letter-spacing: 0; font-weight: 400; font-size: 0.72rem; }
  .task { font-family: var(--font-body); font-size: 1.02rem; margin-bottom: 0.8rem; }
  .task :global(p:last-child) { margin-bottom: 0; }
  .ed { border-radius: 10px; overflow: hidden; border: 1px solid var(--line); min-width: 0; }
  .bar { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.7rem; flex-wrap: wrap; }
  .status { color: var(--ink-3); font-size: 0.82rem; }
  .hint { margin-top: 0.7rem; padding: 0.6rem 0.8rem; background: var(--gold-wash); border-radius: 10px; font-family: var(--font-body); font-size: 0.98rem; }
  .out { margin: 0.7rem 0 0; padding: 0.7rem 0.9rem; background: var(--code-bg); border-radius: 10px; font-family: var(--font-mono); font-size: 0.82rem; line-height: 1.5; max-height: 22rem; overflow: auto; white-space: pre-wrap; word-break: break-word; }
  .errtxt { color: var(--pain); }
  .oktxt { color: var(--good); font-weight: 600; }
  .sol { margin-top: 0.7rem; }
  .sol summary { cursor: pointer; color: var(--ink-2); }
  .sol pre { background: var(--code-bg); padding: 0.7rem 0.9rem; border-radius: 10px; font-family: var(--font-mono); font-size: 0.82rem; overflow: auto; }
</style>
