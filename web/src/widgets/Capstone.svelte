<script>
  import { getContext, onMount, onDestroy } from 'svelte';
  import { EditorView, keymap, lineNumbers } from '@codemirror/view';
  import { EditorState } from '@codemirror/state';
  import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
  import { python } from '@codemirror/lang-python';
  import { HighlightStyle, syntaxHighlighting, indentOnInput, bracketMatching } from '@codemirror/language';
  import { tags as t } from '@lezer/highlight';
  import { py } from '../lib/py.svelte.js';
  import { progress } from '../lib/progress.svelte.js';
  import { LINES } from '../lib/fileMap.js';

  // The blank page. You write microgpt.py from nothing; five milestone tests check your work as you go.
  // Hints come in three levels: the idea, an outline, and the reference code (the real file's own lines).
  const beat = getContext('beat');
  let host, view;
  const START = `# main.py: your own microgpt, from a blank page.
# Use the same names as the real file (docs, uchars, BOS, vocab_size, Value, state_dict, params,
# linear, softmax, rmsnorm, gpt, num_steps, ...) so the milestone tests can find your work.
`;
  let results = $state({});          // id -> { status: 'idle'|'running'|'pass'|'fail', msg }
  let busy = $state(null);
  let hintFor = $state(null);        // { id, level }
  let finalOut = $state('');
  let finalState = $state('idle');
  let counter = 0;

  const ref = (a, b) => LINES.slice(a - 1, b).join('\n');
  const MILESTONES = [
    { id: 1, title: 'The data', goal: 'Read the names into a shuffled list called docs. Build uchars (the sorted distinct characters), BOS (the next free number) and vocab_size.',
      idea: 'Read every non-empty line of input.txt into a list, and shuffle it (after random.seed(42)). Then find the distinct characters across all the names, sort them, and number them from 0. Give the start/end symbol the next free number, and count the total.',
      outline: '# 1. random.seed(42), then docs = every non-empty line of input.txt, shuffled\n# 2. uchars = the sorted distinct characters of all the names\n# 3. BOS = len(uchars): the next free number\n# 4. vocab_size = len(uchars) + 1',
      ref: ref(9, 27), steps: 0, sampling: false,
      test: `assert len(docs) == 32033, "docs should hold all 32,033 names, but has %d" % len(docs)
assert uchars == sorted(set(''.join(docs))), "uchars should be the sorted distinct characters of the names"
assert BOS == len(uchars) == 26 and vocab_size == 27, "BOS should be 26 and vocab_size 27"` },
    { id: 2, title: 'The engine', goal: 'A Value class that remembers how it was made, with +, *, **, log, exp, relu, negation, subtraction, division, and a backward() that does the chain rule.',
      idea: 'A Value stores data, grad (starting at 0), its children and the local slope for each. Every operation returns a new Value with the right local slopes (add: 1 and 1; multiply: each other\'s data; log: 1/x ...). backward() lines the Values up with children first (a recursive helper and a visited set), sets the final grad to 1, then walks the list in reverse adding local_grad × grad into each child.',
      outline: 'class Value:\n    # __init__: data, grad = 0, _children, _local_grads\n    # __add__, __mul__: wrap plain numbers in Value; return Value(result, (self, other), (local slopes))\n    # __pow__, log, exp, relu: one child each\n    # __neg__, __sub__, __truediv__ and the reversed (__radd__, __rsub__, __rmul__, __rtruediv__) versions\n    # backward: build_topo (recursive), self.grad = 1, then for v in reversed(topo): child.grad += local_grad * v.grad',
      ref: ref(29, 72), steps: 0, sampling: false,
      test: `a, b = Value(2.0), Value(3.0); loss = (a * b + 1) ** 2; loss.backward()
assert (a.grad, b.grad) == (42.0, 28.0), "(a*b + 1)**2 at a=2, b=3 should give slopes 42 and 28, got %r and %r" % (a.grad, b.grad)
x = Value(3.0); y = x * x; y.backward()
assert x.grad == 6.0, "x*x should give x a slope of 6 (two paths add up), got %r" % x.grad
import math
v = Value(0.5); z = v.log() + v.exp() + (v * 2).relu() - 3 / v + 2 * v - (1 - v); z.backward()
want = 1 / 0.5 + math.exp(0.5) + 2 + 3 / 0.25 + 2 + 1
assert abs(v.grad - want) < 1e-9, "log/exp/relu/division/reversed operators: slope should be %.6f, got %.6f" % (want, v.grad)` },
    { id: 3, title: 'The model', goal: 'The dials (state_dict, params), the helpers linear, softmax and rmsnorm, and gpt(token_id, pos_id, keys, values) returning one score per symbol.',
      idea: 'Make tables of small random Values: wte (vocab × 16), wpe (8 × 16), lm_head (vocab × 16), and for the layer: attention q/k/v/o (16 × 16) and MLP fc1 (64 × 16), fc2 (16 × 64). Flatten them into params. gpt embeds the letter and position, normalises, runs an attention block (query, key, value, filing cabinet, heads, mixing, skip lane) and an MLP block (expand, relu squared, shrink, skip lane), then applies lm_head.',
      outline: 'n_embd, n_head, n_layer, block_size = 16, 4, 1, 8\nhead_dim = n_embd // n_head\nmatrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) ...]]\nstate_dict = {wte, wpe, lm_head, layer0.attn_wq/wk/wv/wo (wo and mlp_fc2 start at std=0), layer0.mlp_fc1, layer0.mlp_fc2}\nparams = flat list of every Value\ndef linear(x, w) / def softmax(logits) (subtract the max) / def rmsnorm(x)\ndef gpt(token_id, pos_id, keys, values): embed -> rmsnorm -> [attention block] -> [MLP block] -> lm_head',
      ref: ref(74, 144), steps: 0, sampling: false,
      test: `assert len(params) == 4064, "With n_embd=16, n_head=4, n_layer=1, block_size=8 there should be 4064 dials, but params has %d. (Use the file's sizes for this test.)" % len(params)
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
logits = gpt(BOS, 0, keys, values)
assert len(logits) == vocab_size, "gpt should return %d scores (one per symbol), got %d" % (vocab_size, len(logits))
probs = softmax(logits)
assert abs(sum(p.data for p in probs) - 1) < 1e-9, "softmax outputs should add up to 1"
(-probs[3].log()).backward()
assert any(p.grad != 0 for p in params), "after backward, some dials should have non-zero slopes"` },
    { id: 4, title: 'The training loop', goal: 'Adam, the cosine learning rate, and the loop: pick a name, wrap it, run the model at every position, average the surprises, backward, update every dial, wipe the slopes.',
      idea: 'Keep two lists m and v (one running average per dial). Each step: take docs[step % len(docs)], wrap it with BOS on both sides, run gpt at each position and collect -log(probability of the next symbol), average them, call loss.backward(), then for every dial update m and v, correct them, step the dial by lr_t × m_hat / (sqrt(v_hat) + eps) with lr_t decaying by a cosine, and set its grad back to 0.',
      outline: 'learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8\nm = [0.0] * len(params); v = [0.0] * len(params)\nnum_steps = 500\nfor step in range(num_steps):\n    # doc = docs[step % len(docs)]; tokens = [BOS] + [...] + [BOS]; n = min(block_size, len(tokens) - 1)\n    # forward over every position -> losses; loss = (1 / n) * sum(losses)\n    # loss.backward()\n    # lr_t = learning_rate * 0.5 * (1 + cos(pi * step / num_steps))\n    # for each param: m, v, m_hat, v_hat, p.data -= ..., p.grad = 0\n    # print the step and loss',
      ref: ref(146, 184), steps: 40, sampling: false, train: true },
    { id: 5, title: 'The voice', goal: 'Write names: start at BOS, ask gpt, divide the scores by a temperature, softmax, sample with random.choices, stop at BOS. Print each as "sample N: name".',
      idea: 'For each of 20 samples: start with token_id = BOS and empty key/value cabinets; for each position up to block_size call gpt, divide the logits by the temperature, softmax, and use random.choices to pick the next symbol. Stop if it picks BOS, otherwise append the letter. Print the finished name.',
      outline: 'temperature = 0.5\nfor sample_idx in range(20):\n    keys, values = ...\n    token_id = BOS; sample = []\n    for pos_id in range(block_size):\n        # logits = gpt(...); probs = softmax([l / temperature for l in logits]); token_id = random.choices(...)\n        # if token_id == BOS: break; sample.append(uchars[token_id])\n    print(f"sample {sample_idx+1:2d}: {\'\'.join(sample)}")',
      ref: ref(186, 200), steps: 30, sampling: true,
      test: '' },
  ];

  const theme = EditorView.theme({
    '&': { backgroundColor: 'var(--code-bg)', color: 'var(--ink)', fontSize: '0.84rem', borderRadius: '10px', maxWidth: '100%', height: '520px' },
    '.cm-scroller': { overflow: 'auto', fontFamily: 'var(--font-mono)' }, '.cm-content': { padding: '10px 0', caretColor: 'var(--ink)' }, '.cm-line': { padding: '0 12px' },
    '.cm-gutters': { backgroundColor: 'transparent', color: 'var(--ink-3)', border: 'none', fontFamily: 'var(--font-mono)' }, '.cm-activeLine': { backgroundColor: 'transparent' },
    '&.cm-focused': { outline: '2px solid var(--accent)', outlineOffset: '1px' }, '.cm-selectionBackground, &.cm-focused .cm-selectionBackground': { backgroundColor: 'var(--accent-wash) !important' },
  });
  const hl = HighlightStyle.define([
    { tag: [t.keyword, t.controlKeyword, t.operatorKeyword, t.definitionKeyword, t.moduleKeyword], color: 'var(--syn-keyword)' }, { tag: [t.string, t.special(t.string)], color: 'var(--syn-string)' },
    { tag: [t.number, t.bool, t.null], color: 'var(--syn-number)' }, { tag: t.comment, color: 'var(--syn-comment)', fontStyle: 'italic' }, { tag: [t.function(t.variableName), t.className], color: 'var(--syn-func)' },
  ]);
  let saveTimer;
  onMount(() => {
    view = new EditorView({ parent: host, state: EditorState.create({ doc: progress.code.capstone ?? START, extensions: [
      lineNumbers(), history(), indentOnInput(), bracketMatching(), python(), syntaxHighlighting(hl), theme,
      keymap.of([indentWithTab, ...defaultKeymap, ...historyKeymap]),
      EditorView.updateListener.of((u) => { if (u.docChanged) { clearTimeout(saveTimer); saveTimer = setTimeout(() => (progress.code.capstone = view.state.doc.toString()), 400); } }),
      EditorView.contentAttributes.of({ 'aria-label': 'Your microgpt, as Python code' }),
    ] }) });
    for (const m of MILESTONES) if (progress.solved['capstone-' + m.id]) results[m.id] = { status: 'pass', msg: 'passed earlier' };
    py.warm();
  });
  onDestroy(() => { view?.destroy(); clearTimeout(saveTimer); });

  const doc = () => view.state.doc.toString();
  function insert(text, title) {
    const cur = doc();
    view.dispatch({ changes: { from: cur.length, insert: `${cur.endsWith('\n') ? '' : '\n'}\n# ---- ${title} ----\n${text}\n` }, selection: { anchor: cur.length } });
  }
  const tail = (e) => String(e).trim().split('\n').slice(-3).join('\n');
  function mod(src, steps, sampling) {
    let out = src.replace(/num_steps\s*=\s*\d+/, `num_steps = ${steps}`);
    out = out.replace(/range\(20\)/, sampling ? 'range(6)' : 'range(0)');
    return out;
  }
  async function exec(src, tailCode = '') {
    const session = 'cap-' + ++counter;
    let out = '';
    const r = await py.run(src + '\n' + tailCode, (x) => (out += x), { session });
    py.drop(session);
    return { r, out };
  }
  const EVAL = `
def _eval():
    tot, n = 0.0, 0
    for d in docs[1000:1030]:
        toks = [BOS] + [uchars.index(c) for c in d] + [BOS]
        ks, vs = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        for p in range(min(block_size, len(toks) - 1)):
            tot += -softmax(gpt(toks[p], p, ks, vs))[toks[p + 1]].log().data
            n += 1
    return tot / n
print('@@E', _eval())
`;
  async function check(m) {
    if (busy) return;
    busy = m.id; results[m.id] = { status: 'running', msg: 'Running your code…' };
    const src = doc();
    try {
      if (m.id === 4) {
        const a = await exec(mod(src, 0, false), EVAL);
        if (!a.r.ok) throw new Error(tail(a.r.error));
        const b = await exec(mod(src, 40, false), EVAL);
        if (!b.r.ok) throw new Error(tail(b.r.error));
        const ev = (o) => parseFloat((o.match(/@@E ([\d.eE+-]+)/) ?? [])[1]);
        const before = ev(a.out), after = ev(b.out);
        if (!isFinite(before) || !isFinite(after)) throw new Error('Could not measure the loss. Does your file define gpt, softmax, docs, uchars, BOS, block_size and n_layer, and a loop over num_steps?');
        if (!(after < before - 0.15)) throw new Error(`After 40 training steps the loss only went from ${before.toFixed(2)} to ${after.toFixed(2)}. It should fall by more than 0.15. Check the direction of the Adam step, and that grads are reset.`);
        results[m.id] = { status: 'pass', msg: `Trained 40 steps: loss fell from ${before.toFixed(2)} to ${after.toFixed(2)}.` };
      } else if (m.id === 5) {
        const a = await exec(mod(src, 30, true));
        if (!a.r.ok) throw new Error(tail(a.r.error));
        const lines = a.out.split('\n').filter((l) => /^sample\s+\d+:/.test(l));
        if (lines.length < 5) throw new Error(`I expected lines like "sample  1: name", but found ${lines.length}. Print each generated name in that format.`);
        if (lines.filter((l) => l.replace(/^sample\s+\d+:\s*/, '').length >= 1).length < 3) throw new Error('Most of the samples are empty. Does each name start from BOS, and does the loop append letters?');
        results[m.id] = { status: 'pass', msg: `Wrote ${lines.length} samples after 30 steps, e.g. ${lines.slice(0, 3).map((l) => l.replace(/^sample\s+\d+:\s*/, '') || '·').join(', ')}.` };
      } else {
        const code = `${m.test.split('\n').map((l) => '    ' + l).join('\n')}`;
        const a = await exec(mod(src, 0, false), `try:\n${code}\n    print('@@OK')\nexcept AssertionError as _e:\n    print('@@FAIL', _e)\nexcept NameError as _e:\n    print('@@FAIL', 'a name is missing: ' + str(_e))`);
        if (!a.r.ok) throw new Error(tail(a.r.error));
        if (a.out.includes('@@OK')) results[m.id] = { status: 'pass', msg: 'All checks passed.' };
        else throw new Error((a.out.match(/@@FAIL (.*)/) ?? [])[1] ?? 'The checks did not pass.');
      }
      progress.solved['capstone-' + m.id] = true;
    } catch (e) {
      results[m.id] = { status: 'fail', msg: String(e.message ?? e) };
    }
    busy = null;
    if (MILESTONES.every((x) => results[x.id]?.status === 'pass')) beat?.complete();
  }
  async function runAll() {
    if (busy) return;
    busy = 'final'; finalState = 'running'; finalOut = '';
    const r = await py.run(doc(), (x) => (finalOut = (finalOut + x).slice(-5000)), { session: 'cap-final' });
    py.drop('cap-final');
    finalState = r.ok ? 'done' : 'error';
    if (!r.ok) finalOut += '\n' + tail(r.error);
    busy = null;
  }
  const allPass = $derived(MILESTONES.every((x) => results[x.id]?.status === 'pass'));
  const icon = { idle: '○', running: '…', pass: '✓', fail: '✗' };
</script>

<div class="widget wide cap">
  <div class="widget-title">The blank page · rebuild microgpt.py</div>
  <div class="layout">
    <div class="ms">
      {#each MILESTONES as m}
        {@const r = results[m.id] ?? { status: 'idle' }}
        <div class="m" class:pass={r.status === 'pass'} class:fail={r.status === 'fail'}>
          <div class="mh"><span class="ic">{icon[r.status]}</span><strong>{m.id}. {m.title}</strong></div>
          <div class="goal">{m.goal}</div>
          <div class="acts">
            <button class="btn primary" onclick={() => check(m)} disabled={busy !== null}>{r.status === 'running' ? 'Checking…' : 'Check it'}</button>
            <button class="btn ghost" onclick={() => (hintFor = hintFor?.id === m.id ? null : { id: m.id, level: 1 })}>Hints</button>
          </div>
          {#if r.msg}<div class="msg" class:ok={r.status === 'pass'}>{r.msg}</div>{/if}
          {#if hintFor?.id === m.id}
            <div class="hints">
              <div class="lv">
                <button class:on={hintFor.level === 1} onclick={() => (hintFor.level = 1)}>1 · The idea</button>
                <button class:on={hintFor.level === 2} onclick={() => (hintFor.level = 2)}>2 · An outline</button>
                <button class:on={hintFor.level === 3} onclick={() => (hintFor.level = 3)}>3 · Show me</button>
              </div>
              {#if hintFor.level === 1}<p>{m.idea}</p>
              {:else if hintFor.level === 2}<pre>{m.outline}</pre><button class="btn" onclick={() => insert(m.outline.split('\n').map((l) => (l.startsWith('#') || l.startsWith(' ') || l === '' ? l : '# ' + l)).join('\n'), 'outline: ' + m.title)}>Add as comments to my file</button>
              {:else}<p class="muted">This is the real file's own code for this part. Reading it is fine; typing it out yourself is better.</p><pre class="ref">{m.ref}</pre><button class="btn" onclick={() => insert(m.ref, m.title)}>Add to my file</button>{/if}
            </div>
          {/if}
        </div>
      {/each}
    </div>
    <div class="ed">
      <div bind:this={host}></div>
      <div class="foot">
        <button class="btn primary" onclick={runAll} disabled={busy !== null}>{finalState === 'running' ? 'Running your file…' : '▶ Run my whole file'}</button>
        <span class="muted">Runs exactly what is in the editor. With num_steps = 500 expect about a minute.</span>
      </div>
      {#if finalOut}<pre class="out">{finalOut}</pre>{/if}
    </div>
  </div>
  {#if allPass}<div class="win"><strong>All five milestones pass.</strong> You rebuilt a GPT from a blank page. Press “Run my whole file” to watch your own version train and write names.</div>{/if}
</div>

<style>
  .layout { display: grid; grid-template-columns: minmax(250px, 0.9fr) minmax(0, 1.6fr); gap: 1.1rem; align-items: start; } @media (max-width: 960px) { .layout { grid-template-columns: minmax(0, 1fr); } }
  @media (min-width: 961px) { .ed { position: sticky; top: calc(var(--header-h) + 12px); } }
  .m { border: 1px solid var(--line); border-radius: 12px; padding: 0.7rem 0.8rem; margin-bottom: 0.6rem; background: var(--surface); } .m.pass { border-color: var(--good); } .m.fail { border-color: var(--pain); }
  .mh { display: flex; gap: 0.5rem; align-items: baseline; } .ic { width: 1.1rem; font-weight: 800; } .m.pass .ic { color: var(--good); } .m.fail .ic { color: var(--pain); }
  .goal { color: var(--ink-2); font-size: 0.82rem; margin: 0.3rem 0 0.5rem; } .acts { display: flex; gap: 0.4rem; } .msg { margin-top: 0.5rem; font-size: 0.8rem; padding: 0.4rem 0.6rem; border-radius: 8px; background: var(--pain-wash); white-space: pre-wrap; } .msg.ok { background: var(--good-wash); }
  .hints { margin-top: 0.6rem; border-top: 1px dashed var(--line-strong); padding-top: 0.5rem; font-size: 0.82rem; } .lv { display: flex; gap: 0.3rem; flex-wrap: wrap; margin-bottom: 0.4rem; } .lv button { border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); border-radius: 999px; padding: 0.15rem 0.6rem; font-size: 0.72rem; } .lv button.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .hints pre { background: var(--code-bg); border-radius: 8px; padding: 0.5rem 0.6rem; font-family: var(--font-mono); font-size: 0.68rem; overflow: auto; max-height: 220px; white-space: pre; } .hints p { margin: 0 0 0.4rem; font-family: var(--font-body); font-size: 0.95rem; }
  .foot { display: flex; gap: 0.7rem; align-items: center; margin-top: 0.6rem; flex-wrap: wrap; } .out { margin-top: 0.6rem; background: var(--code-bg); border-radius: 10px; padding: 0.6rem 0.8rem; font-family: var(--font-mono); font-size: 0.74rem; max-height: 260px; overflow: auto; white-space: pre-wrap; }
  .win { margin-top: 0.8rem; padding: 0.7rem 1rem; background: var(--good-wash); border-radius: 12px; }
</style>
