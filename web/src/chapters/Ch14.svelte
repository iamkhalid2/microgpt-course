<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import Predict from '../components/Predict.svelte';
  import Cell from '../components/Cell.svelte';
  import Bank from '../components/Bank.svelte';
  import BlockDiagram from '../widgets/BlockDiagram.svelte';
  import ParamCount from '../widgets/ParamCount.svelte';
  import PipelineX from '../widgets/PipelineX.svelte';
  import FilePeek from '../widgets/FilePeek.svelte';
  import { GPTPRE } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const gptCode = `# (linear, softmax and rmsnorm are the recipes from earlier chapters; W holds the real trained dials)

def gpt(token_id, pos_id, keys, values):
    x = [t + p for t, p in zip(W['wte'][token_id], W['wpe'][pos_id])]       # letter + position
    x = rmsnorm(x)

    # ---- attention block ----
    x_residual = x
    x = rmsnorm(x)
    q = linear(x, W['layer0.attn_wq'])
    k = linear(x, W['layer0.attn_wk'])
    v = linear(x, W['layer0.attn_wv'])
    keys.append(k)
    values.append(v)
    x_attn = []
    for h in range(n_head):
        hs = h * head_dim
        q_h = q[hs:hs + head_dim]
        k_h = [ki[hs:hs + head_dim] for ki in keys]
        v_h = [vi[hs:hs + head_dim] for vi in values]
        scores = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim ** 0.5 for t in range(len(k_h))]
        weights = softmax(scores)
        x_attn.extend([sum(weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)])
    x = linear(x_attn, W['layer0.attn_wo'])
    x = [None for a, b in zip(x, x_residual)]        # <- replace None: put the block's answer back on the lane

    # ---- MLP block ----
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, W['layer0.mlp_fc1'])
    x = [max(0, v) ** 2 for v in x]
    x = linear(x, W['layer0.mlp_fc2'])
    x = [None for a, b in zip(x, x_residual)]        # <- replace None: the same again

    return linear(x, W['lm_head'])                    # 16 numbers -> 27 scores

def next_probs(prefix):
    ids = [BOS] + [cfg['uchars'].index(c) for c in prefix]      # BOS, then the letters so far
    keys, values = [], []                                        # the filing cabinet of earlier positions
    for pos, t in enumerate(ids):
        logits = gpt(t, pos, keys, values)                       # read the symbols one at a time
    return softmax(logits)

probs = next_probs('ka')
top = sorted(range(len(probs)), key=lambda i: -probs[i])[:3]
print([(cfg['uchars'][i] if i < 26 else 'end', round(probs[i], 2)) for i in top])
`;
  const gptCheck = `ref = softmax(_D['ref']['ka'])
got = next_probs('ka')
assert len(got) == 27 and abs(sum(got) - 1) < 1e-9, "next_probs should return 27 probabilities that add up to 1."
assert all(abs(a - b) < 2e-3 for a, b in zip(got, ref)), "Your model's predictions for 'ka' don't match the real one. Check that both blocks ADD their answer to the lane."
top1 = max(range(27), key=lambda i: got[i])
assert cfg['uchars'][top1] == 'r', "The most likely letter after 'ka' should be r."
print("This is the real model, in about 40 lines of plain Python, and it agrees with the original to 3 decimal places.")`;
  const gptSolution = `x = [a + b for a, b in zip(x, x_residual)]        # (both places)`;
</script>

<Lesson id="ch14" num={14} title="Assembly" tagline="Every part is on the table. Put them together and count: there are exactly 4,064 dials.">
  <Beat>
    <Pain title="Pieces on the table aren't a model">
      <p>Think of a product assembled from components. You have a <strong>bill of materials</strong>, and at this point you can explain every item on it. Here's yours:</p>
    </Pain>
    <table class="tbl">
      <thead><tr><td>Part</td><td>What it does</td><td>Built in</td></tr></thead>
      <tbody>
        <tr><td>Token and position tables</td><td>give each letter and place coordinates</td><td>Chapter 9</td></tr>
        <tr><td>Linear layers</td><td>turn one list into another with learnable weights</td><td>Chapter 9</td></tr>
        <tr><td>Attention, several heads</td><td>let a position gather what it needs from earlier ones</td><td>Chapters 10–11</td></tr>
        <tr><td>MLP with a ReLU gate</td><td>lets each position think about what it gathered</td><td>Chapter 12</td></tr>
        <tr><td>Skip lanes and RMSNorm</td><td>keep signals and slopes alive</td><td>Chapter 13</td></tr>
        <tr><td>Softmax and the loss</td><td>turn scores into probabilities and grade them</td><td>Chapters 3 and 8</td></tr>
        <tr><td>The autograd engine</td><td>works out every dial's slope</td><td>Chapters 6–7</td></tr>
      </tbody>
    </table>
    <p>What's left is to <strong>wire them together</strong> in the right order. In <code>microgpt.py</code> that wiring is a single function called <code>gpt()</code> (lines 108–144), and it does just one thing: <em>read one symbol, at one position, and return 27 scores for what comes next.</em></p>
  </Beat>

  <Beat>
    <h2>The path of one letter</h2>
    <BlockDiagram lit={['emb', 'norm0', 'attn', 'mlp', 'head', 'skip']} title="gpt(): one symbol in, 27 scores out" />
    <ol class="steps">
      <li><strong>Embed:</strong> look up the letter's 16 coordinates and the position's 16 coordinates, and add them. Normalise.</li>
      <li><strong>Attention block:</strong> normalise a copy, make a query, key and value, file the key and value in the cabinet, let each of the 4 heads look back through the cabinet and blend, mix the heads' answers, and <em>add</em> the result back to the lane.</li>
      <li><strong>MLP block:</strong> normalise a copy, expand to 64, gate, shrink to 16, and <em>add</em> the result back to the lane.</li>
      <li><strong>Output head:</strong> one last linear layer turns the 16 numbers into 27 scores. Softmax (outside the function) turns those into probabilities.</li>
    </ol>
  </Beat>

  <Beat gate>
    <h2>Counting the dials</h2>
    <p>Every table in that picture is full of dials. Let's count them. Here's a calculator: change the design and watch the total, and notice where the dials live.</p>
    <Predict id="ch14-layer" kind="number" answer={3072} tolerance={0}>
      {#snippet q()}<p>Before you play: with 16 numbers per letter (<code>n_embd = 16</code>), <strong>how many dials are in one layer's tables?</strong> There are four attention tables (query, key, value, output), each 16 × 16, plus two MLP tables: one expanding 16 → 64 and one shrinking 64 → 16.</p>{/snippet}
      <p>Four attention tables: 4 × (16 × 16) = <strong>1,024</strong>. The MLP's expand table is 64 × 16 = 1,024 and its shrink table is 16 × 64 = 1,024: another <strong>2,048</strong>. Total <strong>3,072</strong> per layer. Add the letter table (27 × 16 = 432), the position table (8 × 16 = 128) and the output head (27 × 16 = 432), and you get 432 + 128 + 432 + 3,072 = <strong>4,064</strong>: exactly the number the file prints.</p>
    </Predict>
    <ParamCount />
  </Beat>

  <Beat>
    <Aha title="Where the dials are, and what that means">
      <p>In this tiny model, <strong>three quarters of all dials</strong> (3,072 of 4,064) live in the transformer layer's tables. In a real language model the letter tables are bigger, because the vocabulary has tens of thousands of entries, but the layers dominate even more as they get wider, because they grow with the <em>square</em> of the width. And look at the last button in the calculator: this same recipe, at the sizes of a model like GPT-2 small, holds around <strong>163 million</strong> dials. Same 40 lines of logic; only the numbers in lines 75 to 78 change.</p>
    </Aha>
    <p>Now the code. Lines 80 to 90 of <code>microgpt.py</code> create all those tables: a Python <em>dictionary</em> called <code>state_dict</code> that maps a name (<code>'wte'</code>, <code>'layer0.attn_wq'</code>, …) to a table. Every number in it is a <code>Value</code> card from Chapter 7, so the autograd engine can track its slope, and <code>params</code> (line 89) is just a flat list of all 4,064 of those cards. That list is what the optimiser will walk through in a moment.</p>
  </Beat>

  <Beat gate>
    <h2>X-ray the real model</h2>
    <p>Here is the real, trained model reading a name, one letter at a time. Each strip is one of the 16-number lists from the diagram, as a row of coloured cells (orange for positive numbers, blue for negative). Click through the letters of a name, and watch the lane change as each block adds its edit. At the bottom: the probabilities it predicts for the next symbol.</p>
    <PipelineX />
  </Beat>

  <Beat gate>
    <h2>Assemble it yourself</h2>
    <p>Now the real thing, in code you can run. This is the whole <code>gpt()</code> function in plain Python, using the trained dials, plus a helper that feeds a name through it one symbol at a time. You've built every line. <strong>Two steps are missing</strong>, and they're the same step twice: how each block's answer rejoins the lane.</p>
    <Cell id="ch14-gpt" plain={PLAIN['ch14-gpt']} title="The whole model, in about 40 lines" code={gptCode} pre={GPTPRE} rows={48} check={gptCheck} solution={gptSolution}
      hint="Chapter 13: each block's answer is ADDED to what the lane held before the block. The old lane is saved in x_residual; the block's answer is in x.">
      <p>Fill in both skip-lane additions, then run it. The page checks that your model's predictions match the real one's.</p>
    </Cell>
  </Beat>

  <Beat>
    <Aha title="That's the whole model">
      <p>Look at what you just ran: a function of about 35 lines, plus a few kilobytes of numbers, that predicts the next letter of a name better than a counting table can. There's no hidden step. Everything a GPT does, a bigger GPT does too: more layers, wider lists, a bigger vocabulary, more training. The architecture is these lines, repeated.</p>
    </Aha>
    <FilePeek />
    <p>Open the file: lines 74 to 144 are now all yours.</p>
    <Sceptic q="Why does gpt() read one symbol at a time, instead of the whole name at once?">
      <p>It's the simplest way to write it, and it mirrors how a name is <em>generated</em>: one symbol at a time, each depending on the ones before. The filing cabinet of keys and values (<code>keys</code>, <code>values</code>) is how earlier symbols stay available. Production systems process a whole name in parallel on a GPU, using a mask so each position can't see later ones, which is much faster but computes the same answers. In training, <code>microgpt.py</code> runs the function once per position and adds up the losses, and you'll see that next.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <Pain title="We have a model. It knows nothing.">
      <p>Until now, the real dials we've been looking at were already <em>trained</em>. But the file starts with <strong>random</strong> dials (Chapter 9), and a model with random dials writes noise. Training them takes more than the plain gradient descent of Chapter 5: the loss surface of a 4,064-dial model is long, narrow and noisy, and simple downhill steps zigzag and stall. Act IV turns the model from a random contraption into something that writes names. It starts with the most famous optimiser in modern AI.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch14">
      <ul>
        <li><strong>gpt()</strong> reads one symbol at one position and returns 27 scores: embed → attention block → MLP block → output head (lines 108–144).</li>
        <li>Both blocks have a <strong>normalised copy going in</strong> and <strong>add their answer to the lane</strong> coming out.</li>
        <li>The model's design is a handful of numbers (lines 75–78): <code>n_embd = 16</code>, <code>n_head = 4</code>, <code>n_layer = 1</code>, <code>block_size = 8</code>.</li>
        <li><strong>4,064 dials</strong> = 432 (letters) + 128 (positions) + 432 (output) + 3,072 (one layer: 1,024 attention + 2,048 MLP). Heads cost nothing extra.</li>
        <li>The dials live in <code>state_dict</code>, a dictionary of named tables of <code>Value</code> cards, and <code>params</code> is the flat list of all of them (lines 80–90).</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>

<style>
  :global(.steps) { padding-left: 1.2rem; } :global(.steps li) { margin-bottom: 0.5rem; }
  :global(.tbl) { width: 100%; border-collapse: collapse; font-family: var(--font-ui); font-size: 0.9rem; margin: 1rem 0; }
  :global(.tbl td) { padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--line); vertical-align: top; }
  :global(.tbl thead td) { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
</style>
