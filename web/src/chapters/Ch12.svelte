<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import Predict from '../components/Predict.svelte';
  import Cell from '../components/Cell.svelte';
  import Bank from '../components/Bank.svelte';
  import XorLab from '../widgets/XorLab.svelte';
  import MlpX from '../widgets/MlpX.svelte';
  import BlockSwitch from '../widgets/BlockSwitch.svelte';
  import { LINEAR } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const mlpCode = `def relu(v):
    return None       # <- replace None: let positive numbers through, turn negative ones into 0

def mlp(x, fc1, fc2):
    h = linear(x, fc1)                       # expand: a short list becomes a longer one
    h = [relu(v) ** 2 for v in h]            # gate every number (and square it, as microgpt does)
    return linear(h, fc2)                    # shrink: back to a short list

print(mlp([1.0, -2.0], [[1, 0], [0, 1], [1, 1]], [[2, 3, 4], [0.5, 0.5, 0.5]]))     # expect [2.0, 0.5]
`;
  const mlpCheck = `assert relu(-3) == 0 and relu(2) == 2 and relu(0) == 0, "relu should block negatives (-3 -> 0) and pass positives (2 -> 2)."
out = mlp([1.0, -2.0], [[1, 0], [0, 1], [1, 1]], [[2, 3, 4], [0.5, 0.5, 0.5]])
assert out == [2.0, 0.5], "mlp gave %r, expected [2.0, 0.5]" % (out,)
print("The MLP block works. These are lines 138-140 of microgpt.py.")`;
  const mlpSolution = `return max(0, v)`;
</script>

<Lesson id="ch12" num={12} title="Thinking" tagline="Mixing isn't deciding. A gate between two layers is all it takes to make decisions possible.">
  <Beat>
    <Pain title="Where we left off">
      <p>Attention <em>gathers</em> information: it mixes notes from earlier positions into one blended list. But gathering isn't deciding. After the blend, each position needs to <strong>think about what it has collected</strong>: "the last letter was a vowel <em>and</em> this is the third letter <em>and</em> the name started with J, so…". That needs a step that works on each position's list on its own, and that can express <em>conditions</em>.</p>
    </Pain>
    <p>We already have a tool for turning one list into another: the <strong>linear layer</strong> (a scorecard per output). So the obvious plan is to stack a few of them. But there's a trap, and you can see it with plain arithmetic.</p>
  </Beat>

  <Beat gate>
    <p>Say you apply a 10% price rise, then a 20% price rise. The result is the same as a single rise of 32%: <code>1.1 × 1.2 = 1.32</code>. Two steps collapsed into one. The same happens with layers. Take an input (1, 2) and push it through two scorecards, one after the other:</p>
    <div class="calc">
      <p><strong>Layer A</strong> = [[1, 2], [0, 1]] turns (1, 2) into (5, 2). <strong>Layer B</strong> = [[2, 0], [1, 1]] turns (5, 2) into <strong>(10, 7)</strong>.</p>
      <p>Now combine the two layers into <em>one</em> scorecard: [[2, 4], [1, 3]]. Apply it to (1, 2) directly: <strong>(10, 7)</strong>. The same answer.</p>
    </div>
    <Predict id="ch12-stack" kind="choice" options={['100 layers\' worth of power', 'About 10 layers\' worth', 'Just 1 layer\'s worth: they all collapse into a single linear layer']} answer={2}>
      {#snippet q()}<p>A model stacks <strong>100 linear layers</strong>, one after another, with <em>nothing</em> between them. How much power does it have?</p>{/snippet}
      <p>One layer's worth. However many scorecards you chain, the combined effect is always just another single scorecard (whose weights are the products of all the others). A hundred layers would be a hundred times more dials to train for no extra ability. We need something <strong>between</strong> the layers that stops them collapsing.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <h2>The gate that breaks the collapse</h2>
    <p>The something is almost insultingly simple: a <strong>one-way gate</strong> on each number. Positive numbers pass through unchanged; negative numbers are blocked (become 0). It's called <strong>ReLU</strong> ("rectified linear unit"), and in code it's <code>max(0, v)</code>. It's a tiny <em>if-then</em> rule: <em>if the number is positive, keep it; otherwise drop it</em>. And that is enough to make decisions.</p>
    <p>Here's a puzzle that a single scorecard provably can't solve: classify points by whether <em>x and y have the same sign</em> (blue) or not (orange). It's the pattern called <strong>XOR</strong>, "one or the other but not both". Train three tiny networks on it: no hidden layer, a hidden layer with <em>no</em> gate, and a hidden layer <em>with</em> the gate.</p>
    <XorLab />
  </Beat>

  <Beat>
    <Aha title="The gate is everything">
      <p>In our runs, the network with <strong>no hidden layer</strong> and the one with a hidden layer but <strong>no gate</strong> both end up getting exactly <strong>59.2%</strong> right: not much better than a coin flip, and <em>identical</em>, because the stacked layers had collapsed into one. Add a ReLU gate between the layers and it reaches <strong>100%</strong>. With only 4 hidden neurons: 99.2%. With just 2: 71.7%. More gated neurons buy more ability to carve up the space.</p>
    </Aha>
    <p>Why it works: each gated neuron carves the space with one straight <em>fold</em>, ignoring everything on one side of it. Add up enough folds and you can make any shape: corners, bumps, quadrants. A bit like building a sculpture from flat cuts.</p>
    <Sceptic q="The real code has relu(x) ** 2 (squared). Why square it?">
      <p>It's one of the "minor differences" the file mentions in its comment on line 93. Squaring the positive part makes the gate's output grow smoothly from zero instead of with a sharp corner, and makes bigger signals count for more. It's a design choice that works well in small models and not something the idea depends on: plain ReLU works too.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <h2>The MLP block</h2>
    <p>Put it together: <strong>expand</strong> the list with a linear layer (16 numbers → 64, i.e. four times as many "neurons" to work with), <strong>gate</strong> them, and <strong>shrink</strong> back to 16 with a second linear layer. That sandwich is called an <strong>MLP</strong> ("multi-layer perceptron"), and it's applied to <em>each position separately</em> (unlike attention, which mixes positions).</p>
    <Cell id="ch12-mlp" plain={PLAIN['ch12-mlp']} title="Write the MLP block" code={mlpCode} pre={LINEAR} rows={11} check={mlpCheck} solution={mlpSolution}
      hint="The gate lets positive numbers through and replaces negative ones with 0. Python's max(a, b) returns the larger of two numbers.">
      <p>One step is missing: the gate itself. The rest is the <code>linear</code> function from Chapter 9, twice. Lines 138–140 of <code>microgpt.py</code>.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <h2>Inside the real MLP</h2>
    <p>In the real model each position has <strong>64 neurons</strong>. Pick a letter and a position and see which neurons are active.</p>
    <MlpX />
  </Beat>

  <Beat>
    <Aha title="Mostly silent">
      <p>At any given position, only a handful of the 64 neurons are active. Across a few hundred names, about <strong>87% of the neurons are exactly zero</strong> at a typical position. Each neuron is a little specialist that only speaks up in particular situations, and the gate keeps the rest quiet. That's what lets a small number of neurons store many different "if this situation, then that" rules.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <h2>Do attention and the MLP earn their keep?</h2>
    <p>The real transformer block is two halves: <strong>attention</strong> (gather) then the <strong>MLP</strong> (think). Let's measure how much each contributes, by switching them off in the trained model. Move through the training steps too, to see when each half starts to matter.</p>
    <BlockSwitch />
  </Beat>

  <Beat>
    <p>What it shows (on the same 600 names, after 3,000 training steps):</p>
    <ul>
      <li><strong>Both halves help.</strong> The full model scores 2.276. Switch off the MLP and it gets worse by 0.045; switch off attention and it gets worse by 0.088; switch off both (so the model just maps letter plus position straight to the output) and it's 0.104 worse.</li>
      <li><strong>The direct path is already a bigram.</strong> With both halves off the loss is 2.380, almost exactly the counting table's 2.381 on these names. The blocks are what lift the model <em>beyond</em> counting: by about 0.10, or 4%.</li>
      <li><strong>It took time.</strong> At step 1,000, switching off the MLP actually made the model <em>better</em> (2.383 against 2.424). The blocks hadn't yet learned to be useful, and were adding noise. They earn their place later in training.</li>
    </ul>
    <Sceptic q="Only 4% better than counting? After all this?">
      <p>Yes, and it's worth being honest about it. For short names with one clear letter of context, counting captures most of what there is to know. What the transformer buys is <em>room to grow</em>: counting can never use more context or more data in a smart way, while this model can keep improving with more training, more layers and more width (that's the 2.36 versus 2.454 you saw earlier, and it's still falling). The point of the next seven chapters is to see the machine that scales. Names are the gentlest possible test of it.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <Pain title="Making it deep">
      <p>We now have the two halves of a transformer block: <strong>attention</strong> (gather) and the <strong>MLP</strong> (think). A real model repeats this block many times, and the bigger ones stack a hundred of them. But deep stacks have a nasty habit: signals can <strong>explode</strong> or <strong>fade away</strong> as they pass through layer after layer, and the slopes going backwards can too. Making deep stacks trainable takes two small inventions, which are the subject of the next chapter.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch12">
      <ul>
        <li><strong>Stacked linear layers collapse</strong> into one (like 10% then 20% making 32%), so depth alone adds nothing.</li>
        <li>A <strong>ReLU gate</strong> (<code>max(0, v)</code>) between layers breaks the collapse and lets a network make <strong>decisions</strong>. The XOR puzzle shows it: 59.2% without the gate, 100% with it.</li>
        <li>The <strong>MLP block</strong>: expand (16 → 64), gate (relu then squared), shrink (64 → 16), applied to each position separately (lines 138–140).</li>
        <li>In the real model about 87% of the neurons are exactly zero at a typical position.</li>
        <li>Attention and the MLP each help (removing them costs 0.088 and 0.045), together lifting the model about 4% beyond a counting table.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
