<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import TimeTravel from '../components/TimeTravel.svelte';
  import Predict from '../components/Predict.svelte';
  import Cell from '../components/Cell.svelte';
  import Bank from '../components/Bank.svelte';
  import DeepStack from '../widgets/DeepStack.svelte';
  import NormLab from '../widgets/NormLab.svelte';
  import StreamX from '../widgets/StreamX.svelte';
  import BlockDiagram from '../widgets/BlockDiagram.svelte';
  import { PLAIN } from './plain.js';

  const normCode = `def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)         # the average of the squares: how loud the list is
    scale = None                                    # <- replace None: 1 divided by the square root of (ms + 0.00001)
    return [xi * scale for xi in x]

print(rmsnorm([3.0, -4.0]))
`;
  const normCheck = `import math
out = rmsnorm([3.0, -4.0])
assert abs(out[0] - 0.84853) < 1e-3 and abs(out[1] + 1.13137) < 1e-3, "rmsnorm([3, -4]) should be about [0.849, -1.131], but I got %r" % (out,)
loud = rmsnorm([300.0, -400.0])
assert all(abs(a - b) < 1e-3 for a, b in zip(out, loud)), "Making the list 100 times louder should not change the result."
assert abs(math.sqrt(sum(v * v for v in out) / 2) - 1) < 1e-3, "The output should have a typical size (RMS) of 1."
print("rmsnorm works: the same shape at any volume. Lines 103-106 of microgpt.py.")`;
  const normSolution = `scale = (ms + 1e-5) ** -0.5`;

  const resCode = `def block(x, layer):
    x_residual = x                 # remember where we started
    x = layer(x)                   # the layer proposes something
    return [None for a, b in zip(x, x_residual)]       # <- replace None: combine the proposal with where we started

print(block([1.0, 2.0], lambda v: [10.0, 20.0]))     # expect [11.0, 22.0]
print(block([1.0, 2.0], lambda v: [0.0, 0.0]))       # a layer that does nothing leaves x unchanged
`;
  const resCheck = `assert block([1.0, 2.0], lambda v: [10.0, 20.0]) == [11.0, 22.0], "block should ADD the layer's output to the starting list."
assert block([1.0, 2.0], lambda v: [0.0, 0.0]) == [1.0, 2.0], "A layer that outputs zeros should leave the list unchanged."
print("block works. The layer's output becomes an edit to what was already there (lines 134 and 141 of microgpt.py).")`;
  const resSolution = `a + b`;
</script>

<Lesson id="ch13" num={13} title="Staying Alive" tagline="Deep stacks lose their signal or blow it up. Two small ideas keep it alive.">
  <Beat>
    <Pain title="Where we left off">
      <p>We have the two halves of a transformer block, attention (gather) and the MLP (think). Real models stack the block over and over: the largest use around a hundred. Our file uses just one. But <em>why</em> does it contain the unusual-looking lines it does around the block, such as <code>x_residual = x</code> and <code>rmsnorm(x)</code>? Because stacks of layers are fragile.</p>
    </Pain>
    <p>Remember the children's game of "telephone": a whisper passed through twenty people either fades to nothing, or comes out wildly distorted. A signal passing through twenty layers has exactly the same problem, in <em>both</em> directions: the forward signal (the prediction) and the backward one (the slopes that train the early layers).</p>
  </Beat>

  <Beat gate>
    <Predict id="ch13-stack" kind="choice" options={['It vanishes: it shrinks to almost nothing', 'It is fine: the layers are similar, so it stays about the same size', 'It explodes: it grows enormous']} answer={0}>
      {#snippet q()}<p>A stack of 16 blocks, each with <strong>small random starting dials</strong> (the size microgpt uses, 0.02). A list of numbers of typical size 0.87 goes in. <strong>What size comes out the other end?</strong></p>{/snippet}
      <p>It vanishes. Each layer multiplies the signal by a number much smaller than 1 (small dials), so it shrinks at every step, and shrinks <em>exponentially</em>. Don't take my word for it. Run the experiment.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <p>Below is a real simulation of 16 blocks with random dials, run right here. Start with <strong>a plain stack</strong> and look at the two charts: the signal going forward, and the slope coming back. Then try the fixes one at a time. (The chart's vertical axis is a <em>log</em> scale: each gridline is 100,000 times bigger than the one below.)</p>
    <DeepStack />
  </Beat>

  <Beat>
    <p>What you should have seen, with the small dials microgpt uses:</p>
    <ul>
      <li><strong>A plain stack: vanishes.</strong> The signal goes from 0.87 to about 0.001 after <em>one</em> layer, and to about 10<sup>−9</sup> after two. By layer 8 it is so small that the computer rounds it to exactly zero. Slopes can't get back either, so the early layers could never learn.</li>
      <li><strong>RMSNorm alone doesn't save it.</strong> It rescales each block's <em>input</em>, but the block's <em>output</em> is still tiny.</li>
      <li><strong>A skip lane saves it.</strong> The signal stays at 0.87 at <em>every</em> depth, and the slope arrives at every layer undiminished.</li>
    </ul>
    <p>Switch the starting dials to the ten-times-bigger setting and the opposite disease appears. A plain stack <strong>explodes</strong> (over 10<sup>20</sup> by layer 8), and a skip lane <em>alone</em> explodes too. Only the <strong>combination</strong> of both fixes stays healthy, with the signal staying between about 0.9 and 5 over 16 layers. That combination is what a transformer uses.</p>
  </Beat>

  <Beat gate>
    <h2>Fix one: the skip lane (residual connection)</h2>
    <p>The idea: stop making the signal pass <em>through</em> every layer. Instead, give it a <strong>lane that runs straight past</strong>, and let each block only <em>add</em> its contribution to the lane. Think of a shared document. A plain stack is every department rewriting the whole document in turn, with each rewrite losing something. A skip lane is every department proposing <strong>edits</strong> that get applied on top of what's already there, so the original is never lost.</p>
    <Cell id="ch13-residual" plain={PLAIN['ch13-residual']} title="Wrap a layer in a skip lane" code={resCode} rows={8} check={resCheck} solution={resSolution}
      hint="The layer's output and the saved starting list have the same length. Combine them place by place: for each pair (a, b), you want a + b.">
      <p>One step is missing: how to combine. In <code>microgpt.py</code> this appears twice, after attention (line 134) and after the MLP (line 141), each with its <code>x_residual = x</code> above it.</p>
    </Cell>
    <Aha title="Why the slope survives">
      <p>Backwards, an <em>add</em> has local slope 1 (Chapter 6's table). So the slope sails down the skip lane multiplied by 1 at every step, with nothing to shrink it. The block's own path only <em>adds</em> to it. A skip lane is a highway for slopes.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <h2>The model starts by doing nothing</h2>
    <p>Here is a lovely consequence. If the lane carries the signal, a block can start out <strong>contributing nothing</strong> and then learn to add useful edits. That's exactly what <code>microgpt.py</code> does: the output matrices of attention (<code>attn_wo</code>) and the MLP (<code>mlp_fc2</code>) are created with <code>std=0</code>, all zeros (lines 86 and 88). Look at the real model's lane, at five moments of training:</p>
    <StreamX />
  </Beat>

  <Beat>
    <p>At step 0 both blocks add exactly zero: the model is just the letter-and-position path. Then training grows the edits, attention's the most. (This isn't the "all dials zero" problem from Chapter 9: only these two <em>output</em> matrices start at zero, while the others start random, so each block's dials still receive different slopes and can drift apart.)</p>
    <TimeTravel year="2015" who="Kaiming He and colleagues">
      <p>Skip connections were popularised by a Microsoft Research team working on image recognition. They called their networks <strong>ResNets</strong> ("residual networks"), and showed that with skip lanes they could successfully train a network <strong>152 layers deep</strong>, when networks beyond about twenty layers had been getting <em>worse</em>. Every transformer since uses the same trick.</p>
    </TimeTravel>
  </Beat>

  <Beat gate>
    <h2>Fix two: normalise (RMSNorm)</h2>
    <p>The skip lane stops fading, but each block's contribution can still be wildly too loud or too quiet. The second fix: before every block, <strong>rescale the list to a standard volume</strong>. "Volume" here means the <em>root mean square</em> (RMS): square every number, average the squares, take the square root. Divide the list by that, and its typical size is exactly 1, whatever it was before. Turn the volume knob below and watch the output refuse to change.</p>
    <NormLab />
  </Beat>

  <Beat gate>
    <Cell id="ch13-rmsnorm" plain={PLAIN['ch13-rmsnorm']} title="Write RMSNorm" code={normCode} rows={7} check={normCheck} solution={normSolution}
      hint="You want to divide by the typical size, which is the square root of ms. Dividing by a square root is the same as raising to the power -0.5.">
      <p>One step is missing: the scale. Lines 103–106 of <code>microgpt.py</code>. The tiny <code>1e-5</code> stops a division by zero if the list happens to be all zeros.</p>
    </Cell>
    <Sceptic q="GPT-2 used something called LayerNorm. Is RMSNorm the same thing?">
      <p>Nearly. LayerNorm first subtracts the average of the list and then divides by a typical size. RMSNorm skips the subtracting. It's a bit simpler and works about as well, which is why many recent models use it. The comment at the top of the model function in <code>microgpt.py</code> lists it among the small differences from GPT-2.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <h2>The whole path</h2>
    <p>Now every line around the blocks has a purpose. The picture below is the complete data path of one letter through <code>microgpt.py</code>'s <code>gpt()</code> function. The <span style="color: var(--series-2); font-weight: 600;">dashed skip lanes</span> carry the signal straight past each block; each block gets a <strong>fresh RMSNorm</strong> on its way in and <strong>adds</strong> its output back to the lane on its way out (this arrangement is called "pre-norm").</p>
    <BlockDiagram lit={['skip', 'norm0', 'attn', 'mlp']} title="Skip lanes (dashed) and normalisation, in the real model" />
  </Beat>

  <Beat>
    <Pain title="Everything is on the table">
      <p>We've now met every ingredient: embeddings, linear layers, attention with several heads, the MLP with its gate, skip lanes and normalisation, softmax and the loss. What's left is to <strong>assemble</strong> them into the single function the file calls <code>gpt()</code>, count its 4,064 dials, and see it run end to end. That's the next chapter, and you'll find there's nothing in it you haven't already built.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch13">
      <ul>
        <li>Deep stacks make the signal and the slopes <strong>vanish or explode</strong>, exponentially, like a game of telephone.</li>
        <li>A <strong>skip lane</strong> (residual connection): each block <em>adds</em> an edit to what's already there instead of replacing it. Slopes pass down the lane multiplied by 1. (Lines 116, 134, 136, 141.)</li>
        <li>That is why microgpt zero-starts its two output matrices: each block begins by doing nothing, and learns to add.</li>
        <li><strong>RMSNorm</strong>: divide a list by its own typical size, so it always has volume 1 (lines 103–106, used on lines 112, 117 and 137).</li>
        <li>Neither fix is enough alone. A transformer uses both, and stays healthy at any depth.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
