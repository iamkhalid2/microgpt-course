<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import Predict from '../components/Predict.svelte';
  import Cell from '../components/Cell.svelte';
  import Bank from '../components/Bank.svelte';
  import Lessons from '../widgets/Lessons.svelte';
  import TrainLive from '../widgets/TrainLive.svelte';
  import { GPTFN } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const lossCode = `def name_loss(name):
    tokens = [BOS] + [uchars.index(ch) for ch in name] + [BOS]      # wrap the name (line 157)
    n = min(block_size, len(tokens) - 1)                             # at most 8 lessons (line 158)
    keys, values = [], []                                            # an empty filing cabinet
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]     # shown this, must predict that
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = None                                                # <- replace None: the surprise at the right answer
        losses.append(loss_t)
    return sum(losses) / n                                           # the average over the lessons

print(round(name_loss('emma'), 3))       # expect 2.548
print(round(name_loss('olivia'), 3))
`;
  const lossCheck = `def _ref(name):
    toks = [BOS] + [uchars.index(c) for c in name] + [BOS]
    m = min(block_size, len(toks) - 1)
    ks, vs, tot = [], [], 0.0
    for i in range(m):
        tot += -math.log(softmax(gpt(toks[i], i, ks, vs))[toks[i + 1]])
    return tot / m
for nm in ['emma', 'olivia', 'a', 'jonathanwilliam']:
    assert abs(name_loss(nm) - _ref(nm)) < 1e-9, "name_loss(%r) is %.4f but should be %.4f" % (nm, name_loss(nm), _ref(nm))
print("name_loss works: this is lines 155-169 of microgpt.py, run on the trained model.")`;
  const lossSolution = `loss_t = -math.log(probs[target_id])`;
</script>

<Lesson id="ch16" num={16} title="The Loop" tagline="One name is a stack of lessons. Feed it names, one after another, and press the button thousands of times.">
  <Beat>
    <Pain title="Where we left off">
      <p>We can grade <em>one</em> prediction (Chapter 3), get every dial's slope (Chapters 6 and 7), and turn slopes into good steps (Chapter 15). But one prediction is not a lesson, and the model has 4,064 dials to tune. It needs <strong>many</strong> lessons, in a form it can learn from, again and again. Where do they come from?</p>
    </Pain>
    <p>Think of someone training to forecast monthly sales. Give her ten years of history and she can practise 120 times: at each month, look at everything <em>so far</em>, <strong>forecast the next month</strong>, then compare with what actually happened. Every month of history is a free lesson, because the answer is already known.</p>
    <Aha title="The trick that makes language models trainable">
      <p>A name is the same kind of history. Reading it left to right, <strong>every position is a forecasting exercise</strong>: given the letters so far, predict the next one, and the answer is right there in the data. No one has to label anything. One six-letter name gives seven lessons for free. That's the whole secret of how models learn from raw text: <strong>predict the next symbol, at every position</strong>.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <p>Here is one name, split into its lessons, scored by the real trained model. For each position you see the symbol the model is <em>shown</em>, the symbol it must <em>predict</em>, the probability it gave the right answer, and the surprise (loss). The name's loss is the average. Try your own name, and compare the model at the start of training with the model at the end.</p>
    <Lessons />
  </Beat>

  <Beat>
    <p>Notice three things:</p>
    <ul>
      <li><strong>Most of the surprise is in the first letters.</strong> After "e" the model is mostly guessing; by the end of "emma" it's more confident. Later letters are easier because there's more context.</li>
      <li><strong>The model is shown the <em>true</em> letters,</strong> never its own guesses. If it guessed "x" after "e", the next lesson still starts from the real "m". That's called <strong>teacher forcing</strong>, and it keeps every lesson clean.</li>
      <li><strong>The trained model is not better at every name.</strong> On "emma" it scores about 2.55, slightly <em>worse</em> than the counting table's 2.51, but on "olivia" it scores 2.27, much better than the table's 2.50. Only the <em>average</em> over thousands of names is better (2.36 against 2.45), which is why we always judge a model on many examples, never one.</li>
    </ul>
  </Beat>

  <Beat gate>
    <h2>Score a name</h2>
    <p>Write the scoring recipe yourself. It's the loop body of the training code: for each position, run the model, turn its scores into probabilities, take the surprise at the right answer, and average over the positions. It reuses everything you've built, and the trained model's <code>gpt</code> from Chapter 14 is already loaded.</p>
    <Cell id="ch16-loss" plain={PLAIN['ch16-loss']} title="The loss of a whole name" code={lossCode} pre={GPTFN} rows={14} check={lossCheck} solution={lossSolution}
      hint="Chapter 3: surprise = minus the log of the probability. The probability you want is the one the model gave to the right next symbol, probs[target_id]. math.log is available.">
      <p>One step is missing: the surprise at the right answer. Lines 155–169 of <code>microgpt.py</code>.</p>
    </Cell>
  </Beat>

  <Beat>
    <h2>The loop, from the top</h2>
    <p>The training loop is a handful of lines (152–184). Each time round, called a <strong>step</strong>:</p>
    <ol class="steps">
      <li><strong>Pick the next name</strong>: <code>doc = docs[step % len(docs)]</code> (line 156). One name per step. (The <code>%</code> wraps back to the start if the steps ever outnumber the names.)</li>
      <li><strong>Wrap and cut it</strong>: add the start/end symbols, and keep at most 8 lessons (<code>block_size</code>) (lines 157–158).</li>
      <li><strong>Forward</strong>: run the model at every position, collecting the surprise at each, and average them into one number, the loss (lines 160–169).</li>
      <li><strong>Backward</strong>: <code>loss.backward()</code> fills in the slope of every one of the 4,064 dials (line 172). This is your Chapter 7 engine.</li>
      <li><strong>Adam update</strong>: step every dial, then wipe the slopes (lines 174–182). This is Chapter 15.</li>
      <li><strong>Report</strong> the loss (line 184), and go round again.</li>
    </ol>
    <p>That's all training is. 500 times round that loop <em>is</em> the 200-line program.</p>
    <Aha title="Two facts about the data it actually sees">
      <p><strong>500 steps means 500 names.</strong> Out of 32,033, that's just <strong>1.6%</strong> of the data. Even the 3,000-step run you've been exploring has looked at only 9.4% of the names, each exactly once. The model learns the <em>habits</em> of names, not the names themselves, which is why a small slice is enough.</p>
      <p><strong>Long names are cut short.</strong> With only 8 positions, a name longer than 7 letters loses its ending: the model never sees how long names finish. That affects <strong>15%</strong> of the names (4,801 of 32,033). It's a deliberate simplification in a tiny model.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <Predict id="ch16-guess" kind="number" answer={2.7} tolerance={0.35}>
      {#snippet q()}<p>You're about to train the real model from random dials. Pure chance scores 3.30, and the finished 3,000-step model about 2.36. <strong>What loss do you expect after 100 steps?</strong> (Average of the last few steps.)</p>{/snippet}
      <p>In the recording at the start of the course, the loss averaged about <strong>2.74</strong> over the 25 steps up to step 100. A hundred names is enough to learn the basic habits (vowels and consonants, common endings) but not the finer detail. Let's see what <em>your</em> run gives.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <h2>Train it for real</h2>
    <p>This is the genuine <code>microgpt.py</code> from this repository, running in your browser: random dials in, names out. Press start and watch the loss fall and the names change. (It takes about ten seconds for 100 steps and a minute for the file's full 500. If you're curious, try 3,000: that's the run behind the trained weights in the other chapters, and takes about six minutes.)</p>
    <TrainLive defaultSteps={100} />
  </Beat>

  <Beat>
    <p>That's the opening demo from Chapter 0, but now every line of it is yours. At step 0 the model writes noise like <code>slhvgtbo</code> because its dials are random. A few dozen steps later it has found the vowel-and-consonant rhythm. By 100 steps the output is name-shaped, and it keeps sharpening from there.</p>
    <Sceptic q="Why does the loss bounce around so much from step to step?">
      <p>Each step trains on <strong>one name</strong>, and names differ: a short common name might score 2.0, a strange long one 3.5. So the loss of a single step is noisy, and the chart smooths over a dozen steps. This is the same noise that motivated Adam's momentum in Chapter 15.</p>
    </Sceptic>
    <Sceptic q="Only 100 or 500 names? Isn't that tiny?">
      <p>It is tiny, and it's enough to see learning happen. More steps help: the 3,000-step run reached a loss of 2.36 against about 2.5 for 500 steps, and it would keep improving. Real language models are trained on trillions of symbols for exactly this reason: the loop is the same, but the number of times round is vastly larger. The 3,000-step weights you've been exploring in earlier chapters came from running this same file for six times as long.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <Pain title="Trained, but silent">
      <p>The model now holds real knowledge in its 4,064 dials, but all it can do so far is <em>score</em> names and <em>predict one symbol</em>. We've shown samples, but we haven't looked at how they're produced. In the last stretch of the file it writes names from scratch, and that has a surprising amount of craft in it: how to pick a symbol from a list of probabilities, and how to dial between "safe and boring" and "creative and risky". That's what a language model does when you chat with it, and it's the next chapter.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch16">
      <ul>
        <li><strong>Every position is a lesson</strong>: show the true symbols so far, predict the next, compare. One name gives many lessons for free.</li>
        <li><strong>Teacher forcing</strong>: the model always sees the true history, not its own guesses.</li>
        <li>The <strong>loop</strong>: pick a name, wrap and cut it, forward (average the surprises), backward, Adam update, wipe the slopes, report (lines 152–184).</li>
        <li>A name's loss is the <strong>average over its lessons</strong>. A model should be judged on many names, never one.</li>
        <li>500 steps sees just 1.6% of the names; names over 7 letters are cut short (15% of them).</li>
        <li>You ran the real file: random dials in, name-shaped output after about a hundred steps.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>

<style>
  :global(.steps) { padding-left: 1.2rem; } :global(.steps li) { margin-bottom: 0.5rem; }
</style>
