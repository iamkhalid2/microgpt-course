<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import Predict from '../components/Predict.svelte';
  import Cell from '../components/Cell.svelte';
  import Bank from '../components/Bank.svelte';
  import NameProb from '../widgets/NameProb.svelte';
  import LogMachine from '../widgets/LogMachine.svelte';
  import Scoreboard from '../widgets/Scoreboard.svelte';
  import { BIGRAM, LOSSFN } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const surpriseCode = `import math

def surprise(p):
    return None      # <- replace None: minus the natural log of p  (Python: math.log)

for p in [0.99, 0.5, 0.1, 1/27, 0.01, 0.001]:
    print(f"p = {p:<8.4f}  surprise = {surprise(p):.3f}")
`;
  const surpriseCheck = `assert abs(surprise(0.5) - 0.6931) < 1e-3, "surprise(0.5) should be about 0.693 (that's -ln of a half)."
assert surprise(1) == 0, "Something certain should have zero surprise."
assert abs(surprise(1/27) - math.log(27)) < 1e-9, "surprise(1/27) should be ln(27) = 3.296."
assert surprise(0.001) > surprise(0.1), "Rarer events should be MORE surprising. Did you forget the minus sign?"
print("surprise(1/27) = %.3f: that is the score of a model that knows nothing." % surprise(1/27))`;
  const surpriseSolution = `return -math.log(p)`;

  const lossCode = `def name_loss(name):
    tokens = [BOS] + [uchars.index(ch) for ch in name] + [BOS]
    losses = []
    for prev, nxt in zip(tokens, tokens[1:]):
        p = P(prev, nxt)              # how likely did the bigram table think this step was?
        losses.append(None)           # <- replace None: the surprise of p
    return sum(losses) / len(losses)  # the AVERAGE surprise per step

for n in ['emma', 'olivia', 'mia']:
    print(n, round(name_loss(n), 3))
`;
  const lossCheck = `import math
def _want(name):
    t = [BOS] + [uchars.index(c) for c in name] + [BOS]
    return sum(-math.log(P(a, b)) for a, b in zip(t, t[1:])) / (len(t) - 1)
assert isinstance(name_loss('emma'), float), "name_loss should return a number. Is losses full of None?"
for n in ['emma', 'olivia', 'mia', 'a']:
    assert abs(name_loss(n) - _want(n)) < 1e-9, "name_loss(%r) is %.4f, but should be %.4f. Average the surprise of every step." % (n, name_loss(n), _want(n))
print("name_loss works. 'emma' scores %.3f: on average each letter surprised the table by that much." % name_loss('emma'))`;
  const lossSolution = `losses.append(-math.log(p))`;

  const breakCode = `# What if a name has a pair of letters the table has NEVER seen?
print(P(uchars.index('x'), uchars.index('q')))      # how likely is 'q' right after 'x'?
print(name_loss('xqzvb'))
`;
</script>

<Lesson id="ch3" num={3} title="Scoring" tagline="Is model B better than model A? By how much? We need a number.">
  <Beat>
    <Pain title="Where we left off">
      <p>We have three generators: random, the letter wheel, the bigram table. You could <em>feel</em> that each was better than the last, by how often you got fooled. But feelings don't scale, and "feels better" isn't something a computer can <em>improve</em>. Everything in the rest of this course, every bit of actual learning, needs one thing: <strong>a single number that says how good the model is</strong>.</p>
    </Pain>
    <p>So how do you score a name-maker? Not by judging what it <em>writes</em>. That's subjective, and a model could write one good name over and over. Flip the question around instead:</p>
    <Aha title="The flip">
      <p>Show the model a <strong>real name</strong>. Ask: <em>how likely did you think that was?</em> A model that has learned the habits of real names will find real names <strong>unsurprising</strong>. A model that hasn't will find them shocking. Grade it on how shocked it is by reality.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <p>A name appears one symbol at a time, so a model's opinion of the <em>whole</em> name is built from its opinion of each step: <code>e</code> after the start, <em>and</em> <code>m</code> after <code>e</code>, <em>and</em> <code>m</code> after <code>m</code>, and so on. "And" means <strong>multiply</strong>. Try it. Type names, switch models, and watch the probabilities multiply together.</p>
    <NameProb />
  </Beat>

  <Beat>
    <p>Look at those numbers. Even for the <em>best</em> model here, <code>emma</code> gets only "1 chance in about 290,000". Pure chance gives "1 chance in 14 million". That's not a bug. Any specific name is a long shot. But it leaves us with two problems, and I want you to find the second one yourself.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch3-long" kind="choice" options={['anna', 'alexandria', 'They get about the same, since both are plausible names']} answer={1}>
      {#snippet q()}<p>Suppose a perfectly good model reads <code>anna</code> (4 letters) and <code>alexandria</code> (10 letters). Both are reasonable names. <strong>Which one gets the smaller probability, just from how probabilities multiply?</strong></p>{/snippet}
      <p><strong>alexandria.</strong> Every extra letter multiplies in another fraction smaller than 1, so longer names <em>always</em> end up with smaller products, regardless of how good the model is. Comparing raw products punishes long names for being long.</p>
      <p>That's problem one: the score is <em>unfair to length</em>. Problem two is that the numbers get absurdly tiny (a 14-letter name like <code>michaelanthony</code> comes out around 10⁻¹⁶, and some go below 10⁻²⁰) and computers eventually can't represent them at all. Both have the same cure.</p>
    </Predict>
  </Beat>

  <Beat>
    <Aha title="The cure: count the zeros">
      <p>Multiplying tiny numbers piles on zeros. <span class="mono">0.001 × 0.0001 = 0.0000001</span>. The zeros add: <strong>3 + 4 = 7</strong>. So if we just track <em>how many zeros</em> each probability has, multiplying becomes <em>adding</em>. That "how many zeros" idea is what a <strong>logarithm</strong> is.</p>
    </Aha>
    <p>The log of 0.001 is −3 (three zeros, pointing down). Logs work smoothly for numbers that aren't exact powers of ten too. Flip the sign so that <em>bigger means worse</em>, and we get a measure with a proper name: <strong>surprise</strong>.</p>
    <p class="calc"><span class="mono">surprise(p) = −log(p)</span></p>
    <p>Play with it. Panel A shows how surprise responds to probability. Panel B is the payoff: surprises <em>add</em> where probabilities <em>multiply</em>.</p>
  </Beat>

  <Beat gate>
    <LogMachine />
    <Sceptic q="Why the minus sign? And why this particular log: there are others?">
      <p>The minus sign is cosmetic: log of a probability is always negative (or zero), and we'd rather have a score where <em>higher = worse</em> and zero means <em>perfect</em>. Negate it and that's what you get.</p>
      <p>As for which log: there are different ones (base 10 counts zeros; base 2 counts bits; base <em>e</em> is called the "natural log"). They differ only by a constant factor, like inches versus centimetres. <code>microgpt.py</code> uses <code>math.log</code>, which is the natural log, because it makes the calculus in Act II come out cleanly. So the numbers you'll see are in "nats" (natural-log units). Nothing else about them matters.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <p>Now code it. First the surprise function itself, one line:</p>
    <Cell id="ch3-surprise" plain={PLAIN['ch3-surprise']} title="Surprise" code={surpriseCode} rows={8} check={surpriseCheck} solution={surpriseSolution}
      hint="Python's math.log(x) is the natural log. Put a minus sign in front.">
      <p>One step is missing. Once it works, watch how the output changes as p gets smaller.</p>
    </Cell>
  </Beat>

  <Beat>
    <Aha title="Pure chance has a score now: 3.296">
      <p>If a model knows nothing, all 27 symbols get probability 1/27 at every step. The surprise of each step is <span class="mono">−log(1/27) = log 27 = 3.296</span>. That's the score of pure guessing, the number to beat. It's also why, in the training replay on the first page, the loss started near 3.3.</p>
    </Aha>
    <p>Second problem was <em>length</em>. The fix is to <strong>average</strong> instead of multiply: take the surprise at each step and find the mean. A four-letter name and a ten-letter name now both get "average surprise per step", and they're finally comparable.</p>
  </Beat>

  <Beat gate>
    <p>Put the two together. This is the moment the chapter has been building to: a function that takes <em>any name</em> and returns <em>how surprised the bigram table is by it, on average</em>. The probability function <code>P(prev, nxt)</code> is provided. It's the count in the table cell divided by its row's total.</p>
    <Cell id="ch3-loss" plain={PLAIN['ch3-loss']} title="Score a name" code={lossCode} pre={BIGRAM} rows={13} check={lossCheck} solution={lossSolution}
      hint="You already wrote surprise(p) in the last cell. Here you can use -math.log(p) directly (math is already imported).">
      <p>One step is missing: what to store for each step.</p>
    </Cell>
    <Aha title="Four of microgpt's lines, in your own code">
      <p>Look at lines 162, 167, 168 and 169 of <code>microgpt.py</code>: <code>losses = []</code> · <code>losses.append(loss_t)</code> · <code>loss_t = -probs[target_id].log()</code> · <code>loss = (1 / n) * sum(losses)</code>. That's <code>name_loss</code>: start a list, add each step's surprise, average them. The model's "loss" <em>is</em> this number.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <h2>The scoreboard</h2>
    <p>Now do it for the whole dataset, for every model we've built. This number is called the <strong>loss</strong>: the average surprise per symbol, over all names. <strong>Lower is better.</strong></p>
    <Scoreboard />
  </Beat>

  <Beat>
    <p>Three things to take from that board.</p>
    <ol>
      <li><strong>Every step up is real.</strong> Random 3.296 → letter frequencies 2.823 → bigram 2.454. You can now say <em>how much</em> better a model is, in one number.</li>
      <li><strong>Reality can shock you to infinity.</strong> Switch to "names it never counted" and the bigram score explodes. One unseen pair means probability 0, which means surprise <span class="mono">−log(0) = ∞</span>. A model must <strong>never say never</strong>. The later models in this course avoid it by construction: the function that turns their numbers into probabilities (softmax, Chapter 8) doesn't produce exact zeros in practice.</li>
      <li><strong>The table isn't just memorising.</strong> Give it a little humility and it scores almost the same on names it never saw. With only 729 numbers there's nothing to memorise <em>with</em>.</li>
    </ol>
    <Sceptic q="What happens if I feed name_loss a name with an unseen pair, in Python?">
      <p>Try it. It's a one-line experiment.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <Cell id="ch3-break" plain={PLAIN['ch3-break']} title="Break it on purpose" code={breakCode} pre={LOSSFN} rows={4} allowError>
      <p><code>x</code> is never followed by <code>q</code> in any name, so <code>P</code> is 0. Run it and read the error.</p>
    </Cell>
    <p><code>ValueError: math domain error</code> is Python's way of saying "the log of zero doesn't exist". It's the same ∞ the scoreboard showed, just as a crash. Keep that in mind for Chapter 8: it's why the function that makes probabilities needs to be built so that it can <em>never</em> produce a zero.</p>
  </Beat>

  <Beat>
    <Pain title="The ceiling">
      <p>Look once more at the bottom of the scoreboard, at the preview bars. The real 200-line model, after only <strong>500</strong> steps of training, is roughly <em>level</em> with our humble counting table. After <strong>3,000</strong> it's clearly ahead, at about 2.36. And it can keep going, while the table is stuck at 2.454 forever. Counting can't improve by being trained longer. There's nothing to train.</p>
      <p>To beat the table we need a model with <strong>knobs</strong> that can be turned to <em>reduce this loss</em>: a function whose settings we can tune. And since we now have a number for "how good", we can finally ask the question that drives the rest of the course: <strong>which way should each knob turn to make that number go down?</strong></p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch3">
      <ul>
        <li><strong>Surprise</strong> = −log(probability): zero for certain things, huge for rare ones.</li>
        <li>Why logs: they turn a product of many tiny probabilities into an easy sum.</li>
        <li><strong>Loss</strong> = the <em>average</em> surprise per step. It's the score of a model, and lower is better.</li>
        <li>The code for it: <code>losses = []</code> … <code>-probs[target_id].log()</code> … <code>(1/n) * sum(losses)</code> (lines 162–169).</li>
        <li>Two landmarks: <strong>3.296</strong> for pure guessing, <strong>2.454</strong> for the bigram table.</li>
        <li>A model must never assign probability exactly 0.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
