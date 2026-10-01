<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import Predict from '../components/Predict.svelte';
  import Cell from '../components/Cell.svelte';
  import Bank from '../components/Bank.svelte';
  import SoftmaxLab from '../widgets/SoftmaxLab.svelte';
  import NeuralBigram from '../widgets/NeuralBigram.svelte';
  import { SOFTMAX } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const softmaxCode = `import math

def softmax(logits):
    biggest = max(logits)                              # shift so the biggest score becomes 0 (keeps exp from overflowing)
    exps = [math.exp(x - biggest) for x in logits]     # turn every score into a positive number
    total = sum(exps)
    return [None for e in exps]                        # <- replace None: each number's share of the total

print(softmax([2.0, 1.0, 0.1]))
`;
  const softmaxCheck = `p = softmax([1.0, 2.0, 3.0])
want = [0.09003057, 0.24472847, 0.66524096]
assert all(abs(a - b) < 1e-6 for a, b in zip(p, want)), "softmax([1, 2, 3]) should be about %r, but I got %r" % (want, p)
assert abs(sum(softmax([5, -3, 0.5, 2])) - 1) < 1e-9, "The probabilities must add up to exactly 1."
assert all(x > 0 for x in softmax([-50, 0, 50])), "Every probability must be positive, even for very negative scores."
q = softmax([1000.0, 1001.0])
assert abs(q[1] - 0.7310585786) < 1e-6, "Huge scores should still work. Did you subtract the biggest score before exp?"
print("softmax works: positive, adds to 1, and survives huge scores.")`;
  const softmaxSolution = `return [e / total for e in exps]`;

  const signalCode = `def error_signal(logits, target):
    probs = softmax(logits)
    # How should the loss change if we nudge each score up? What the model predicted, minus what actually happened.
    return [None for i in range(len(probs))]       # <- replace None: probs[i] minus (1 if it is the right answer, else 0)

print(error_signal([0.3, -1.2, 2.0], 2))
`;
  const signalCheck = `import math
logits, target, h = [0.3, -1.2, 2.0, 0.5], 2, 1e-6
g = error_signal(logits, target)
base = -math.log(softmax(logits)[target])
for i in range(len(logits)):
    up = list(logits); up[i] += h
    wig = (-math.log(softmax(up)[target]) - base) / h
    assert abs(g[i] - wig) < 1e-4, "Score %d: your error signal is %.4f, but the wiggle test says %.4f" % (i, g[i], wig)
print("Matches the wiggle test for every score. The whole 'learning signal' is: predicted minus actual.")`;
  const signalSolution = `probs[i] - (1 if i == target else 0)`;
</script>

<Lesson id="ch8" num={8} title="Softmax" tagline="Dials make scores, not probabilities. Here is the one small invention that fixes that.">
  <Beat>
    <Pain title="Where we left off">
      <p>Chapters 4 to 7 gave us a way to <em>train</em> dials: score them with a loss, work out each dial's slope, step downhill. But look at what we trained. Two dials that were <em>already</em> probabilities, and we had to keep them between 0 and 1 by hand (remember how a step that was too big threw a dial out of range and crashed the loss).</p>
    </Pain>
    <p>A real model won't be so tidy. Its dials will combine in all sorts of ways and come out as <strong>scores</strong>: 2.7 for one letter, −4 for another, 0.1 for a third. A score can be any number, but a probability must be:</p>
    <ul>
      <li><strong>positive</strong> (never 0, or the loss blows up as in Chapter 3),</li>
      <li><strong>one of a set that adds up to exactly 1</strong>, and</li>
      <li><strong>in the same order as the scores</strong>: a bigger score should mean a bigger probability.</li>
    </ul>
    <p>We need a function that takes any list of scores and turns it into probabilities. Let's invent one.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch8-positive" kind="choice" options={['Square it (x²)', 'Take its absolute value', 'Raise e to its power (eˣ, "exp")', 'Add 100 to it']} answer={2}>
      {#snippet q()}<p>Step one: make every score <strong>positive</strong> without scrambling their order (a bigger score must stay bigger). Which operation does that?</p>{/snippet}
      <p><strong>exp</strong>, which is "e (≈ 2.718) raised to the power of the score". It's positive for <em>every</em> input (e<sup>−100</sup> is tiny but not zero), and it always goes up when the score goes up. Squaring and absolute value both treat −3 and +3 the same, which scrambles the order. Adding 100 only works until someone gives you −101.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <p>Step two: divide each positive number by the total, so each becomes a <strong>share of the whole</strong> and the shares add to 1. Put the two steps together and you have <strong>softmax</strong>: a "soft" version of picking the maximum, where the biggest score gets the biggest share but everyone gets <em>something</em>. Try it. Move the scores and watch the shares.</p>
    <SoftmaxLab />
  </Beat>

  <Beat>
    <p>Same idea, four ways:</p>
    <div class="calc">
      <p><strong>In words:</strong> make every score positive with exp, then give each its share of the total</p>
      <p><strong>In numbers:</strong> scores 1, 2, 3 → exp gives 2.72, 7.39, 20.09 → total 30.19 → probabilities <strong>0.09, 0.24, 0.67</strong></p>
      <p><strong>In code:</strong> <code>exps = [math.exp(x) for x in scores]</code>, then <code>[e / sum(exps) for e in exps]</code></p>
      <p><strong>As a symbol:</strong> probability<sub>i</sub> = e<sup>score<sub>i</sub></sup> ÷ (sum of e<sup>score</sup> over all letters)</p>
    </div>
    <Aha title="Two things the lab showed you">
      <p><strong>Only the gaps between scores matter.</strong> Adding the same number to every score changes nothing, because it multiplies every exp by the same factor, which cancels in the division. <strong>And that is a trick.</strong> Subtracting the biggest score before the exp changes no probability, but it keeps the numbers small. Without it, a score of 800 makes e<sup>800</sup> overflow to infinity and the answer becomes NaN ("not a number"). It's the line <code>max_val = max(...)</code> in <code>microgpt.py</code>.</p>
    </Aha>
    <Sceptic q="Why exp? Why not just divide each score by the sum of the scores?">
      <p>Because scores can be negative. Scores of 3 and −3 add to 0 and you'd be dividing by zero; scores of 5 and −2 would give a "probability" of −0.67. Exp first makes everything positive, <em>then</em> sharing out the total is safe. There's also a deeper reason that you'll see in a few minutes: exp turns <em>adding</em> to a score into <em>multiplying</em> a probability, which is exactly how "odds" behave.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <Cell id="ch8-softmax" plain={PLAIN['ch8-softmax']} title="Write softmax" code={softmaxCode} rows={9} check={softmaxCheck} solution={softmaxSolution}
      hint="After the exps, each one's probability is its share of the total: that number divided by total.">
      <p>One step is missing: turning the positive numbers into shares. You've just written lines 97–101 of <code>microgpt.py</code>.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <h2>The loss, through softmax</h2>
    <p>Now chain it together. The model produces scores for the 27 possible next symbols; softmax turns them into probabilities; and the loss (from Chapter 3) is the surprise at the <em>right</em> answer: <code>−log(probability of the right letter)</code>. That is lines 166 to 168 of <code>microgpt.py</code>: <code>probs = softmax(logits)</code>, then <code>loss_t = -probs[target_id].log()</code>.</p>
    <Predict id="ch8-loss" kind="number" answer={0.693} tolerance={0.05}>
      {#snippet q()}<p>Quick check of your Chapter 3 skills. The model gives the <strong>right</strong> next letter a probability of <strong>0.5</strong>. What's the loss (−ln 0.5)?</p>{/snippet}
      <p>0.693, which also happens to be the loss of a fair coin flip. Give the right answer probability 1 and the loss is 0; give it 0.01 and the loss is 4.6.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <Aha title="The prettiest fact in the course">
      <p>If you push the loss back through softmax with the chain rule from Chapter 6, the slope for each score comes out astonishingly simple:</p>
      <p class="calc"><span class="mono">slope = (what the model predicted) − (what actually happened)</span></p>
      <p>For the right answer: its predicted probability minus 1 (always negative, so raise that score). For every wrong answer: its predicted probability minus 0 (always positive, so lower it). It's exactly a <em>forecast error</em>, the kind a business analyst computes every month. Try it.</p>
    </Aha>
    <Cell id="ch8-signal" plain={PLAIN['ch8-signal']} title="The error signal" code={signalCode} pre={SOFTMAX} rows={8} check={signalCheck} solution={signalSolution}
      hint="For score i: the model's probability probs[i], minus 1 if i is the target and minus 0 if it is not. Python has a neat way to write it: (1 if i == target else 0).">
      <p>One step is missing. The page checks it against the wiggle test for every score.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <h2>A table of dials, trained</h2>
    <p>Now we can finally do what we promised in Chapter 4. Take the bigram table, but instead of <em>counting</em>, make each of the 729 cells a <strong>dial</strong> (a raw score). Run each row through softmax to get probabilities, grade with the loss, and train with gradient descent using the error signal you just built.</p>
    <p>The dials start as tiny random numbers, so softmax gives nearly equal probabilities and the loss starts at pure chance, 3.296. Press play and watch it fall.</p>
    <NeuralBigram />
  </Beat>

  <Beat>
    <Aha title="It rediscovered counting">
      <p>No one told it to count. It just followed slopes downhill, and it ended up with a table whose probabilities match the counted table almost exactly (loss 2.455 against 2.454, with a counted bigram table as the best any one-letter-memory model can do). Two completely different routes to the same table.</p>
    </Aha>
    <p>There's something elegant hiding in the learned dials. Since softmax gives each letter a probability proportional to e<sup>dial</sup>, matching the counts means <strong>the dials differ by the logarithm of the count ratios</strong>. We checked: after training, the gap between the "a" and "n" dials in the row after "e" is −1.37, and log(count of "ea" ÷ count of "en") is also −1.37. Dials are <strong>log-counts</strong>. That's what "scores" mean here: <em>how many times more likely, on a log scale</em>.</p>
    <Sceptic q="So why bother, if it only gets the same table as counting?">
      <p>Because counting only works when the model <em>is</em> a table. Now that we can <strong>train</strong> a model by slopes, the dials can be anything: not one dial per pair of letters, but dials inside a machine that reads <em>many</em> letters, and shares what it learns across similar situations. The softmax-and-loss recipe stays exactly the same. That's Act III.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <Pain title="Where this leaves us">
      <p>We can now train any model that produces a score for each possible next letter, as long as we can compute those scores with the arithmetic our engine understands. But our only model still looks back <em>one</em> letter. To look further back we need a way to <strong>feed a letter into a model as numbers that mean something</strong>, so that similar letters can share what the model learns. Act III starts there.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch8">
      <ul>
        <li><strong>Softmax</strong>: any scores → positive (exp) → shares of the total. Probabilities that add to 1, in the same order as the scores, never zero.</li>
        <li>Only the <strong>gaps</strong> between scores matter, so subtracting the biggest score is free and prevents overflow.</li>
        <li>The loss through softmax is <code>−log(probability of the right answer)</code>, and its slope for each score is <strong>predicted minus actual</strong>.</li>
        <li>A table of dials trained this way <strong>rediscovers counting</strong>: dials are log-counts.</li>
        <li>In <code>microgpt.py</code>: <code>softmax</code> (lines 97–101) and <code>probs = softmax(logits)</code> (line 166).</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
