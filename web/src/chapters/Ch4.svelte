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
  import DialLab from '../widgets/DialLab.svelte';
  import HillClimb from '../widgets/HillClimb.svelte';
  import ScaleLab from '../widgets/ScaleLab.svelte';
  import { VC } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const climbCode = `import random

a, b = 0.5, 0.5                      # start with both dials in the middle
best = loss(a, b)                    # the loss with the dials where they are now

for attempt in range(500):
    a2 = a + random.gauss(0, 0.05)   # a trial setting for dial A: a small random nudge
    b2 = b + random.gauss(0, 0.05)   # a trial setting for dial B
    if None:                         # <- replace None: adopt the trial only if it made the loss LOWER
        a, b, best = a2, b2, loss(a2, b2)

print('dials:', round(a, 3), round(b, 3), ' loss:', round(best, 4))
`;
  const climbCheck = `_a = _n['vv'] / (_n['vv'] + _n['vc'])
_b = _n['cv'] / (_n['cv'] + _n['cc'])
_floor = loss(_a, _b)
assert abs(best - loss(a, b)) < 1e-9, "best should always equal loss(a, b). When you adopt a trial setting, remember its loss as well."
assert best < loss(0.5, 0.5) - 0.01, "The loss barely moved from the start (%.4f). Is the 'if' letting good nudges through?" % best
assert best < _floor + 0.003, "After 500 attempts the loss is %.4f, but the best possible is %.4f. Are you adopting only the nudges that make the loss LOWER?" % (best, _floor)
print("Your hill-climber found dials %.0f%% and %.0f%%, within %.5f of the best possible loss." % (a * 100, b * 100, best - _floor))`;
  const climbSolution = `if loss(a2, b2) < best:`;
</script>

<Lesson id="ch4" num={4} title="Knobs" tagline="A counting table can't improve. Give the model dials, and let the score tell you how to turn them.">
  <Beat>
    <Pain title="Where we left off">
      <p>Our best model so far, the bigram table, scores <strong>2.454</strong>, and it will score 2.454 forever. Its 729 numbers were each set <em>once</em>, by counting, and there is nothing left to improve. Worse, counting only works for tables. For any model that is cleverer than a table, there is nothing to count.</p>
    </Pain>
    <p>So we're going to change the question. Stop asking <em>"what do the counts say?"</em> and start asking:</p>
    <Aha title="The new question">
      <p>Imagine the model is a machine covered in <strong>dials</strong>. Every setting of the dials gives a different model, and some are better than others. We already have a ruler for "better" (the loss from last chapter). So: <strong>what setting of the dials makes the loss as low as possible?</strong></p>
    </Aha>
    <p>Those dials have a technical name: <strong>parameters</strong>. A model is a machine with parameters; <strong>training</strong> is finding good settings for them. The bigram table is a model with 729 parameters (all set by counting). The real <code>microgpt.py</code> has <strong>4,064</strong>, and counting can't set them, so it has to find them some other way. That "other way" is the subject of the next several chapters.</p>
  </Beat>

  <Beat>
    <h2>The smallest model with dials</h2>
    <p>Let's start tiny, so we can see everything. Forget predicting the exact next letter among 27 choices. Ask only a yes/no question: <strong>is the next letter a vowel?</strong> (We'll count a, e, i, o, u as vowels.)</p>
    <p>Our model has just <strong>two dials</strong>:</p>
    <ul>
      <li><strong>Dial A</strong>: the chance the next letter is a vowel, <em>when the last letter was a vowel</em>.</li>
      <li><strong>Dial B</strong>: the chance the next letter is a vowel, <em>when the last letter was a consonant</em>.</li>
    </ul>
    <p>The loss is exactly what you built in Chapter 3: the average surprise of the model, graded on <strong>real names</strong> (164,080 neighbouring letter pairs from the dataset). With both dials at 50%, the model is just flipping a coin and the loss is 0.6931. Your job is to beat it.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch4-guess" kind="choice" options={['Dial A (after a vowel) will end up higher', 'Dial B (after a consonant) will end up higher', 'They will end up about the same']} answer={1}>
      {#snippet q()}<p>Before you touch anything: in real names, <strong>which dial will end up set higher</strong> in the best model?</p>{/snippet}
      <p>You may remember the checkerboard from Chapter 2: after a consonant, a vowel is likely; after a vowel, much less so. So B should be high and A low. Let's see how high.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <p>Now be the optimiser. Turn the dials and watch the loss. <strong>Get it under the goal line.</strong> A map of every possible setting is available if you get stuck, but try without it first: turn one dial at a time, and notice which direction helps.</p>
    <DialLab />
  </Beat>

  <Beat>
    <Aha title="You just trained a model">
      <p>No counting. You only had a score and two dials, and you turned them until the score was as good as it gets. <strong>That is training.</strong> Everything that follows is a better way of doing exactly this.</p>
    </Aha>
    <p>And look at where the dials landed: about <strong>18%</strong> and <strong>67%</strong>. Those are the very same numbers we <em>counted</em> in Chapter 2 (after a vowel a vowel follows 18.2% of the time; after a consonant, 66.8%). Two completely different routes, counting and dial-turning, arrive at the same place.</p>
    <TimeTravel year="1913" who="Markov, again">
      <p>These two numbers are what Markov tallied by hand in <em>Eugene Onegin</em>. His method was counting. Yours was feeling your way down a slope. For a model this small the answers agree. The difference is that the second method <strong>keeps working when counting becomes impossible</strong>, and that is why it's the one that scales.</p>
    </TimeTravel>
    <Sceptic q="Why does the best setting happen to equal the counted frequencies?">
      <p>Because the loss measures how surprised the model is by the real data, and the least surprised any model can be is when its probabilities <em>match the real frequencies</em>. A model that said 50% when the truth is 18% is, on average, more shocked by reality than one that says 18%. So "lowest loss" and "matches the counts" are the same destination for this model. For richer models, there are no simple counts, but the loss still points the way.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <h2>Hand the dials to the computer</h2>
    <p>You used your eyes and your hands. A computer has neither, so it needs a recipe. The simplest one: <strong>try a small random nudge to the dials. If the loss got lower, keep it. If not, undo it. Repeat.</strong> It's called <em>random hill-climbing</em> (or here, valley-descending).</p>
    <p>Run it below. Then play with <strong>the size of each nudge</strong>. Try big, then tiny, then in between, and watch how the hill-climber behaves.</p>
    <HillClimb />
  </Beat>

  <Beat>
    <p>What you should have seen:</p>
    <ul>
      <li><strong>Medium nudges</strong> (around 0.05) work well: the climber sprints into the valley. Over 200 test runs it needed a median of 22 attempts to get within 0.003 of the best possible loss.</li>
      <li><strong>Huge nudges</strong> get you to the right neighbourhood fast, then stall. Every attempt overshoots, so nothing ever fine-tunes. In our test runs, a nudge of 0.5 stopped improving about 0.0006 short of the best.</li>
      <li><strong>Tiny nudges</strong> are safe but painfully slow. In our test runs, 300 attempts at a nudge of 0.001 had only crawled to a loss of about 0.63.</li>
    </ul>
    <Aha title="Hold on to this: the step size matters">
      <p>It's a trade-off we'll meet again in Chapter 5, under the name <strong>learning rate</strong>: big steps are fast but clumsy, tiny steps are precise but slow. And you'll see it in <code>microgpt.py</code>, which has a <code>learning_rate</code> that gradually <em>shrinks</em> during training: big steps early, careful ones late.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <p>Now write the hill-climber yourself. The loss function is provided for you: <code>loss(a, b)</code> tells you how good a pair of dial settings is. There's one missing step: <em>when</em> should we adopt a trial setting?</p>
    <Cell id="ch4-climb" plain={PLAIN['ch4-climb']} title="Write the hill-climber" code={climbCode} pre={VC} rows={13} check={climbCheck} solution={climbSolution}
      hint="You want to keep the trial only if its loss is LOWER than best. In Python, 'is lower than' is the < sign.">
      <p>One step is missing. Choose it, or write it in Python.</p>
    </Cell>
  </Beat>

  <Beat>
    <Pain title="The catch">
      <p>Two dials took a few dozen attempts. But <code>microgpt.py</code> has <strong>4,064</strong> dials. What does hill-climbing do as the number of dials grows? Don't guess. Measure it. Each row below is a real experiment, running in your browser right now.</p>
    </Pain>
  </Beat>

  <Beat gate>
    <ScaleLab />
  </Beat>

  <Beat>
    <p>The pattern is clear: <strong>double the dials, roughly double the attempts.</strong> It settles at about 11 attempts per dial. If the pattern keeps holding, 4,064 dials means roughly <strong>45,000 attempts</strong>. For a real model each attempt means running the model on some data. In <code>microgpt.py</code>, one pass over a single name takes about 11 milliseconds (we measured), so even the clean bowl-shaped problem would cost minutes of pure guessing, and real problems are bumpy and noisy, which makes it far worse. Modern language models have <em>billions</em> of dials.</p>
    <p>But the real problem is more fundamental than speed. Think about what each attempt tells you:</p>
    <Aha title="One attempt = one yes/no">
      <p>"Better" or "worse". That's it. One bit of information per attempt, spent on a <strong>random</strong> direction, even though the model has thousands of dials and the right move for <em>each</em> dial is different. We're like someone tuning a 4,064-channel mixing desk blindfolded, being told only "warmer" or "colder" after every random twiddle.</p>
    </Aha>
    <Sceptic q="Couldn't we just try each dial one at a time, up and down?">
      <p>Good instinct, and that is exactly where the next chapter starts. Nudging <em>one</em> dial and seeing what the loss does tells you something much more useful than a random twiddle: it tells you that dial's <strong>direction and steepness</strong>. The catch is the price: one test per dial, for 4,064 dials, <em>every single step</em>. We'll see that it works, and then we'll find a way to get <em>all</em> the answers at once.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <Pain title="What we actually want">
      <p>A model that, in one go, tells us for <em>every</em> dial: <strong>"turn me this way, this much."</strong> That quantity has a name, the <strong>slope</strong> of the loss with respect to that dial, and it's the single most important idea in machine learning. We'll invent it in Chapter 5 with nothing but a wiggle. Then chapters 6 and 7 build the trick that gets all 4,064 slopes at once.</p>
      <p>How much would "all at once" cost? We measured it on the real file: one forward pass over a name takes about <strong>11 ms</strong>, and getting <em>every</em> dial's slope too takes about <strong>29 ms</strong>. That's roughly 2.6 times one attempt, for 4,064 dials' worth of answers.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch4">
      <ul>
        <li>A <strong>parameter</strong> is a dial. A <strong>model</strong> is a machine with parameters. <strong>Training</strong> is finding the setting of the dials that makes the loss lowest.</li>
        <li>Counting is <em>one</em> way to find the settings. Turning dials until the loss is lowest is a more general way, and it agreed with counting on the vowel model.</li>
        <li>Random hill-climbing works for two dials but needs roughly 11 attempts per dial, and each attempt only says "better" or "worse".</li>
        <li>Step size is a trade-off: too big stalls, too small crawls.</li>
        <li>In <code>microgpt.py</code>, <code>num params: 4064</code> (line 90) reports how many dials the model has.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
