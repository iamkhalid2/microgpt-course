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
  import WiggleLab from '../widgets/WiggleLab.svelte';
  import Compass from '../widgets/Compass.svelte';
  import DescentLab from '../widgets/DescentLab.svelte';
  import { VC } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const slopeCode = `def slope(f, x, h=0.0001):
    # Nudge x up by a tiny h. How much did f change, per unit of nudge?
    return None     # <- replace None

print(slope(lambda x: x * x, 3))      # the exact answer is 6
print(slope(lambda x: 5 * x, 10))     # a straight line rising 5 for every 1: should be 5
`;
  const slopeCheck = `assert abs(slope(lambda x: x * x, 3) - 6) < 0.01, "slope of x*x at 3 should be about 6, but I got %r" % (slope(lambda x: x * x, 3),)
assert abs(slope(lambda x: 5 * x, 10) - 5) < 1e-3, "a line that rises 5 per 1 should have slope 5, but I got %r" % (slope(lambda x: 5 * x, 10),)
assert abs(slope(lambda x: -2 * x + 1, 0) + 2) < 1e-3, "a line that falls 2 per 1 should have slope -2, but I got %r" % (slope(lambda x: -2 * x + 1, 0),)
print("slope() works: it measured a hill's steepness using nothing but the function itself.")`;
  const slopeSolution = `return (f(x + h) - f(x)) / h`;

  const descentCode = `a, b = 0.5, 0.5
lr = 0.2                            # learning rate: how big a step to take

for step in range(40):
    h = 0.0001
    slope_a = (loss(a + h, b) - loss(a, b)) / h      # wiggle dial A, see what the loss does
    slope_b = (loss(a, b + h) - loss(a, b)) / h      # wiggle dial B
    a = None                        # <- replace None: move dial A AGAINST its slope
    b = b - lr * slope_b

print('dials:', round(a, 3), round(b, 3), ' loss:', round(loss(a, b), 4))
`;
  const descentCheck = `_a = _n['vv'] / (_n['vv'] + _n['vc'])
_b = _n['cv'] / (_n['cv'] + _n['cc'])
_floor = loss(_a, _b)
assert isinstance(a, float), "a should be a number. Did you replace the None with a formula?"
assert loss(a, b) < loss(0.5, 0.5) - 0.01, "The loss barely moved (%.4f). Check the direction: we want to move AGAINST the slope." % loss(a, b)
assert loss(a, b) < _floor + 0.003, "Loss is %.4f, but the best possible is %.4f. Are you subtracting lr * slope_a?" % (loss(a, b), _floor)
print("Gradient descent found dials %.0f%% and %.0f%% in 40 steps." % (a * 100, b * 100))`;
  const descentSolution = `a = a - lr * slope_a`;
</script>

<Lesson id="ch5" num={5} title="The Wiggle" tagline="Which way should each dial turn? Nudge it a hair, and measure what the score does.">
  <Beat>
    <Pain title="Where we left off">
      <p>Random nudging told us only "better" or "worse", one bit per attempt, spent in a random direction. Every dial needs its own answer to a richer question: <strong>which way should I turn, and how much does it matter?</strong></p>
    </Pain>
    <p>You already think this way if you've ever run a business. A factory manager doesn't ask "is the profit good?" She asks: <em>"if I make <strong>one more</strong> unit, how much does profit change?"</em> Economists call that <strong>marginal</strong> change. If the answer is "up by £3", make more. If it's "down by £1", make fewer. If it's "about zero", you've found the sweet spot.</p>
    <p>We want exactly that, for every dial: <strong>"if I turn you up a hair, how much does the loss change?"</strong> That number is the dial's <strong>slope</strong>.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch5-up" kind="choice" options={['Up', 'Down', 'It stays about the same']} answer={0}>
      {#snippet q()}<p>Back to the two-dial vowel model, with both dials at 50%. The best setting is around A = 18%, B = 67%. <strong>If you raise dial A a little, from 50% to 51%, does the loss go up or down?</strong></p>{/snippet}
      <p>Up. Dial A is already <em>above</em> its best value (18%), so raising it moves you further away. Let's measure exactly how much, using nothing but the loss itself.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <h2>The wiggle test</h2>
    <p>The recipe has three steps: (1) note the loss where you are; (2) nudge the dial by a small amount <strong>h</strong> and note the loss again; (3) divide the <em>change in loss</em> by the <em>size of the nudge</em>. That ratio is "how much the loss moves per unit of dial".</p>
    <p>Below, the blue dot is where the dial sits and the orange dot is the nudged position. Start with a big nudge, then <strong>shrink h</strong> and watch what happens to the estimate.</p>
    <WiggleLab />
  </Beat>

  <Beat>
    <Aha title="It settles">
      <p>With a clumsy big nudge the estimate is off (we got 1.27 with a nudge of 0.4). But as the nudge shrinks, the estimate stops changing: <strong>0.628, 0.5435, 0.5359, 0.5350</strong>. It <em>settles on one number</em>: the slope at that exact spot. This is the central idea of calculus, and now you've seen it: the <strong>derivative is the number the wiggle test settles on as the wiggle gets tiny.</strong></p>
    </Aha>
    <p>The same idea, four ways. This is how we'll treat every piece of maths in this course: words first, then numbers, then code, then the symbol.</p>
    <div class="calc">
      <p><strong>In words:</strong> slope = (change in loss) ÷ (change in dial)</p>
      <p><strong>In numbers:</strong> raise A from 0.50 to 0.51 and the loss goes from 0.6931 to about 0.6985, so slope ≈ 0.0054 ÷ 0.01 ≈ <strong>0.54</strong></p>
      <p><strong>In code:</strong> <code>(f(x + h) - f(x)) / h</code></p>
      <p><strong>As a symbol:</strong> mathematicians write it <code>dL/dA</code>, said "dee-L dee-A": a tiny change in the loss per tiny change in A.</p>
    </div>
    <p>The sign carries the message. A slope of <strong>+0.54</strong> means "raising this dial raises the loss, so turn it <em>down</em>". A negative slope means "turn it <em>up</em>". Zero means flat.</p>
    <TimeTravel year="1600s" who="Newton and Leibniz">
      <p>Isaac Newton and Gottfried Leibniz, working separately in the late 1600s, both worked out this "settling" idea and built it into what we now call <strong>calculus</strong>. It was invented to answer questions about how fast things change, like planets, falling stones and, later, profit and loss. Three centuries on, it's what trains every AI model.</p>
    </TimeTravel>
  </Beat>

  <Beat gate>
    <h2>A compass for the dials</h2>
    <p>Now do it for <em>both</em> dials at once. The two slopes together say which way is downhill on the whole map. Click different spots and read the compass.</p>
    <Compass />
    <Sceptic q="You said flat means we've arrived. Couldn't flat also be the top of a hill?">
      <p>Yes, in general, and it's a fair worry. A flat spot can be a valley floor, a hilltop, or a ridge. For our two-dial problem the map is a single bowl, so flat means valley. For big models the landscape is far more complicated, and there are flat spots that aren't the bottom. The remarkable thing, which nobody fully predicted, is that following slopes downhill works extremely well in practice anyway. We'll keep an eye on this.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <h2>Gradient descent</h2>
    <p>Now the rule that powers (almost) all of modern AI. Wiggle each dial to get its slope, then <strong>step against the slope</strong>, by an amount scaled by a number called the <strong>learning rate</strong>:</p>
    <p class="calc"><span class="mono">new dial = old dial − learning rate × slope</span></p>
    <p>(The minus sign is because a positive slope means "raising me raises the loss", so we go the other way. The collection of all the slopes is called the <strong>gradient</strong>, hence <em>gradient descent</em>.) Take some steps. Then break it: push the learning rate up toward 1 and see what happens.</p>
    <DescentLab />
  </Beat>

  <Beat>
    <p>What happened, in our own test runs:</p>
    <ul>
      <li><strong>Learning rate 0.2</strong>: within 0.003 of the best in <strong>5 steps</strong>. Random nudging needed a median of 22 attempts.</li>
      <li><strong>Learning rate 0.05</strong>: safe but slow, about 19 steps.</li>
      <li><strong>Learning rate 0.7</strong>: overshoots back and forth, ringing like a bell, and takes about 11 steps.</li>
      <li><strong>Learning rate 0.9 or more</strong>: each step overshoots the valley and lands on the far wall, higher than before. The dials get thrown out of range and the loss becomes ∞. We say the training <strong>diverged</strong>.</li>
    </ul>
    <Aha title="You've now seen the same trade-off twice">
      <p>Too big and it overshoots, too small and it crawls. Chapter 4's nudge size and this chapter's learning rate are the same dilemma. Real training has the same fear, which is why <code>microgpt.py</code> chooses its learning rate carefully and shrinks it as training goes on.</p>
    </Aha>
    <p>And here's how it appears in <code>microgpt.py</code>. Each parameter has a <code>.grad</code>: that parameter's slope. After computing them all, the update line moves every parameter against its slope: <code>p.data -= lr_t * ...</code>. (The real file adds refinements to the basic rule, called <em>Adam</em>, which we'll invent in Chapter 15.)</p>
  </Beat>

  <Beat gate>
    <h2>Build it</h2>
    <p>First the wiggle test itself, as a recipe that works on <em>any</em> function:</p>
    <Cell id="ch5-slope" plain={PLAIN['ch5-slope']} title="Measure a slope" code={slopeCode} rows={7} check={slopeCheck} solution={slopeSolution}
      hint="Compare f at the nudged spot, f(x + h), with f at the original spot, f(x). Take the difference, then divide by the size of the nudge.">
      <p>One step is missing. Note that <code>slope</code> is handed a <em>function</em> <code>f</code>, so the same recipe measures steepness for anything.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <p>Now use it to train. This is the whole of gradient descent, in about eight lines. The loss function <code>loss(a, b)</code> for the two-dial model is provided.</p>
    <Cell id="ch5-descent" plain={PLAIN['ch5-descent']} title="Write gradient descent" code={descentCode} pre={VC} rows={13} check={descentCheck} solution={descentSolution}
      hint="The rule is: new dial = old dial minus learning rate times slope. The line for B shows the pattern.">
      <p>One step is missing: how to change dial A. The line for dial B is already there to guide you.</p>
    </Cell>
    <TimeTravel year="1847" who="Augustin-Louis Cauchy">
      <p>The French mathematician Cauchy described this very method in 1847: to find the lowest point, repeatedly step against the slope. He was motivated by astronomical calculations. It took until the computer age to apply it to millions of dials at once.</p>
    </TimeTravel>
  </Beat>

  <Beat>
    <Pain title="The price of a wiggle">
      <p>Slopes are exactly the information we wanted. But look at how we got them: to find the slope of <em>each</em> dial, we had to run the model <em>again</em>, once per dial. For two dials that's three loss tests per step (one for where we are, one per wiggle). For <code>microgpt.py</code> it's one per dial: <strong>4,065 tests for a single step</strong>.</p>
      <p>One pass of the real model over a single name takes about 11 ms (we measured). So a single update costs about <strong>4,065 × 11 ms ≈ 44 seconds</strong>, and 500 updates would take around <strong>6 hours</strong>. Yet the real file finishes a forward-and-slopes pass in about <strong>29 ms</strong>: roughly <em>1,500 times</em> faster.</p>
    </Pain>
    <p>So the slopes are the right <em>information</em> at the wrong <em>price</em>. Following slopes needs far fewer steps than random nudging (5 instead of 22 here), but each step costs one test per dial, which, with thousands of dials, eats all the savings. Somehow the real file gets every dial's slope for about the price of <em>two and a half</em> forward passes.</p>
    <Aha title="The trick, previewed">
      <p>It doesn't wiggle each dial separately. It works out how the loss depends on each dial by following the <em>chain of calculations</em> backwards from the loss to the dials, once. That trick is the <strong>chain rule</strong>, and it's the next chapter.</p>
    </Aha>
  </Beat>

  <Beat last>
    <Bank chapter="ch5">
      <ul>
        <li>A <strong>slope</strong> (derivative) says how much the loss changes per unit nudge of a dial. It's the number the <em>wiggle test</em> settles on as the wiggle shrinks.</li>
        <li>The <strong>sign</strong> says which way to turn: positive means turn down, negative means turn up, zero means flat.</li>
        <li><strong>Gradient descent</strong>: <code>new dial = old dial − learning rate × slope</code>, repeated.</li>
        <li><strong>Learning rate</strong> is a trade-off: too small crawls, too big overshoots and can diverge.</li>
        <li>The loop shape <code>for step in range(num_steps)</code> that every training run follows (lines 151–153).</li>
        <li>The catch: wiggling costs one extra test per dial, which is hopeless at 4,064 dials.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
