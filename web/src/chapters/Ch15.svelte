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
  import OptRace from '../widgets/OptRace.svelte';
  import EmaLab from '../widgets/EmaLab.svelte';
  import StepScale from '../widgets/StepScale.svelte';
  import LrDecay from '../widgets/LrDecay.svelte';
  import { ADAMFN } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const adamCode = `def adam_step(p, grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.95, eps=1e-8):
    m = None                                        # <- replace None: keep beta1 of the old m, mix in (1 - beta1) of the new slope
    v = beta2 * v + (1 - beta2) * grad ** 2         # the same for the squared slope: how big this dial's slopes typically are
    m_hat = m / (1 - beta1 ** t)                    # start-up corrections (t = which step this is, counting from 1)
    v_hat = v / (1 - beta2 ** t)
    p = None                                        # <- replace None: step against the smoothed slope, scaled by its typical size
    return p, m, v
`;
  const adamCheck = `def _ref(grads, lr=0.01):
    p, m, v = 1.0, 0.0, 0.0
    for t, g in enumerate(grads, 1):
        m = 0.9 * m + 0.1 * g
        v = 0.95 * v + 0.05 * g * g
        p = p - lr * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.95 ** t)) ** 0.5 + 1e-8)
    return p
grads = [0.5, -0.2, 0.8, 0.3, -0.1]
p, m, v = 1.0, 0.0, 0.0
for t, g in enumerate(grads, 1):
    p, m, v = adam_step(p, g, m, v, t)
assert abs(p - _ref(grads)) < 1e-12, "After 5 steps p should be %.10f, but I got %.10f. Check both the momentum line and the update line." % (_ref(grads), p)
big = adam_step(0.0, 1000.0, 0.0, 0.0, 1)[0]
small = adam_step(0.0, 0.001, 0.0, 0.0, 1)[0]
assert abs(big + 0.01) < 1e-4 and abs(small + 0.01) < 1e-4, "A dial should move by about lr (0.01) on its first step, whatever its slope size."
print("adam_step works. A slope of 1000 and a slope of 0.001 both move the dial by about 0.01. Lines 175-182 of microgpt.py.")`;
  const adamSolution = `m = beta1 * m + (1 - beta1) * grad
p = p - lr * m_hat / (v_hat ** 0.5 + eps)`;

  const raceCode = `# The valley from the page. The steep dial (y) is 1000 times steeper than the shallow one (x).
B = 1000
x, y = -3.0, 1.2
mx = my = vx = vy = 0.0                 # each dial keeps its own running averages

for t in range(1, 3001):
    gx, gy = x, B * y                   # the two slopes
    x, mx, vx = adam_step(x, gx, mx, vx, t, lr=0.1)
    y, my, vy = adam_step(y, gy, my, vy, t, lr=0.1)
    if 0.5 * (x * x + B * y * y) < 0.001:
        print('Adam solved it in', t, 'steps')
        break
`;
  const raceCheck = `assert t == 85, "Adam should solve this valley in 85 steps (the page's race says so), but it took %d. Check your adam_step." % t
print("Your Adam solves the valley in 85 steps. Plain gradient descent, at its best, needed 2,335.")`;
</script>

<Lesson id="ch15" num={15} title="Adam" tagline="One speed limit for every dial is a disaster. Give each dial its own, and smooth out the noise.">
  <Beat>
    <Pain title="Where we left off">
      <p>We have a model with 4,064 dials, a loss, and a way to get every dial's slope. In Chapter 5 we trained with plain gradient descent: <code>dial = dial − learning rate × slope</code>. Real models break it. Not because the idea is wrong, but because of one assumption it makes without telling you: <strong>that one learning rate suits every dial.</strong></p>
    </Pain>
    <p>It doesn't. Some dials sit on steep slopes (a nudge changes the loss a lot) and others on shallow ones. The slope sizes can differ by hundreds of times. It's like setting the speed limit for a whole motorway by its <em>sharpest bend</em>: safe, but everywhere else you crawl.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch15-scale" kind="choice" options={['Both dials step by about the same amount', 'The steep dial steps about 1,000 times further than the shallow one', 'The shallow dial steps further, to compensate']} answer={1}>
      {#snippet q()}<p>Plain gradient descent steps each dial by <code>learning rate × slope</code>. Suppose dial A's slope is <strong>1,000 times larger</strong> than dial B's. <strong>How do their step sizes compare?</strong></p>{/snippet}
      <p>A steps about 1,000 times further. The rule multiplies by the slope, so a steeper dial gets a bigger step. To stop the steep dial from overshooting and exploding, you must shrink the learning rate for <em>everyone</em>, and then the shallow dial moves at a snail's pace. Let's watch that happen.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <h2>The race</h2>
    <p>A valley with two dials. The steep dial is 30, 1,000, or 100,000 times steeper than the shallow one. Three optimisers start from the same place and race to the bottom: plain gradient descent (given its <em>best</em> allowed learning rate), momentum, and <strong>Adam</strong>. The chart shows the loss against steps on log scales.</p>
    <OptRace />
  </Beat>

  <Beat>
    <p>What the race shows (the numbers on the screen, measured by the page):</p>
    <ul>
      <li><strong>On a mild valley (30×), plain descent is fine.</strong> At its best it needs 68 steps; Adam needs 63.</li>
      <li><strong>As the steep dial gets steeper, plain descent collapses.</strong> At 1,000× it needs <strong>2,335 steps</strong> while Adam needs <strong>85</strong>. At 100,000× plain descent <em>doesn't finish in 3,000 steps</em>; Adam needs <strong>132</strong>.</li>
      <li><strong>Plain descent has a cliff.</strong> Raise its learning rate just past the limit (the 1.0× mark on the slider) and it explodes. Adam kept working with the <em>same</em> learning rate, 0.1, at every steepness. (On the mild valley it reached the bottom for each of the five learning rates we tried, from 0.05 up to 1.0.)</li>
    </ul>
    <Aha title="Adam is plain descent with two upgrades">
      <p><strong>1. Momentum</strong> smooths the slope over recent steps, so noise and zigzag cancel out. <strong>2. Per-dial scaling</strong> divides each dial's step by how big its slopes <em>typically</em> are, so steep and shallow dials move about equally. Name: <strong>Ada</strong>ptive <strong>m</strong>oment estimation. Let's build each upgrade.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <h2>Upgrade 1: momentum, a moving average</h2>
    <p>The slope you measure at any one step is <strong>noisy</strong>. (Remember, the real model sees one name at a time, and each name pulls the dials its own way.) A business analyst facing noisy weekly sales doesn't react to one week; she looks at a <strong>moving average</strong>. Same here. Keep a running figure <code>m</code>: each step, keep most of the old value and mix in a little of the new slope. The "memory" β₁ (0.9 in <code>microgpt.py</code>) says how much old value to keep.</p>
    <EmaLab />
    <Aha title="The start-up correction">
      <p>Because <code>m</code> starts at 0, the first few values are dragged down towards zero (the blue line climbs from the floor instead of starting at the right level). Tick the box: dividing by <code>(1 − β₁<sup>step</sup>)</code> fixes it exactly, and the correction fades away on its own as the steps pile up. That's the <code>m_hat</code> in the code.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <h2>Upgrade 2: each dial's own step size</h2>
    <p>Do the same moving-average trick on the <em>square</em> of the slope, <code>v</code>. That tells you how big this dial's slopes typically are. Then divide the step by its square root, <code>√v</code>. A dial whose slopes are huge gets a smaller step; one whose slopes are tiny gets a bigger one. Drag the two slopes apart:</p>
    <StepScale />
  </Beat>

  <Beat gate>
    <h2>Build it</h2>
    <p>Put both upgrades together. Here's one Adam step for a single dial. You need it twice-removed from anything mysterious: two moving averages, two corrections, one update.</p>
    <Cell id="ch15-adam" plain={PLAIN['ch15-adam']} title="One Adam step" code={adamCode} rows={10} check={adamCheck} solution={adamSolution}
      hint="First blank: the moving average, keeping beta1 of the old m and (1 - beta1) of the new slope. Second blank: p minus lr times m_hat, divided by (the square root of v_hat plus eps). The v line above shows the pattern for the first one.">
      <p>Two steps are missing: the momentum line and the final update. Lines 175–182 of <code>microgpt.py</code> do this for every dial, looping over the list <code>params</code>.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <p>Now use your own Adam on the valley from the race, with the steep dial 1,000 times steeper. The page said Adam needs 85 steps. Does yours?</p>
    <Cell id="ch15-race" plain={PLAIN['ch15-race']} title="Your Adam, in the race" code={raceCode} pre={ADAMFN} rows={13} check={raceCheck}>
      <p>Run it. (Plain gradient descent at its best needed 2,335 steps for this same valley.)</p>
    </Cell>
    <TimeTravel year="1964 to 2014" who="Polyak, Hinton, Kingma and Ba">
      <p>Momentum was proposed by Boris Polyak in 1964 (he called it the "heavy ball" method). Scaling each dial's step by its typical slope size was popularised by Geoffrey Hinton in a 2012 lecture (the method is called RMSProp). In 2014, Diederik Kingma and Jimmy Ba combined the two, added the start-up corrections, and called it <strong>Adam</strong>. It became the default way to train neural networks, and it's still the optimiser behind most of them.</p>
    </TimeTravel>
  </Beat>

  <Beat gate>
    <h2>One more thing: shrink the steps as you go</h2>
    <p>With a fixed learning rate, even Adam keeps jittering near the bottom (on the mild valley it kept bouncing at a loss around 10<sup>−4</sup>, never settling). The cure is the one you already know from Chapters 4 and 5: big steps early when you're far from a good setting, small careful steps late. <code>microgpt.py</code> does it with a smooth curve called a <strong>cosine schedule</strong>: the learning rate follows half a cosine wave from its full value down to zero by the last step (line 175). Slide through training:</p>
    <LrDecay />
  </Beat>

  <Beat>
    <h2>The loop, in the file</h2>
    <p>Here are the three lines of setup (147–149: the learning rate, the two memories β₁ = 0.9 and β₂ = 0.95, the safety ε, and the two lists <code>m</code> and <code>v</code> that hold a running average for <em>every one of the 4,064 dials</em>) and the nine lines of update (174–182). They are the code you just wrote, applied to each dial in turn:</p>
    <ol class="steps">
      <li><code>lr_t = learning_rate * 0.5 * (1 + math.cos(…))</code>: today's decayed learning rate (line 175).</li>
      <li>for each dial: update its two running averages (177, 178), correct them (179, 180), and step: <code>p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)</code> (181).</li>
      <li><code>p.grad = 0</code>: wipe the slope, ready for the next step (182). You met this line in Chapter 7: forget it and the slopes pile up.</li>
    </ol>
    <Sceptic q="Why do we need a tiny eps (1e-8) in the denominator?">
      <p>If a dial's slopes were exactly zero for a while, √v̂ would be zero and we'd divide by zero. The tiny ε makes that division safe without changing the result noticeably in the normal case. It's the same kind of safety cushion as the 1e-5 in RMSNorm.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <Pain title="Almost there">
      <p>Look at what we have: a model that turns a letter into 27 scores (Chapter 14), a loss that grades them (Chapters 3 and 8), an engine that works out all 4,064 slopes (Chapters 6 and 7), and an optimiser that turns the slopes into good steps (this chapter). One thing is left: <strong>the loop</strong> that feeds it names, one after another, and presses the button thousands of times. It's short. And in the next chapter you'll run the real thing, in your browser, and watch gibberish turn into names.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch15">
      <ul>
        <li>Plain gradient descent uses <strong>one learning rate for every dial</strong>, so it must be tiny to keep the steepest dial from exploding. On a valley with a 1,000× steeper dial it needs 2,335 steps; Adam needs 85.</li>
        <li><strong>Momentum</strong>: a moving average of the slope (<code>m = β₁·m + (1 − β₁)·slope</code>) smooths out noise.</li>
        <li><strong>Per-dial scaling</strong>: a moving average of the squared slope (<code>v</code>), then divide the step by <code>√v</code>, so every dial moves about the learning rate per step.</li>
        <li><strong>Start-up corrections</strong> (<code>m_hat</code>, <code>v_hat</code>) fix the averages' slow start from zero.</li>
        <li><strong>Cosine decay</strong>: the learning rate shrinks smoothly to zero over training.</li>
        <li>In the file: setup on lines 146–149 and the update on lines 174–182.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
