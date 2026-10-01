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
  import Funnel from '../widgets/Funnel.svelte';
  import GraphLab from '../widgets/GraphLab.svelte';
  import { BACKFNS } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const backCode = `def add_backward(upstream):
    # c = a + b. Raise a by 1 and c rises by 1. Same for b.
    grad_a = None        # <- replace None
    grad_b = upstream
    return grad_a, grad_b

def mul_backward(a, b, upstream):
    # c = a * b. Raise a by 1 and c rises by b. Raise b by 1 and c rises by a.
    grad_a = None        # <- replace None
    grad_b = a * upstream
    return grad_a, grad_b

print(mul_backward(2, 3, 14))     # the chain example: should print (42, 28)
`;
  const backCheck = `assert add_backward(5) == (5, 5), "add_backward(5) should give (5, 5), but gave %r" % (add_backward(5),)
assert mul_backward(2, 3, 14) == (42, 28), "mul_backward(2, 3, 14) should give (42, 28), but gave %r" % (mul_backward(2, 3, 14),)
assert mul_backward(4, -1, 2) == (-2, 8), "mul_backward(4, -1, 2) should give (-2, 8), but gave %r" % (mul_backward(4, -1, 2),)
print("Both backward rules work. These are the only two you need for most of a neural network's arithmetic.")`;
  const backSolution = `grad_a = upstream          # in add_backward
grad_a = b * upstream      # in mul_backward`;

  const verifyCode = `a, b = 2.0, 3.0

# forward: compute the loss
c = a * b
d = c + 1
loss = d ** 2

# backward: slopes, from the end back to the start
grad_d = 2 * d                                  # the slope of d squared is 2d
grad_c, _ = add_backward(grad_d)                # d = c + 1
grad_a, grad_b = mul_backward(a, b, grad_c)     # c = a * b
print('going backwards:', grad_a, grad_b)

# check it with the wiggle test
h = 1e-6
print('wiggle test:    ', (((a + h) * b + 1) ** 2 - loss) / h, ((a * (b + h) + 1) ** 2 - loss) / h)
`;
</script>

<Lesson id="ch6" num={6} title="Chains" tagline="The loss sits at the end of a long chain of steps. A dial's slope is just the product of the steps in between.">
  <Beat>
    <Pain title="Where we left off">
      <p>We know what we want: <strong>every dial's slope, all at once, cheaply.</strong> Wiggling each dial separately costs one extra run of the model <em>per dial</em>, which is hopeless at 4,064 dials.</p>
    </Pain>
    <p>But look at how the model is actually built. A dial doesn't touch the loss directly. It's multiplied by something, added to something, squashed, multiplied again, normalised, and eventually, many steps later, it feeds the loss. <strong>The loss sits at the end of a long chain of small steps.</strong> And chains have a wonderful property: if you know how each <em>link</em> behaves, you can work out the whole chain without ever wiggling the dial.</p>
    <p>You already know this property from business. Let's start there.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch6-double" kind="choice" options={['Clicks per £ of spend', 'The share of clicks who sign up', 'The share of signups who pay', '£ per paying customer', 'Any one of them: they all do the same']} answer={4}>
      {#snippet q()}<p>A marketing funnel: <em>ad spend → clicks → signups → paying customers → revenue</em>. Each stage converts the last at some ratio. Suppose you could <strong>double just one</strong> of the four ratios. Which doubles the revenue you get from each extra pound of ad spend?</p>{/snippet}
      <p>Any of them. The revenue from an extra pound is the <em>product</em> of all four ratios, and doubling any one factor of a product doubles the product. Play with the funnel below to see it.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <Funnel />
  </Beat>

  <Beat>
    <Aha title="That is the chain rule">
      <p>The effect of the <em>first</em> thing on the <em>last</em> thing is the <strong>product of the conversion ratios along the way</strong>. That's the whole idea. Mathematicians call each ratio a <strong>local slope</strong> (how much one step's output changes per unit change of its input), and the product rule for chains the <strong>chain rule</strong>.</p>
    </Aha>
    <p>Same idea, four ways again:</p>
    <div class="calc">
      <p><strong>In words:</strong> the slope of the end with respect to the start = the slope of each link, multiplied together</p>
      <p><strong>In numbers:</strong> 0.8 × 0.10 × 0.25 × £40 = <strong>£0.80</strong> of revenue per extra £1 of spend</p>
      <p><strong>In code:</strong> <code>slope = 1</code>, then for each link going back, <code>slope = slope * link_slope</code></p>
      <p><strong>As a symbol:</strong> <code>dR/dS = dR/dC × dC/dG × dG/dS</code>, which reads like cancelling fractions, and that's a good way to remember it.</p>
    </div>
    <p>One catch. A funnel's ratios stay fixed. But the steps inside a neural network <em>don't</em>: squaring a number multiplies small changes by 2 if the number is 1, but by 20 if the number is 10. The ratio depends on <strong>where you are</strong>. So each link's slope has to be evaluated <em>at the current values</em>. That's why we always run the model forwards first, remember every value, and only then work backwards.</p>
  </Beat>

  <Beat>
    <h2>The five link-slopes you'll need</h2>
    <p>Neural networks are built from very few kinds of step. Each has a simple local slope, and you can check every one of them with the wiggle test:</p>
    <table class="tbl">
      <thead><tr><td>Step</td><td>What it does</td><td>Local slope (per unit change of input)</td></tr></thead>
      <tbody>
        <tr><td><code>c = a + b</code></td><td>add</td><td>1 for <code>a</code>, 1 for <code>b</code></td></tr>
        <tr><td><code>c = a × b</code></td><td>multiply</td><td><code>b</code> for <code>a</code>, <code>a</code> for <code>b</code> (each gets the <em>other</em> one)</td></tr>
        <tr><td><code>c = a²</code></td><td>square</td><td><code>2a</code></td></tr>
        <tr><td><code>c = log(a)</code></td><td>logarithm</td><td><code>1 / a</code></td></tr>
        <tr><td><code>c = −a</code></td><td>negate</td><td><code>−1</code></td></tr>
      </tbody>
    </table>
    <p>If you scroll back up to the <code>Value</code> class in <code>microgpt.py</code> (lines 39–57), you'll see this table written as code (plus two more steps, <code>exp</code> and <code>relu</code>, that we'll meet later): each operation stores its result <em>and</em> its local slopes. That's the whole trick, and in Chapter 7 you'll write it yourself.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch6-guess" kind="number" answer={42} tolerance={14} unit="">
      {#snippet q()}<p>Here's a three-step chain: <span class="mono">c = a × b</span>, then <span class="mono">d = c + 1</span>, then <span class="mono">loss = d²</span>. With <strong>a = 2 and b = 3</strong>: c = 6, d = 7, loss = 49. <strong>What's the slope of the loss with respect to a?</strong> In other words, if you nudge a up a hair, how many times that nudge does the loss move? (A guess is fine. We'll compute it properly next.)</p>{/snippet}
      <p>It's <strong>42</strong>. The loss moves 42 times as fast as a does. You'll see exactly why in a moment: three local slopes, multiplied: <strong>14</strong> (for squaring at 7) × <strong>1</strong> (for adding) × <strong>3</strong> (for multiplying, the other input).</p>
    </Predict>
  </Beat>

  <Beat gate>
    <h2>Watch the chain rule run backwards</h2>
    <p>This is the algorithm at the heart of every neural network, and you can step through it. The forward pass has already filled in every value. Now press <strong>Next step</strong>: the slope of the loss with respect to itself is 1, and it flows <em>backwards</em>, each link multiplying by its own local slope. Check the answer against the wiggle test at the end. Then switch to the other two examples.</p>
    <GraphLab />
  </Beat>

  <Beat>
    <Aha title="One input used twice: the paths add">
      <p>In the second example, <span class="mono">loss = a × a</span>, the input <code>a</code> feeds the multiplication <em>twice</em>. Each use sends slope back, and the slopes <strong>add</strong>: 3 + 3 = 6, which is 2a. If a dial is used in several places in a model (and in <code>microgpt.py</code> they all are), its slope is the sum over every path. That's why the real code says <code>child.grad += …</code> with a <code>+=</code>, and why every <code>grad</code> starts at 0.</p>
    </Aha>
    <Sceptic q="Why go backwards? Couldn't we push slopes forward instead?">
      <p>You could, and it's a real technique ("forward mode"). But it answers a different question: <em>how does everything change when I nudge this one input?</em>, so you'd need one pass <strong>per input</strong>. We have one output (the loss) and thousands of inputs (the dials). Going backwards answers <em>how does the loss change with respect to every input at once</em>, in a single pass. When there are many inputs and one output, backwards wins by a factor of the number of inputs.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <h2>Write the two rules</h2>
    <p>Each kind of step needs a small "backward rule": given the slope arriving from downstream (the <strong>upstream</strong> slope), work out the slope to hand back to each input. Here are the two you need most:</p>
    <Cell id="ch6-backward" plain={PLAIN['ch6-backward']} title="Write the backward rules" code={backCode} rows={15} check={backCheck} solution={backSolution}
      hint="Each rule is the local slope times the upstream slope. For adding, the local slope is 1. For multiplying, a's local slope is b.">
      <p>Two steps are missing: one for adding and one for multiplying.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <p>Now chain your two rules together by hand, exactly as the page just did, and compare with the wiggle test:</p>
    <Cell id="ch6-verify" plain={PLAIN['ch6-verify']} title="Backwards versus the wiggle test" code={verifyCode} pre={BACKFNS} rows={16}>
      <p>Run it. The two rows should agree: 42 and 28, up to a rounding whisker.</p>
    </Cell>
  </Beat>

  <Beat>
    <Pain title="The payoff">
      <p>Count the work. The backward pass visits <strong>each step once</strong>, doing a multiplication or two per link. So its cost is about the same as the forward pass, no matter how many dials there are. We measured it on the real file:</p>
      <ul>
        <li>Forward pass over one name: <strong>≈ 11 ms</strong></li>
        <li>Forward <em>and</em> backward (every dial's slope): <strong>≈ 29 ms</strong></li>
        <li>The wiggle approach for the same slopes: 4,065 forward passes ≈ <strong>44 seconds</strong></li>
      </ul>
      <p>All 4,064 slopes for about <strong>2.6 forward passes</strong>, which makes it about 1,500× cheaper. That is why neural networks are trainable at all.</p>
    </Pain>
    <TimeTravel year="1970 and 1986" who="Backpropagation">
      <p>Seppo Linnainmaa, a Finnish master's student, described this backwards-chain-rule method for general computations in 1970. In 1986, David Rumelhart, Geoffrey Hinton and Ronald Williams showed in a famous paper that it could train neural networks to learn useful internal representations, and it became known as <strong>backpropagation</strong>. It has been the engine of nearly every AI breakthrough since.</p>
    </TimeTravel>
    <p>You've just understood it. There's no more maths in backpropagation than <em>multiply the local slopes along each path, and add up the paths</em>. What's left is bookkeeping: remembering the values, the local slopes and which step fed which. The bookkeeping is a program, and writing it is the next chapter.</p>
  </Beat>

  <Beat last>
    <Bank chapter="ch6">
      <ul>
        <li>The loss is the end of a <strong>chain</strong> of small steps. Each step has a <strong>local slope</strong>, worked out at the current values.</li>
        <li>The <strong>chain rule</strong>: the slope of the end with respect to the start is the <em>product</em> of the local slopes along the way.</li>
        <li>If a dial feeds the loss by several paths, <strong>add</strong> the paths' contributions.</li>
        <li>Going <strong>backwards</strong> gets every dial's slope in one sweep, at about the cost of one forward pass: ~29 ms instead of ~44 s.</li>
        <li>The one-line summary of what <code>microgpt.py</code> does here (line 29): "apply the chain rule recursively across a computation graph".</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>

<style>
  :global(.tbl) { width: 100%; border-collapse: collapse; font-family: var(--font-ui); font-size: 0.9rem; margin: 1rem 0; }
  :global(.tbl td) { padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--line); vertical-align: top; }
  :global(.tbl thead td) { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
</style>
