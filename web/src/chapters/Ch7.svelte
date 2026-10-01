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
  import CardLab from '../widgets/CardLab.svelte';
  import OrderPuzzle from '../widgets/OrderPuzzle.svelte';
  import TopoWalk from '../widgets/TopoWalk.svelte';
  import Translate from '../widgets/Translate.svelte';
  import FilePeek from '../widgets/FilePeek.svelte';
  import { ENGINE, ENGINE_VC } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const cardsCode = `def card(data, children=(), local_grads=(), label=''):
    # a card is a little record with four boxes (the label is just for drawing pictures later)
    return {'data': data, 'grad': 0, 'children': children, 'local_grads': local_grads, 'label': label}

def add(a, b):
    return card(a['data'] + b['data'], (a, b), (1, 1), '+')

def mul(a, b):
    return card(a['data'] * b['data'], (a, b), None, '×')    # <- replace None: the local slopes for (a, b)

z = mul(card(2.0), card(3.0))
print(z['data'], z['local_grads'])          # expect 6.0 and (3.0, 2.0)
`;
  const cardsCheck = `z = mul(card(2.0), card(3.0))
assert z['data'] == 6.0, "mul should multiply the two values: 2 x 3 = 6"
assert tuple(z['local_grads']) == (3.0, 2.0), "The local slopes should be (3.0, 2.0): each ingredient gets the OTHER one's value. I got %r" % (z['local_grads'],)
w = add(card(2.0), card(3.0))
assert tuple(w['local_grads']) == (1, 1), "add's local slopes should be (1, 1)"
print("Two kinds of card work. A card now remembers how it was made.")`;
  const cardsSolution = `return card(a['data'] * b['data'], (a, b), (b['data'], a['data']), '×')`;

  const topoCode = `def topo_order(root):
    order, seen = [], set()
    def visit(c):
        if id(c) not in seen:
            seen.add(id(c))
            for parent in c['children']:
                visit(parent)
            None          # <- replace None: the card's ingredients are all done. What do we do with the card?
    visit(root)
    return order
`;
  const topoCheck = `a, b = card(2.0), card(3.0)
c = mul(a, b); d = add(c, card(1.0)); loss = power(d, 2)
order = topo_order(loss)
assert len(order) == 6, "The list should contain all 6 cards, but it has %d. Is every card being added exactly once?" % len(order)
pos = {id(x): i for i, x in enumerate(order)}
for x in order:
    for p in x['children']:
        assert pos[id(p)] < pos[id(x)], "A card appears BEFORE one of the cards it was made from. Add the card only after visiting its ingredients."
assert order[-1] is loss, "The final card (the loss) should come last."
print("topo_order works: every card comes after its ingredients.")`;
  const topoSolution = `order.append(c)`;

  const backCode = `def backward(root):
    order = topo_order(root)
    root['grad'] = 1
    for c in reversed(order):
        for parent, local in zip(c['children'], c['local_grads']):
            parent['grad'] += None      # <- replace None: local slope times this card's slope
`;
  const backCheck = `a, b = card(2.0), card(3.0)
loss = power(add(mul(a, b), card(1.0)), 2)
backward(loss)
assert (a['grad'], b['grad']) == (42.0, 28.0), "Expected slopes 42 and 28 for a and b, but got %r and %r" % (a['grad'], b['grad'])
x = card(3.0); y = mul(x, x); backward(y)
assert x['grad'] == 6.0, "x times x should give x a slope of 6 (two paths of 3 each). I got %r. Are you ADDING each path's contribution with += ?" % (x['grad'],)
print("backward works: slopes 42 and 28 for the chain, and 6 for x times x.")`;
  const backSolution = `parent['grad'] += local * c['grad']`;

  const showCode = `a, b = card(2.0, label='a'), card(3.0, label='b')
c = mul(a, b)
d = add(c, card(1.0))
loss = power(d, 2)

backward(loss)
print('slopes:', a['grad'], b['grad'])
show(loss)             # draw the graph your code just built
`;

  const trainCode = `a, b = card(0.5), card(0.5)          # the two dials, as cards
lr = 0.2

for step in range(40):
    loss = vowel_loss(a, b)          # forward: builds a fresh graph of cards for the loss
    a['grad'] = 0                    # forget last step's slopes on the dials...
    b['grad'] = 0
    backward(loss)                   # backward: one sweep gives BOTH slopes
    a['data'] = None                 # <- replace None: move dial A against its slope
    b['data'] = b['data'] - lr * b['grad']

print('dials:', round(a['data'], 3), round(b['data'], 3), ' loss:', round(vowel_loss(a, b)['data'], 4))
`;
  const trainCheck = `import math
_a = _n['vv'] / (_n['vv'] + _n['vc']); _b = _n['cv'] / (_n['cv'] + _n['cc'])
_floor = -(_n['vv'] * math.log(_a) + _n['vc'] * math.log(1 - _a) + _n['cv'] * math.log(_b) + _n['cc'] * math.log(1 - _b)) / _N
assert isinstance(a['data'], float), "a['data'] should be a number. Did you replace the None?"
_now = vowel_loss(a, b)['data']
assert _now < 0.6931 - 0.01, "The loss barely moved (%.4f). Check the direction: move AGAINST the slope." % _now
assert _now < _floor + 0.003, "Loss is %.4f, but the best possible is %.4f." % (_now, _floor)
print("Trained with your own engine: dials %.0f%% and %.0f%%. No wiggling needed." % (a['data'] * 100, b['data'] * 100))`;
  const trainSolution = `a['data'] = a['data'] - lr * a['grad']`;

  const breakCode = `a, b = card(0.5), card(0.5)
lr = 0.2

for step in range(40):
    loss = vowel_loss(a, b)
    # (this time we forgot to reset a['grad'] and b['grad'] to 0 here!)
    backward(loss)
    a['data'] = a['data'] - lr * a['grad']
    b['data'] = b['data'] - lr * b['grad']
    print(step, round(a['data'], 3), round(b['data'], 3))
`;

  const classCode = `import math

class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), None)    # <- replace None: the local slopes

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

a, b = Value(2.0), Value(3.0)
loss = (a * b + 1) ** 2
loss.backward()
print(loss.data, a.grad, b.grad)
`;
  const classCheck = `assert (a.grad, b.grad) == (42.0, 28.0), "Expected slopes 42 and 28, but got %r and %r. Check the local slopes in __mul__." % (a.grad, b.grad)
x = Value(3.0); y = x * x; y.backward()
assert x.grad == 6.0, "x * x should give x a slope of 6, but got %r" % (x.grad,)
print("Your Value class works. It gives the same answers as the cards, and reads like ordinary arithmetic.")`;
  const classSolution = `return Value(self.data * other.data, (self, other), (other.data, self.data))`;
</script>

<Lesson id="ch7" num={7} title="The Machine" tagline="Numbers that remember how they were made, so a program can run the chain rule for us.">
  <Beat>
    <Pain title="Where we left off">
      <p>Last chapter we did the chain rule on a graph that <em>we</em> had drawn. The real model's loss comes from thousands of tiny steps, far too many to draw or to differentiate by hand. We need a <strong>program</strong> that builds the graph automatically as the arithmetic happens, and then runs the chain rule backwards by itself.</p>
    </Pain>
    <p>Think of a spreadsheet. Click a cell and Excel can show you its <em>precedents</em>: the cells it was calculated from. The spreadsheet remembers how every number was made. That's the whole idea of this chapter:</p>
    <Aha title="Numbers that remember">
      <p>Give every number a little <strong>memory card</strong>. When two cards are added or multiplied, the result is a <em>new card</em> that remembers who its parents were. Do enough arithmetic and you've built the computation graph without drawing anything, as a side effect of the calculation.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <Predict id="ch7-remember" kind="text" min={12} placeholder="e.g. its value, and what it was made from, and…">
      {#snippet q()}<p>For a number to later take part in a backward sweep, <strong>what does its card need to remember?</strong> Think about what the backward sweep did in Chapter 6, then list what you think the card needs.</p>{/snippet}
      <p>The backward sweep needs four things, and the card has exactly four boxes:</p>
      <ol>
        <li><strong>its value</strong>, which we call <code>data</code>;</li>
        <li><strong>its slope</strong>, which we call <code>grad</code> (short for gradient), starting at 0 and filled in during the backward sweep;</li>
        <li><strong>which cards it was made from</strong>, called <code>children</code> (the ingredients);</li>
        <li><strong>how sensitive it is to each of them</strong>, called <code>local_grads</code>: the local slopes from Chapter 6's table.</li>
      </ol>
    </Predict>
  </Beat>

  <Beat gate>
    <h2>Make some cards</h2>
    <p>Try it by hand first. Each time you do an operation, the engine makes a new card that remembers its ingredients. Build a few, press <strong>Run backward</strong>, and watch the slopes appear on the cards.</p>
    <CardLab />
  </Beat>

  <Beat gate>
    <h2>Build the cards yourself</h2>
    <p>Now let's write the card maker. A card is a small <strong>record</strong> (in Python, a <em>dictionary</em>: boxes with names). Then one recipe per kind of arithmetic, each producing a new card. Adding is done for you. Multiplying needs one decision: what local slopes should the new card remember?</p>
    <Cell id="ch7-cards" plain={PLAIN['ch7-cards']} title="Cards that remember" code={cardsCode} rows={13} check={cardsCheck} solution={cardsSolution}
      hint="From Chapter 6: when c = a × b, nudging a changes c by b per unit, and nudging b changes c by a per unit. So the slopes, in the order (a, b), are (b's value, a's value).">
      <p>One step is missing: the local slopes for multiplying.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <Pain title="A problem of order">
      <p>The backward sweep has a rule: a card must pass its slope back to its ingredients <strong>only after it has received all of its own slope</strong> from the cards that use it. Otherwise it would hand back an incomplete answer. So we need the cards in an order where <em>every card comes after its ingredients</em>; then walking that list <em>backwards</em> does the right thing.</p>
    </Pain>
    <p>It's like cooking: you can't ice the cake before baking it, or bake it before mixing the batter. Try the puzzle.</p>
    <OrderPuzzle />
  </Beat>

  <Beat gate>
    <p>A program can do this with a surprisingly small trick: a recipe that <strong>calls itself</strong>. To list a card, first ask each of its ingredients to list themselves, and only then add the card. Programmers call a recipe that calls itself <strong>recursion</strong>. It sounds scarier than it is, so step through one and watch the stack of unfinished jobs.</p>
    <TopoWalk />
    <Sceptic q="Isn't a recipe calling itself an infinite loop?">
      <p>It would be, except for two things that stop it. First, we keep a memory of cards we have already visited, so we never do any card twice. Second, the chain always bottoms out: a card with no ingredients (a plain input) has nothing to ask, so it simply adds itself. Each call hands a <em>smaller</em> job to the next one, like a boss delegating down to the intern who has nobody left to delegate to.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <Cell id="ch7-topo" plain={PLAIN['ch7-topo']} title="Line the cards up" code={topoCode} pre={ENGINE} rows={11} check={topoCheck} solution={topoSolution}
      hint="By the time the for-loop has finished, every ingredient has been visited and added. So now it is the card's own turn to join the list.">
      <p>One step is missing: what to do with a card once all its ingredients are done.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <h2>The backward sweep</h2>
    <p>Everything is in place. The backward sweep is the chain rule from Chapter 6, written as a loop: start the final card at slope 1, walk the list <em>backwards</em>, and for each card hand every ingredient <em>(local slope × the card's own slope)</em>, <strong>added</strong> to whatever it has already received (remember: a card used by two others collects from both).</p>
    <Cell id="ch7-backward" plain={PLAIN['ch7-backward']} title="Write the backward sweep" code={backCode} pre={ENGINE} rows={8} check={backCheck} solution={backSolution}
      hint="You are handing slope to the ingredient (parent). It receives (local slope) × (this card's slope), and you must ADD it to what is already there: use +=.">
      <p>One step is missing: the single line that does the chain rule.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <p>That's the whole engine: about 25 lines. Run your own engine on Chapter 6's example and see the graph <em>your code</em> built. The picture is drawn from your actual cards.</p>
    <Cell id="ch7-show" plain={PLAIN['ch7-show']} title="X-ray your own code" code={showCode} pre={ENGINE} rows={8}>
      <p>Run it. You should get slopes 42 and 28, and a picture of six cards.</p>
    </Cell>
  </Beat>

  <Beat>
    <Pain title="The payoff">
      <p>Chapter 4's two-dial model, scored on the real names, needed a graph of <strong>22 cards</strong> to compute its loss. Chapter 5 found each slope by wiggling, with extra runs. Your engine will get <strong>both slopes in a single backward sweep</strong> and train the model, with no wiggling at all.</p>
    </Pain>
  </Beat>

  <Beat gate>
    <Cell id="ch7-train" plain={PLAIN['ch7-train']} title="Train with your own engine" code={trainCode} pre={ENGINE_VC} rows={13} check={trainCheck} solution={trainSolution}
      hint="Gradient descent from Chapter 5: new value = old value − learning rate × slope. Dial B's line shows the pattern, and the slope now lives in the card's 'grad' box.">
      <p>One step is missing: how to move dial A. The function <code>vowel_loss</code>, which builds the loss out of cards, is provided.</p>
    </Cell>
    <Aha title="You've just built autograd">
      <p>It's called <strong>automatic differentiation</strong>, or autograd. The loss function was written as plain arithmetic, with no calculus, and the engine produced the exact slopes by itself. The same 25 lines, with a few more kinds of step, will train <code>microgpt.py</code>.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <Predict id="ch7-reset" kind="choice" options={['Nothing: the two lines are just tidying up', 'The old slopes would still be in the dial cards, and the new ones would be added on top', 'The program would crash straight away']} answer={1}>
      {#snippet q()}<p>Look at the two lines in the training loop that set the dials' slopes to 0. <strong>What would happen if we forgot them?</strong> (Remember: a backward sweep <em>adds</em> into the slope boxes.)</p>{/snippet}
      <p>The old slopes would stay in the boxes and the new ones pile on top. Each step's slopes would then be a running total of every step so far, and the dials would lurch. Let's see what that does.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <Cell id="ch7-break" plain={PLAIN['ch7-break']} title="Break it: forget to reset" code={breakCode} pre={ENGINE_VC} rows={11} allowError>
      <p>The same loop, minus the reset. Run it and watch the dials.</p>
    </Cell>
    <p>The first step is right. From then on, last step's slopes are still in the boxes, so they get added in and the dials overshoot. By step 4 dial A is <strong>1.24</strong>: a "probability" over 100%, which the log can't take, and the program crashes. The cure is the single line we've been writing, <code>grad = 0</code>, and that is why it appears in <code>microgpt.py</code> too (it's <code>p.grad = 0</code> at the end of every update).</p>
  </Beat>

  <Beat>
    <h2>Python's nicer clothes</h2>
    <p>Our engine works, but it reads clumsily: <code>power(add(mul(a, b), card(1.0)), 2)</code> for what is really <code>(a × b + 1)²</code>. Python has a feature that lets you teach it what <code>+</code> and <code>×</code> mean for your own kind of record, so that arithmetic reads like arithmetic. That feature is called a <strong>class</strong>:</p>
    <ul>
      <li>A <strong>class</strong> is a <em>template</em> for records, like a blank invoice. We'll call ours <code>Value</code>.</li>
      <li>Each <code>Value</code> is one filled-in <em>card</em> made from the template.</li>
      <li>Special recipe names such as <code>__add__</code> say what <code>+</code> means for these cards. They're written with double underscores so they don't clash with your own names.</li>
    </ul>
    <p>Nothing about the engine changes. It's the same idea, in different clothes. Compare:</p>
    <Translate />
  </Beat>

  <Beat gate>
    <p>Here is the <strong>whole</strong> class version: the card, its arithmetic, and the backward sweep. You've written every idea in it already. There is one step missing, the multiplication's local slopes, which you know.</p>
    <Cell id="ch7-class" plain={PLAIN['ch7-class']} title="The same engine, as a class" code={classCode} rows={42} check={classCheck} solution={classSolution}
      hint="Same as before: nudging self changes the product by other.data per unit, and nudging other changes it by self.data. In the order (self, other):">
      <p>One step is missing. Then notice how <code>loss = (a * b + 1) ** 2</code> reads like ordinary arithmetic.</p>
    </Cell>
  </Beat>

  <Beat>
    <h2>The rest of the class</h2>
    <p>The real <code>Value</code> in <code>microgpt.py</code> has a few more methods, and they're all the same recipe with a different local slope, or a convenient rewrite of one you know:</p>
    <table class="tbl">
      <thead><tr><td>Method</td><td>Meaning</td><td>Local slope or trick</td></tr></thead>
      <tbody>
        <tr><td><code>log()</code></td><td>logarithm</td><td><code>1 / x</code> (from Chapter 6)</td></tr>
        <tr><td><code>exp()</code></td><td>"e to the power x"</td><td>its own value: the slope of eˣ is eˣ</td></tr>
        <tr><td><code>relu()</code></td><td>a one-way gate: keep x if positive, else 0</td><td><code>1</code> if x was positive, else <code>0</code> (blocked)</td></tr>
        <tr><td><code>__neg__</code>, <code>__sub__</code></td><td>−x and a − b</td><td>just rewrites: −x is x × −1, and a − b is a + (−b), so no new slopes are needed</td></tr>
        <tr><td><code>__truediv__</code></td><td>a ÷ b</td><td>rewritten as a × b<sup>−1</sup></td></tr>
        <tr><td><code>__radd__</code>, <code>__rmul__</code>, …</td><td>when the <em>plain number</em> comes first, as in <code>3 + v</code></td><td>Python first asks the number, which has no idea what a Value is, then falls back to the Value's "reversed" method</td></tr>
      </tbody>
    </table>
    <p>That's the entire class. Open the file: lines 30–72 are all yours now.</p>
    <FilePeek />
    <TimeTravel year="2020" who="micrograd">
      <p>This design comes from <strong>micrograd</strong>, a tiny autograd engine Andrej Karpathy wrote to show that backpropagation is nothing mysterious. <code>microgpt.py</code> uses the same idea as its foundation, which is why the file can say "everything else is just efficiency".</p>
    </TimeTravel>
  </Beat>

  <Beat last>
    <Bank chapter="ch7">
      <ul>
        <li>A number that <strong>remembers how it was made</strong>: its value, its slope, its ingredients, and its local slopes.</li>
        <li>Doing arithmetic on such numbers builds the <strong>computation graph</strong> automatically.</li>
        <li>Lining the cards up (ingredients first) with a <strong>recursive</strong> recipe, then sweeping <strong>backwards</strong> adding <code>local slope × this card's slope</code>.</li>
        <li>Why slopes must be <strong>reset to zero</strong> before each backward sweep.</li>
        <li>A <strong>class</strong> as a template for cards, and <code>__add__</code> etc. as "what + means". That's the whole <code>Value</code> class (lines 30–72).</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
