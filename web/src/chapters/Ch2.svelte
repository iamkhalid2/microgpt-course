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
  import LetterBars from '../widgets/LetterBars.svelte';
  import Wheel from '../widgets/Wheel.svelte';
  import NameMaker from '../widgets/NameMaker.svelte';
  import BigramGrid from '../widgets/BigramGrid.svelte';
  import ChainWalk from '../widgets/ChainWalk.svelte';
  import FakeOrFact from '../widgets/FakeOrFact.svelte';
  import ContextWall from '../widgets/ContextWall.svelte';
  import { VOCAB, COUNTS, TABLE } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const countCode = `# Count how often each letter appears in the names.
counts = {ch: 0 for ch in uchars}      # a dictionary: {'a': 0, 'b': 0, 'c': 0, ...}

for doc in docs:
    for ch in doc:
        None        # <- replace None: add 1 to this letter's count

print('a:', counts['a'], '  e:', counts['e'], '  q:', counts['q'])
`;
  const countCheck = `want = {c: sum(d.count(c) for d in docs) for c in uchars}
assert counts != {c: 0 for c in uchars}, "All the counts are still zero. Inside the loop, add 1 to counts[ch]."
assert counts == want, "Close, but some counts are off. Each letter of each name should add exactly 1 to its own count."
print("Counted %s letters in all. That table of counts is your first model." % format(sum(counts.values()), ','))`;
  const countSolution = `counts[ch] += 1`;

  const choicesCode = `import random

weights = [counts[ch] for ch in uchars]          # one weight per letter: bigger count = bigger wedge
print(random.choices(uchars, weights=weights, k=12))
`;

  const tableCode = `V = len(uchars) + 1                      # 27 symbols: 26 letters + BOS
table = [[0] * V for _ in range(V)]      # a 27 x 27 grid of zeros. table[prev][nxt]

for doc in docs:
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    for prev, nxt in zip(tokens, tokens[1:]):      # walk the name two symbols at a time
        None        # <- replace None: add 1 to table[prev][nxt]

print('names that start with a:', table[BOS][0])
print("'q' is followed by 'u' this many times:", table[uchars.index('q')][uchars.index('u')])
`;
  const tableCheck = `want = [[0] * V for _ in range(V)]
for d in docs:
    t = [BOS] + [uchars.index(c) for c in d] + [BOS]
    for a, b in zip(t, t[1:]):
        want[a][b] += 1
assert table != [[0] * V for _ in range(V)], "The table is still all zeros. Inside the loop, add 1 to the cell table[prev][nxt]."
assert table == want, "Nearly. Every neighbouring pair (prev, nxt) should add exactly 1 to table[prev][nxt]."
print("Table done: %s pairs counted into %d cells." % (format(sum(map(sum, table)), ','), V * V))`;
  const tableSolution = `table[prev][nxt] += 1`;

  const nameCode = `import random

def make_name():
    tok = BOS                          # always start from the start symbol
    name = ''
    while True:
        tok = random.choices(range(V), weights=table[tok])[0]    # spin the wheel for this symbol's row
        if tok == BOS:
            None                       # <- replace None: what should happen when the name ends?
        name += uchars[tok]
    return name

for _ in range(10):
    print(make_name())
`;
  const nameCheck = `out = [make_name() for _ in range(400)]
assert all(n.isalpha() for n in out if n), "Names should only contain letters."
avg = sum(map(len, out)) / len(out)
assert 3 < avg < 9, "Average length is %.1f. Real names average about 6, so something's off with how the name ends." % avg
print("make_name() works. Average length %.1f letters (real names: 6.1)." % avg)`;
  const nameSolution = `break`;
</script>

<Lesson id="ch2" num={2} title="Habits" tagline="Random is hopeless. Real names have habits, and habits can be counted.">
  <Beat gate>
    <Pain title="Where we left off">
      <p>Our random generator writes <code>kyxqtrgrj</code>. Real names don't. So real names must <em>prefer</em> some letters and some pairings over others. If we could measure those preferences, we could generate from them.</p>
    </Pain>
    <Predict id="ch2-common" kind="choice" options={['e', 'a', 'n', 'i']} answer={1}>
      {#snippet q()}<p>Let's measure. Across all 32,033 names, <strong>which letter appears most often?</strong></p>{/snippet}
      <p><strong>a</strong>, at 14.9% of all the symbols, ahead of <code>e</code> at 9%. Many of the most common names end in <em>a</em> (emma, olivia, ava, isabella, sophia…), which inflates it. Look at the chart and watch for a second thing: the bar on the far right.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <p>Here's every letter, counted. Play with the two switches: <em>sort it</em>, and <em>switch from counts to shares</em>.</p>
    <LetterBars />
  </Beat>

  <Beat>
    <p>That switch from "how many times" to "share of all" is the most important move in this whole chapter, so let me name it. A <strong>probability</strong> is just a count divided by the total:</p>
    <p class="calc"><span class="mono">33,885 times a appears ÷ 228,146 symbols = 0.1485</span></p>
    <ul>
      <li>A probability is a number from <strong>0</strong> (never happens) to <strong>1</strong> (always happens).</li>
      <li>If you list <em>every</em> possible outcome, their probabilities add up to exactly <strong>1</strong>. Something has to happen.</li>
    </ul>
    <p>That's all probability will mean for the rest of the course. Nothing more exotic is hiding behind the word.</p>
    <p>One more piece of vocabulary: that ⏎ bar. It counts <em>every name ending</em>, so it's exactly 32,033, one per name. That's how a generator will know when to stop: it will "draw" ⏎ sometimes, like any other symbol.</p>
  </Beat>

  <Beat gate>
    <p>Now you try it in code. A Python <strong>dictionary</strong> (<code>{'{'}'a': 0, 'b': 0{'}'}</code>) stores a number against each key, which is perfect for counts.</p>
    <Cell id="ch2-count" plain={PLAIN['ch2-count']} title="Count the letters" code={countCode} pre={VOCAB} rows={9} check={countCheck} solution={countSolution}
      hint="Inside the inner loop, ch is the current letter. counts[ch] is its running count. Add 1 to it with +=.">
      <p>One step is missing. Choose it, or write it in Python.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <h2>Spinning a probability</h2>
    <p>Now we can make a generator that respects the letter frequencies. Picture a roulette wheel where each letter's wedge is as wide as its probability. Spin it, and whatever lands under the pointer is your next letter.</p>
    <p>That's called <strong>sampling</strong>: picking a random outcome <em>in proportion to its probability</em>. Try a few spins, and spin until you land on the orange ⏎.</p>
    <Wheel />
  </Beat>

  <Beat gate>
    <Predict id="ch2-unigram" kind="choice" options={['Yes, mostly: the letters are in the right proportions', 'Better than random, but the letters won’t fit together', 'No, worse than random letters']} answer={1}>
      {#snippet q()}<p>Suppose we write names by spinning this wheel again and again, one letter per spin, until we land on ⏎. <strong>Will the results look like names?</strong></p>{/snippet}
      <p>Better than random: the lengths are sensible now and <code>a</code> and <code>e</code> show up often. But the letters don't <em>fit together</em>. See for yourself:</p>
    </Predict>
  </Beat>

  <Beat gate>
    <NameMaker model="unigram" title="Names from the letter-frequency wheel" need={2} />
    <Sceptic q="But it's using the real proportions. Why are the names still junk?">
      <p>Because every spin forgets the last one. The wheel has no memory, so after writing <code>q</code> it's just as likely to write <code>x</code> as <code>u</code>. The proportions are right <em>overall</em>, but the rules of how letters go <em>together</em> are nowhere in the wheel.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <p>That same wheel is one line of Python. It's the function <code>microgpt.py</code> uses on line 196 to pick its next letter.</p>
    <Cell id="ch2-choices" plain={PLAIN['ch2-choices']} title="One function, many spins" code={choicesCode} pre={COUNTS} rows={5}>
      <p><code>random.choices</code> takes a list of options and a list of <code>weights</code> (how wide each wedge is) and spins the wheel for you. Run it a few times.</p>
    </Cell>
  </Beat>

  <Beat>
    <Pain title="The wheel has amnesia">
      <p>What we want: after a <code>q</code>, spin a wheel where <code>u</code> is huge. After an <code>a</code>, spin a <em>different</em> wheel. So don't use one wheel. Use <strong>27 wheels, one for each symbol you might have just written</strong>. Look at the previous letter, then spin <em>that letter's</em> wheel.</p>
    </Pain>
    <p>A table of 27 wheels is just a 27×27 grid: one <strong>row</strong> per "previous symbol", one <strong>column</strong> per "next symbol", and in each cell, how often that pair happened. We call it a <strong>bigram</strong> table, because it's about pairs. Darker cell = more likely. Click rows to see them as lists.</p>
  </Beat>

  <Beat gate>
    <BigramGrid />
    <p>Take a minute with it. A few things worth hunting for:</p>
    <ul>
      <li>The <strong>q</strong> row. <code>u</code> follows it 76% of the time, which is what you'd expect. But also look at ⏎: 10% of names that reach <code>q</code> just… end there.</li>
      <li>The <strong>start</strong> row (the top one): which letters begin names? <code>a</code> (13.8%), <code>k</code>, <code>m</code>, <code>j</code>. Hardly any start with <code>x</code> or <code>q</code>.</li>
      <li>Vowels and consonants take turns. After a consonant, a vowel follows <strong>67%</strong> of the time. After a vowel, only <strong>18%</strong> (counting a, e, i, o, u as the vowels). You can see that as a checkerboard.</li>
    </ul>
  </Beat>

  <Beat>
    <TimeTravel year="1913" who="Andrey Markov">
      <p>That checkerboard you just spotted is the very first "language model". Markov, a Russian mathematician, sat down with Pushkin's novel-in-verse <em>Eugene Onegin</em> and counted, by hand, the first 20,000 letters, tallying whether each was a vowel or a consonant <em>and what tended to follow what</em>. He showed that letters aren't independent of their neighbours. A chain of things where each depends on the last has been called a <strong>Markov chain</strong> ever since.</p>
    </TimeTravel>
    <TimeTravel year="1948" who="Claude Shannon">
      <p>Shannon, the founder of information theory, generated random "English" from letter-pair and word-pair counts, and noticed that the more neighbours you condition on, the more English the output looks. You'll test that idea yourself in a few minutes. Every language model since, including the one in <code>microgpt.py</code>, is a descendant.</p>
    </TimeTravel>
  </Beat>

  <Beat gate>
    <h2>Build the table</h2>
    <p>The table isn't magic. You fill it by sliding a window of two over every name. That <code>zip(tokens, tokens[1:])</code> trick pairs each symbol with the one after it. (You'll meet exactly this idea again, in line 164 of <code>microgpt.py</code>.)</p>
    <Cell id="ch2-table" plain={PLAIN['ch2-table']} title="Count the pairs" code={tableCode} pre={VOCAB} rows={10} check={tableCheck} solution={tableSolution}
      hint="You want table[prev][nxt] to go up by one. Same move as the letter counts, but the dictionary key is now two-dimensional.">
      <p>One step is missing. Once it passes, the table you just built is the same one you were hovering over.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <h2>Now make names from it</h2>
    <p>Generating from a bigram table is a loop: start with ⏎, look at its row, spin, write the letter, then go to <em>that</em> letter's row, spin again, and so on, until the wheel lands on ⏎. Step through one by hand first.</p>
    <ChainWalk />
  </Beat>

  <Beat gate>
    <p>Now write that loop. This is the shape of generation in <em>every</em> language model, including GPT-scale ones.</p>
    <Cell id="ch2-make" plain={PLAIN['ch2-make']} title="Write the name maker" code={nameCode} pre={TABLE} rows={14} check={nameCheck} solution={nameSolution}
      hint="A while True loop never ends by itself. What Python keyword leaves a loop?">
      <p>One step is missing: what to do when the wheel lands on the stop symbol.</p>
    </Cell>
    <Aha title="You just rewrote the heart of microgpt's inference">
      <p>Compare it with lines 191–199 of <code>microgpt.py</code>: start with BOS, pick the next symbol by weighted random choice, stop and <code>break</code> when BOS comes up, otherwise append the letter. Same loop. The only difference is where the weights come from: a table of counts here, a neural network there.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <h2>Can you tell the difference?</h2>
    <p>Earlier, random letters were easy to spot. Let's see how the smarter generators hold up. Play a few rounds at <em>each</em> level and watch your record.</p>
    <FakeOrFact levels={['chance', 'unigram', 'bigram']} need={6} title="Real name, or made by a model?" />
  </Beat>

  <Beat>
    <Aha title="That feeling is the whole game">
      <p>As the model gets better, you get fooled more. When you <em>can't tell</em> which name is fake, the model has captured the real habits. The next chapter turns that feeling into a <strong>number</strong>, so we don't have to rely on squinting at names.</p>
    </Aha>
    <Sceptic q="Isn't this cheating? The 'model' is just the data, counted.">
      <p>Fair, and worth sitting with. The table <em>is</em> the model: 27×27 = <strong>729 numbers</strong>, and "training" it was nothing but counting. That's a perfectly legitimate model. The trouble is what it <em>can't</em> do, which is the next beat.</p>
      <p>By contrast, <code>microgpt.py</code> has 4,064 numbers, and it can't set them by counting. It needs a cleverer way, and inventing that way is the subject of Act II.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <Pain title="The wall">
      <p>A bigram remembers <em>one</em> letter. Real names obviously depend on more: <code>-ann-</code>, <code>-ella</code>, <code>-son</code>. So why not remember two letters, or three, or six? Just make a bigger table. Try it.</p>
    </Pain>
    <ContextWall />
  </Beat>

  <Beat>
    <p>That's the wall. A table that remembers n letters needs 27<sup>n</sup> rows, and it explodes long before the names run out of patterns to teach it. By six letters of memory, almost every row of the table is empty (only about 0.01% of them ever get an example), and the few that are filled rest on a handful of samples.</p>
    <p>Worse, a lookup table has no way to <em>generalise</em>. If it has seen <code>…anna</code> but never <code>…anne</code>, it can't guess that they're similar. Each row is an island.</p>
    <Pain title="What we actually need">
      <p>Not a table that stores answers, but a <strong>function with adjustable knobs</strong>, where nearby inputs give nearby answers, so that learning about <code>anna</code> teaches it something about <code>anne</code>. And we'd need a way to find good settings for the knobs, since we can't just count them any more. Those two ideas, <em>parameters</em> and <em>learning them</em>, are the rest of the course.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch2">
      <ul>
        <li><strong>Probability</strong> = count ÷ total, and a list of probabilities adds up to 1.</li>
        <li><strong>Sampling</strong> = spinning a wheel whose wedges are the probabilities (<code>random.choices</code>, line 196).</li>
        <li>The <strong>generation loop</strong>: start at BOS, pick, append, stop on BOS (lines 191–199).</li>
        <li>A first real model: the <strong>bigram table</strong>, 729 numbers set by counting.</li>
        <li>Why tables hit a wall, and what we'll need instead.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
