<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import Predict from '../components/Predict.svelte';
  import Cell from '../components/Cell.svelte';
  import Bank from '../components/Bank.svelte';
  import DotLab from '../widgets/DotLab.svelte';
  import LinearLab from '../widgets/LinearLab.svelte';
  import RealEmbedding from '../widgets/RealEmbedding.svelte';
  import { PLAIN } from './plain.js';

  const dotCode = `def dot(x, w):
    # multiply matching numbers together, then add up the results
    return None     # <- replace None

def linear(x, w):
    # w is a list of rows. One dot product per row gives one score per row.
    return [dot(x, row) for row in w]

print(dot([1, 2, 3], [4, 5, 6]))                          # expect 32
print(linear([1.0, 2.0], [[3.0, 4.0], [0.5, -1.0]]))      # expect [11.0, -1.5]
`;
  const dotCheck = `assert dot([1, 2, 3], [4, 5, 6]) == 32, "dot([1, 2, 3], [4, 5, 6]) should be 4 + 10 + 18 = 32, but I got %r" % (dot([1, 2, 3], [4, 5, 6]),)
assert dot([1, 0], [0, 1]) == 0, "Two lists at right angles should score 0."
assert linear([1.0, 2.0], [[3.0, 4.0], [0.5, -1.0]]) == [11.0, -1.5], "linear gave %r" % (linear([1.0, 2.0], [[3.0, 4.0], [0.5, -1.0]]),)
print("dot and linear work. 'linear' is exactly the function on line 94 of microgpt.py.")`;
  const dotSolution = `return sum(xi * wi for xi, wi in zip(x, w))`;

  const embedCode = `import json

W = json.load(open('weights.json'))['checkpoints']['3000']     # the real trained dials
wte = W['wte']      # token table: 27 rows (one per symbol), 16 numbers each
wpe = W['wpe']      # position table: 8 rows (one per position), 16 numbers each

token_id = 0        # the letter a
pos_id = 2          # the third spot in a name

tok_emb = wte[token_id]
pos_emb = wpe[pos_id]
x = [None for t, p in zip(tok_emb, pos_emb)]      # <- replace None: combine the two rows, number by number

print([round(v, 3) for v in x[:5]])
`;
  const embedCheck = `want = [t + p for t, p in zip(wte[0], wpe[2])]
assert len(x) == 16, "x should have 16 numbers (one per place), but has %d" % len(x)
assert all(abs(a - b) < 1e-12 for a, b in zip(x, want)), "x should be the token row PLUS the position row, number by number."
print("That list of 16 numbers is exactly what the real model sees for 'a' in third position (line 111 of microgpt.py).")`;
  const embedSolution = `t + p`;

  const matrixCode = `import random

def matrix(nout, nin, std=0.02):
    # a table with nout rows and nin columns, filled with small random numbers
    return [[None for _ in range(nin)] for _ in range(nout)]     # <- replace None: one random number with spread std

m = matrix(3, 4)
print(len(m), 'rows of', len(m[0]), 'numbers')
print([round(v, 3) for v in m[0]])
`;
  const matrixCheck = `assert len(m) == 3 and all(len(r) == 4 for r in m), "matrix(3, 4) should have 3 rows of 4 numbers."
big = [v for row in matrix(60, 60) for v in row]
mean = sum(big) / len(big)
sd = (sum((v - mean) ** 2 for v in big) / len(big)) ** 0.5
assert all(isinstance(v, float) for v in big), "Every cell should be a random number."
assert abs(mean) < 0.003, "The numbers should centre on 0 (mean was %.4f)." % mean
assert 0.017 < sd < 0.023, "The spread should be about 0.02 (it was %.4f). Use random.gauss(0, std)." % sd
print("Random starting dials with spread %.3f. This is line 80 of microgpt.py." % sd)`;
  const matrixSolution = `random.gauss(0, std)`;
</script>

<Lesson id="ch9" num={9} title="Letters as Points" tagline="Give every letter coordinates, so the model can notice that some letters behave alike.">
  <Beat>
    <Pain title="Where we left off">
      <p>We can train any model that turns inputs into a score per next-letter. But our only model, the bigram table, has a hidden flaw that matters now that it's trainable: <strong>every row is an island</strong>. The row for "a" and the row for "e" are 27 separate sets of dials. Whatever the model learns after "a", it learns nothing about "e", even though a human can see the two letters behave alike (both vowels, both followed by consonants).</p>
    </Pain>
    <p>Business has a name for this problem. A database that identifies customers by <strong>ID number</strong> can't tell that customer 5,812 and customer 5,813 are alike. Describe each customer by <strong>features</strong> (age, spend, region) and similar customers become <em>nearby points</em>, so what you learn about one transfers to the other.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch9-rep" kind="choice" options={['Keep a separate row of dials for each letter, as the table does', 'Describe each letter by a short list of numbers (coordinates) that the model can learn', 'Use the letter\'s position in the alphabet as its number']} answer={1}>
      {#snippet q()}<p>Which way of feeding letters to a model lets it <strong>learn that "a" and "e" are alike</strong>?</p>{/snippet}
      <p>Coordinates. If "a" and "e" end up with similar coordinates, then anything the model works out from one's numbers automatically applies to the other. Using alphabet position (a=1, b=2…) is a trap: it says "d" is close to "e", which isn't true for names. Better to let the model <em>learn</em> the coordinates.</p>
    </Predict>
  </Beat>

  <Beat>
    <Aha title="An embedding is a learned address">
      <p>Give each of the 27 symbols a row of <strong>16 numbers</strong>: its coordinates in a 16-dimensional space. These aren't chosen by us. They're <strong>dials</strong>, 27 × 16 = 432 of them, trained like all the others. Putting a letter's row of numbers where the letter used to go is called an <strong>embedding</strong>. In <code>microgpt.py</code> the table is <code>wte</code> ("word/token embeddings").</p>
    </Aha>
    <p>To work with coordinates we need two small tools. The first measures how much two lists of numbers <strong>agree</strong>.</p>
  </Beat>

  <Beat gate>
    <h2>The dot product</h2>
    <p>Take two lists of numbers. Multiply the numbers that sit in the same place, then add up the results. That's the <strong>dot product</strong>. It's a score for how much the two lists point the same way. Drag the arrow tips and watch it.</p>
    <DotLab />
    <Aha title="Why it works">
      <p>If both lists are big in the same places, every product is big and positive, so the total is large. If one is big where the other is negative, those products are negative and cancel it. If they are unrelated, positives and negatives roughly cancel to zero. It's like two customers who like the same products: lots of overlap, high agreement.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <h2>Scorecards</h2>
    <p>The second tool: a <strong>linear layer</strong>. Suppose we want <em>several</em> scores from the same list of numbers, one per option. Give each option its own row of weights and take the dot product of the input with each row. A table of such rows is called a <strong>matrix</strong>, and "a matrix times a list" is nothing more than this: one dot product per row. Think of a hiring scorecard with one column of weights per role.</p>
    <LinearLab />
  </Beat>

  <Beat gate>
    <Cell id="ch9-dot" plain={PLAIN['ch9-dot']} title="Write dot and linear" code={dotCode} rows={11} check={dotCheck} solution={dotSolution}
      hint="Pair up the numbers with zip(x, w), multiply each pair, and add everything up with sum(...).">
      <p>One step is missing: the dot product itself. <code>linear</code> then simply uses it once per row. This is line 94 of <code>microgpt.py</code>: <code>sum(wi * xi for wi, xi in zip(wo, x))</code>.</p>
    </Cell>
    <p>Every time the real model "thinks", it's doing a lot of these: a linear layer turns a 16-number list into another list of numbers, scoring it against rows of dials. It's the model's basic move. The final step of the model is also one: a matrix called <code>lm_head</code> turns the final 16 numbers back into <strong>27 scores</strong>, one per possible next symbol, which softmax (last chapter) turns into probabilities.</p>
  </Beat>

  <Beat gate>
    <h2>What does the real model's map of letters look like?</h2>
    <p>Here is the real 432-dial embedding table from the 200-line model, after training. We can't draw 16 dimensions, so the picture squashes them down to 2 while keeping as much of the spread as possible (a standard trick). Drag the slider through training, from <strong>step 0</strong> to <strong>step 3,000</strong>, and watch the letters organise themselves.</p>
    <RealEmbedding />
  </Beat>

  <Beat>
    <Aha title="The model invented 'vowel'">
      <p>At step 0 the table is random noise: no letter is closer to any other (the average distance is the same, 0.11, whether you compare two vowels or a vowel and a consonant). By step 3,000 the vowels have <strong>clustered</strong> on one side: two vowels are on average 1.73 apart, but a vowel and a consonant are 2.35 apart. Nobody told the model which letters are vowels. It worked out that grouping them lowers the loss.</p>
    </Aha>
    <Sceptic q="Are those 16 numbers 'meaning' something like 'vowel-ness'?">
      <p>Not in any single number, no. The meaning is spread across all 16, and which direction means what is something the model invents for itself. That's why we squash the picture down to 2 dimensions to look at it. The honest summary is: <strong>distance in this space measures "behaves alike when predicting the next letter"</strong>, and training pulls such letters together.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <h2>Where in the name are we?</h2>
    <p>One more thing a model needs: <strong>position</strong>. The letter "a" at the start of a name behaves differently from "a" at the end. So there's a second table, <code>wpe</code>, with a row of 16 numbers for each position (up to 8, <code>block_size</code>). The model <strong>adds</strong> the two rows together to make one list that says <em>both which letter and where</em>. Try it with the real dials, in the real model's own numbers:</p>
    <Cell id="ch9-embed" plain={PLAIN['ch9-embed']} title="Letter plus position" code={embedCode} rows={14} check={embedCheck} solution={embedSolution}
      hint="You want one list of 16 numbers. Take the first number of each row and add them, then the second of each, and so on. zip pairs them up.">
      <p>One step is missing: combining the two rows. Lines 109–111 of <code>microgpt.py</code> do exactly this.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <h2>Where do the starting dials come from?</h2>
    <p>Training needs starting values for all these tables. Two rules: they must be <strong>small</strong> (so the first guesses aren't wild) and they must be <strong>random</strong>. The second rule is surprising, so let's look at it.</p>
    <Sceptic q="Why random? Why not just start every dial at zero?">
      <p>If every dial in a layer starts with the same value, every dial in it receives the <em>same slope</em>, moves by the <em>same amount</em>, and stays identical forever. A table of 27 "different" letters would remain 27 identical copies. Small random numbers break the tie, so dials can drift apart and specialise. The bell-curve ("gaussian") shape just means most starting values are close to 0 and a few are a bit further out.</p>
    </Sceptic>
    <Cell id="ch9-matrix" plain={PLAIN['ch9-matrix']} title="A table of random dials" code={matrixCode} rows={9} check={matrixCheck} solution={matrixSolution}
      hint="Python's random.gauss(mean, spread) gives one random number from a bell curve. You want mean 0, and spread std.">
      <p>One step is missing: what goes in each cell. This is line 80 of <code>microgpt.py</code> (there, each cell is also wrapped in a <code>Value</code> from Chapter 7 so the engine can track it).</p>
    </Cell>
  </Beat>

  <Beat>
    <Pain title="What we still can't do">
      <p>We now feed the model "this letter, at this position" as a rich list of 16 numbers, and it can turn lists into scores with linear layers. But the model at this point still looks only at <strong>one</strong> letter at a time: the one it's just been handed. A bigram with better bookkeeping. To use the <em>earlier</em> letters of the name, it needs a way to <strong>look back</strong> and pick out which earlier letters matter for the prediction it's making right now. That mechanism has a famous name, and it's the next chapter.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch9">
      <ul>
        <li>An <strong>embedding</strong>: each symbol gets a learned list of 16 numbers (its coordinates), so similar symbols can become nearby points and share what the model learns.</li>
        <li>The <strong>dot product</strong>: multiply matching numbers and add. A score for how much two lists agree.</li>
        <li>A <strong>linear layer</strong> (a matrix times a list): one dot product per row of dials, one score per row. It's <code>linear</code> on line 94.</li>
        <li>A <strong>position embedding</strong> added to the letter's row, so the model knows where it is (lines 109–111).</li>
        <li>Starting dials are <strong>small and random</strong> (line 80), because identical dials would stay identical.</li>
        <li>The real model's vowels end up clustered together, a distinction nobody gave it.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
