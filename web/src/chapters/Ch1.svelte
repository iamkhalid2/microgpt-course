<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import Predict from '../components/Predict.svelte';
  import Cell from '../components/Cell.svelte';
  import Bank from '../components/Bank.svelte';
  import NameBrowser from '../widgets/NameBrowser.svelte';
  import TokenToy from '../widgets/TokenToy.svelte';
  import FakeOrFact from '../widgets/FakeOrFact.svelte';
  import LineStepper from '../widgets/LineStepper.svelte';
  import { PLAIN } from './plain.js';
  import { LOAD_DOCS, VOCAB } from './pysnips.js';

  const loadCode = `# The first job: read the names into a list called docs.
docs = [l.strip() for l in open('input.txt').read().strip().split('\\n') if l.strip()]
print(len(docs), 'names')
print(docs[:5])
`;

  const vocabCode = `# 1. Find every distinct character in the names, in sorted order.
#    Glue all the names into one long string, turn it into a set (that drops repeats), then sort it.
uchars = None   # <- replace None with your code

# 2. The letters take the numbers 0..25. Give "start/end of a name" the next free number.
BOS = None      # <- replace None with your code

print(uchars)
print('BOS =', BOS)
`;
  const vocabCheck = `assert uchars is not None, "uchars is still None. Replace it with your code."
assert isinstance(uchars, list), "sorted(...) returns a list. That's what we want here."
assert uchars == sorted(set(''.join(docs))), "Not quite: uchars should be every distinct character in the names, sorted."
assert BOS == len(uchars), "BOS should be the next free number after the last letter. How many letters are there?"
print("%d letters, and BOS = %d. That's the whole alphabet." % (len(uchars), BOS))`;
  const vocabSolution = `uchars = sorted(set(''.join(docs)))
BOS = len(uchars)`;

  const encodeCode = `def encode(doc):
    # Turn a name into numbers, with BOS on both ends.
    # For example: encode('emma') should give [26, 4, 12, 12, 0, 26]
    return None   # <- replace None

print(encode('emma'))
`;
  const encodeCheck = `assert encode('emma') == [26, 4, 12, 12, 0, 26], "encode('emma') should be [26, 4, 12, 12, 0, 26], but I got %r" % (encode('emma'),)
assert encode('a') == [26, 0, 26], "a one-letter name should still get BOS on both sides"
print("Every name is now a list of numbers. That is line 157 of microgpt.py, nearly word for word.")`;
  const encodeSolution = `def encode(doc):
    return [BOS] + [uchars.index(ch) for ch in doc] + [BOS]`;

  const babbleCode = `import random

def babble():
    name = ''
    while True:
        t = random.randrange(len(uchars) + 1)   # any of the 27 numbers, all equally likely
        if t == BOS:
            break                               # the dice said "stop"
        name += None                            # <- replace None: add the LETTER for number t
    return name

for _ in range(8):
    print(babble())

print('average length:', sum(len(babble()) for _ in range(2000)) / 2000)
`;
  const babbleCheck = `out = [babble() for _ in range(300)]
assert all(isinstance(n, str) for n in out), "babble() should return a string."
assert all(n.isalpha() or n == '' for n in out), "Names should contain letters only. Did you add the letter uchars[t], not the number t?"
avg = sum(map(len, out)) / len(out)
assert 12 < avg < 45, "The average length looks off (%.1f). Is the loop stopping when it draws BOS?" % avg
print("babble() works. Average length: %.0f letters." % avg)`;
  const babbleSolution = `name += uchars[t]`;

  const demoProgram = [
    { code: "names = ['emma', 'ava']" },
    { code: 'total = 0' },
    { code: 'for name in names:' },
    { code: 'total = total + len(name)', depth: 1 },
    { code: 'print(total)' },
  ];
  const demoSteps = [
    { line: 0, en: 'Make a box labelled names and put a list in it. Square brackets mean a list: an ordered row of items. This one holds two names.', vars: { names: "['emma', 'ava']" }, changed: 'names' },
    { line: 1, en: 'Make a box labelled total and put 0 in it. The = sign does not mean "is equal to". It means "put the thing on the right into the box on the left".', vars: { names: "['emma', 'ava']", total: '0' }, changed: 'total' },
    { line: 2, en: 'A loop: "for each item in the list, run the indented lines below". First item: take out "emma" and call it name.', vars: { names: "['emma', 'ava']", total: '0', name: "'emma'" }, changed: 'name' },
    { line: 3, en: 'len(name) counts the letters: 4. Add that to what is in total (0) and put the answer back in the total box.', vars: { names: "['emma', 'ava']", total: '4', name: "'emma'" }, changed: 'total' },
    { line: 2, en: 'Back to the loop. Next item in the list: "ava".', vars: { names: "['emma', 'ava']", total: '4', name: "'ava'" }, changed: 'name' },
    { line: 3, en: 'len("ava") is 3. The total box held 4, so it now holds 7.', vars: { names: "['emma', 'ava']", total: '7', name: "'ava'" }, changed: 'total' },
    { line: 4, en: 'No items left, so the loop is over and we move on. Show what is in the total box.', vars: { names: "['emma', 'ava']", total: '7', name: "'ava'" }, out: '7' },
  ];

  const seedCode = `import random

random.seed(42)
print([random.randrange(27) for _ in range(6)])

random.seed(42)                                   # rewind the dice...
print([random.randrange(27) for _ in range(6)])   # ...and they fall the same way again
`;
</script>

<Lesson id="ch1" num={1} title="Babble" tagline="Computers can't read letters, and random letters aren't names.">
  <Beat gate>
    <Pain title="The situation">
      <p>We want a machine that invents names. Before we're clever about it, let's be honest about what we have: one text file.</p>
    </Pain>
    <p>This is <code>input.txt</code>: <strong>32,033 real names</strong>, one per line. (It's from Karpathy's <em>makemore</em> project.) That is the model's entire education. No dictionary, no grammar book, no internet.</p>
    <NameBrowser />
  </Beat>

  <Beat gate>
    <Predict id="ch1-longest" kind="number" answer={15} tolerance={2} unit="letters">
      {#snippet q()}<p>Names vary in length. <strong>How many letters are in the longest name in the file?</strong></p>{/snippet}
      <p>15. And the average is only about 6.1. Remember that: a name generator has to get the <em>letters</em> right <em>and</em> the <em>length</em> right, which sounds like two problems and will turn out to be one.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <p>Here's the thing about you: you've never studied names, and yet you have a real feel for what a name looks like. Test that. Below, one name is real and three are random letters.</p>
    <FakeOrFact levels={['chance']} need={3} title="Real or random letters?" />
  </Beat>

  <Beat>
    <p>Easy, right? Random letters are obviously not names. Hold that feeling of "obviously". The whole course is about giving a machine the same instinct you just used without thinking.</p>
    <p>But notice what you can't do yet: explain <em>why</em> <code>xqzvb</code> looks wrong. To make a machine have that instinct, we first have to hand it the data in a form it can work with.</p>
  </Beat>

  <Beat gate>
    <Pain title="The pain">
      <p>A computer doesn't read letters. It does arithmetic on <em>numbers</em>. Right now "emma" means nothing to it.</p>
    </Pain>
    <p>The fix is almost insultingly simple: give every distinct character a number. List the characters that appear in the data, sort them, and number them <code>0, 1, 2, …</code>. Try it:</p>
    <TokenToy />
    <Sceptic q="Why is 'a' equal to 0? Isn't there already a standard numbering, like 97?">
      <p>There is (ASCII / Unicode), and we could use it. But we'd waste most of the numbers, since names only use 26 letters, and the model would have 97-ish slots to handle for no benefit. Numbering just the characters <em>in our data</em> keeps it tight.</p>
      <p>The key thing to see: these numbers are <strong>labels, not quantities</strong>. <code>z</code> (25) isn't "more" than <code>a</code> (0). We could shuffle the numbering and nothing would change. Each one is just a name tag for a character.</p>
    </Sceptic>
    <Sceptic q="What's the ⏎ thing? That's not a letter.">
      <p>Good catch, and it's the reason the alphabet has 27 entries, not 26. A model reading names one letter at a time needs to know <em>where a name starts</em> and <em>where it ends</em>. So we invent one extra symbol that's not a letter, called <strong>BOS</strong> ("beginning of sequence"), and wrap every name in it: <code>⏎ e m m a ⏎</code>.</p>
      <p>When the model later writes a name, it begins with BOS, and when it <em>chooses</em> BOS as its next symbol, that means "I'm done." One symbol, two jobs.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <h2>How to read the code on this page</h2>
    <p>From here on we'll use real Python. You don't need to know it already, and you don't need to memorise it. Reading code is a lot like reading a recipe: it's a list of steps, done in order. Step through this tiny program and watch what each line does to the "boxes" (variables) that hold its information.</p>
    <LineStepper program={demoProgram} steps={demoSteps} />
    <div class="gloss wide">
      <div><code>=</code> put the right side into the box on the left</div>
      <div><code>[ … ]</code> a list: an ordered row of things</div>
      <div><code>for x in things:</code> repeat the indented lines once per item</div>
      <div><code>def name(…):</code> define a named recipe to reuse later</div>
      <div><code>+=</code> add to what's already in the box</div>
      <div><code>#</code> a note for humans; the computer ignores it</div>
    </div>
    <p class="muted">The indentation (spaces at the start of a line) matters: it means "this line belongs inside the thing above it".</p>
  </Beat>

  <Beat gate>
    <h2>Now you do it, for real</h2>
    <p>Everything below runs real Python in your browser, reading the same <code>input.txt</code> that <code>microgpt.py</code> reads. Each exercise has <strong>two routes</strong>, and they count the same:</p>
    <ul>
      <li><strong>Plain English:</strong> you read a recipe, and wherever a step is missing you pick the right one. The real Python line sits beside every step, so you absorb it as you go.</li>
      <li><strong>Python:</strong> you write the missing line in a real editor. Switch any time using the toggle on the exercise, and open "Read it in plain English" if a line confuses you.</li>
    </ul>
    <p>Either way, the same program runs and the same check decides whether it works. First, line 19 of <code>microgpt.py</code>. Read it aloud like a sentence: <em>"docs is, for every line in the file, that line with stray spaces removed, but only if the line isn't empty."</em></p>
    <Cell id="ch1-load" plain={PLAIN['ch1-load']} title="Run it" code={loadCode} rows={4}>
      <p>Press <strong>Run</strong>. Curious? In the Python route, change the <code>5</code> to something else and run it again.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <p>Next, the alphabet. In <code>microgpt.py</code> this takes two short lines.</p>
    <Cell id="ch1-vocab" plain={PLAIN['ch1-vocab']} title="Build the alphabet" code={vocabCode} pre={LOAD_DOCS} rows={8}
      check={vocabCheck} solution={vocabSolution}
      hint="''.join(docs) glues every name into one giant string. set(...) of that keeps each character once. sorted(...) puts them in order. And BOS is just len(uchars): the number right after the last letter.">
      <p>Two steps are missing. Choose them, or write them in Python.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <p>One more, and then the data is ready. We need a recipe that turns a name into its list of numbers, wrapped in the start/end symbol on both sides.</p>
    <Cell id="ch1-encode" plain={PLAIN['ch1-encode']} title="Names to numbers" code={encodeCode} pre={VOCAB} rows={6}
      check={encodeCheck} solution={encodeSolution}
      hint="uchars.index('e') gives 4: the position of 'e' in the list, which is its number. Make the middle with a list comprehension over the letters, then put [BOS] on each side with +.">
      <p>One step is missing: what <code>encode</code> should hand back.</p>
    </Cell>
    <Aha title="You just wrote real code from microgpt.py">
      <p>That's line 157, nearly word for word. The model never sees letters. It sees lists like <code>[26, 4, 12, 12, 0, 26]</code>.</p>
    </Aha>
  </Beat>

  <Beat gate>
    <h2>The first generator (it's terrible)</h2>
    <p>Now let's build the simplest possible name-maker. Start from BOS. Pick one of the 27 symbols at random, all equally likely. If it's BOS, stop. Otherwise write the letter down and pick again.</p>
    <Predict id="ch1-avglen" kind="number" answer={26} tolerance={5} unit="letters">
      {#snippet q()}<p>Each pick has a 1-in-27 chance of being BOS (stop). <strong>On average, how many letters will this generator write before it stops?</strong></p>{/snippet}
      <p>26. It's like rolling a 27-sided die until one specific face comes up: you wait about 27 rolls, and 26 of them write a letter. Real names average about 6. So this generator will ramble.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <Cell id="ch1-babble" plain={PLAIN['ch1-babble']} title="Write the babbler" code={babbleCode} pre={VOCAB} rows={12}
      check={babbleCheck} solution={babbleSolution}
      hint="t is a number. uchars[t] is the letter with that number. Add that to name.">
      <p>One step is missing. Once it works, run it a few times and read what the babbler says.</p>
    </Cell>
  </Beat>

  <Beat>
    <Aha title="Two things are wrong, not one">
      <p>The <strong>letters</strong> are wrong (<code>xqzvb</code> isn't a name) <em>and</em> the <strong>length</strong> is wrong (26 instead of 6). We'll soon see that these aren't separate problems. Both come from the same flaw: the generator <em>ignores the data</em>. It treats every symbol as equally likely, every time. Real names clearly don't.</p>
    </Aha>
    <h2>Two lines you've now earned</h2>
    <p>Two tiny things in the first lines of <code>microgpt.py</code> that you now have the context for:</p>
    <Cell id="ch1-seed" plain={PLAIN['ch1-seed']} title="Order among chaos" code={seedCode} rows={6}>
      <p>Line 12 is <code>random.seed(42)</code>. Run this and compare the two rows. Randomness in a computer isn't really random; it's a long, scrambled list of numbers. A <strong>seed</strong> picks where in the list to start. Same seed, same "random" numbers, which means when something goes wrong you can reproduce it exactly.</p>
    </Cell>
    <Sceptic q="Why does microgpt.py shuffle the names right after loading them?">
      <p>Because the file is <em>not</em> in random order: it starts <code>emma, olivia, ava, isabella…</code> and ends <code>zyrie, zyron, zzyzx</code>. Later, the model will study the names one after another, nudging its numbers a little after each. If it saw a long run of similar names, it would get yanked toward whatever those looked like. Shuffling means every stretch of the training looks like the whole dataset.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <Pain title="Where this leaves us">
      <p>Random symbols can't make names. The data has <em>habits</em> (<code>q</code> is nearly always followed by <code>u</code>; names rarely start with <code>x</code>; vowels and consonants alternate), and a generator that ignores them is hopeless. Next chapter we teach the generator to count those habits, which, it turns out, is an idea from over a century ago.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch1">
      <ul>
        <li>The dataset: a list of 32,033 names, shuffled (lines 14–21).</li>
        <li>The tokenizer: sorted unique characters become numbers, plus one extra BOS symbol (lines 23–27).</li>
        <li>Wrapping each name in BOS on both ends (line 157).</li>
        <li>What <code>random.seed</code> is for (lines 9–12).</li>
        <li>A baseline to beat: a generator that knows nothing.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
