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
  import LookupLab from '../widgets/LookupLab.svelte';
  import ScaleSoftmax from '../widgets/ScaleSoftmax.svelte';
  import AttentionX from '../widgets/AttentionX.svelte';
  import { SOFTMAX } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const attendCode = `def attend(q, keys, values):
    d = len(q)
    # 1. match the query against every earlier position's key (dot product), scaled by the square root of d
    scores = [sum(qi * ki for qi, ki in zip(q, k)) / d ** 0.5 for k in keys]
    # 2. turn the scores into weights that add up to 1
    weights = softmax(scores)
    # 3. blend the values using those weights
    return [None for j in range(len(values[0]))]      # <- replace None: sum over positions of (weight x value[j])

print(attend([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [0.0, 10.0]]))
`;
  const attendCheck = `import math
out = attend([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [0.0, 10.0]])
e = math.exp(1 / math.sqrt(2)); w0 = e / (e + 1); w1 = 1 / (e + 1)
assert len(out) == 2, "The answer should be a list of 2 numbers (one per place in a value)."
assert abs(out[0] - 10 * w0) < 1e-6 and abs(out[1] - 10 * w1) < 1e-6, "Expected about [%.3f, %.3f] (the values blended by weights %.3f and %.3f), but got %r" % (10 * w0, 10 * w1, w0, w1, out)
same = attend([1.0, 2.0], [[0.5, 0.5]], [[3.0, 4.0]])
assert all(abs(a - b) < 1e-9 for a, b in zip(same, [3.0, 4.0])), "With only one earlier position it gets all the weight, so the answer is just its value."
print("attend works. These are lines 129-131 of microgpt.py: score, softmax, blend.")`;
  const attendSolution = `sum(w * v[j] for w, v in zip(weights, values))`;
</script>

<Lesson id="ch10" num={10} title="Looking Back" tagline="Which earlier letters matter right now? Let each position ask a question, and blend the answers.">
  <Beat>
    <Pain title="The thing we've been dodging">
      <p>Our model still has the goldfish memory of the bigram: when predicting the next letter it sees <em>only</em> the letter in front of it. But names have structure that spreads over several letters. After <code>ann</code> a name likes to continue <code>a</code> or <code>e</code>; after <code>ka</code> it might continue <code>y</code> or <code>r</code>. To use that, the model must be able to <strong>look back at earlier letters</strong>.</p>
    </Pain>
    <p>The obvious fix is to add up (or average) the coordinates of all the earlier letters. Try the thought experiment: after reading <code>m-a-r-i-a</code>, the model averages five letters' worth of numbers. But in <code>maria</code> the "m" at the start matters much more for what comes next than the "r" in the middle, and which letter matters <em>depends on the situation</em>. A plain average treats every earlier letter as equally important, always. It blurs the very information we wanted.</p>
    <Aha title="What we want">
      <p>Each position should be able to <strong>choose</strong> which earlier positions to listen to, and how much, depending on what it's trying to work out. A <em>selective</em> average rather than a plain one.</p>
    </Aha>
  </Beat>

  <Beat>
    <h2>The meeting analogy</h2>
    <p>Picture a meeting. You're about to make a decision and you have a <strong>question</strong> in mind. Everyone who has spoken so far wears a <strong>badge</strong> saying what they're an expert in, and has a <strong>note</strong> with what they actually know. You look around the table and compare your question with each badge: those whose badge matches your question get your attention, in proportion to how well it matches. Your answer is a <strong>blend of their notes</strong>, weighted by that attention.</p>
    <p>Each earlier position supplies two things, and the current position supplies one:</p>
    <ul>
      <li><strong>Query</strong>: what this position is looking for (your question).</li>
      <li><strong>Key</strong>: what each earlier position advertises (its badge).</li>
      <li><strong>Value</strong>: what each earlier position contributes if listened to (its note).</li>
    </ul>
    <p>All three are produced from a position's 16-number list by a <strong>linear layer</strong> (Chapter 9): <code>q = linear(x, attn_wq)</code>, <code>k = linear(x, attn_wk)</code>, <code>v = linear(x, attn_wv)</code>. Three matrices of dials, <em>learned</em>. Nobody tells the model what to ask or advertise. Training works that out.</p>
  </Beat>

  <Beat gate>
    <p>And the <strong>match</strong> between a question and a badge? Chapter 9's <strong>dot product</strong>: high when they point the same way. Then <strong>softmax</strong> (Chapter 8) turns the matches into attention weights that add to 1, and the answer is the weighted blend of the values. Try it by hand with four memos and a question you control.</p>
    <LookupLab />
  </Beat>

  <Beat>
    <Aha title="That is attention">
      <p>Three steps: <strong>(1) match</strong> the query against every key with a dot product, <strong>(2) softmax</strong> the matches into weights, <strong>(3) blend</strong> the values using those weights. That's the whole mechanism behind every modern language model. The rest of the transformer is plumbing around it.</p>
    </Aha>
    <p>One fix to step 1: before softmax, the real code divides each match by the <strong>square root of the list length</strong> (<code>/ head_dim**0.5</code>, line 129). Why? Let's break it and see.</p>
  </Beat>

  <Beat gate>
    <ScaleSoftmax />
  </Beat>

  <Beat>
    <p>With longer lists, raw dot products get bigger (they grow like √d), and big scores make softmax <strong>overconfident</strong>: nearly all the weight on one position, which is almost a hard choice. Overconfident softmax also gives tiny slopes for everyone else, so the model <em>stops learning</em> from the other positions. Dividing by √d keeps the scores in the same friendly range however long the lists are.</p>
    <Sceptic q="What does 'earlier positions' include? Can a position look at itself, or at later letters?">
      <p>It looks at <strong>itself and everything before it</strong>, never at later letters. That's not an extra rule in the code, just a consequence of how it runs: at each position the model has only seen the letters up to that point, and only those have filed their keys and values (<code>keys[li].append(k)</code>, <code>values[li].append(v)</code>, lines 121–122). Think of the keys and values as memos in a filing cabinet that grows as the name is read. This is essential: if the model could peek at the answer (the next letter) during training, it would learn nothing useful.</p>
    </Sceptic>
  </Beat>

  <Beat gate>
    <Cell id="ch10-attend" plain={PLAIN['ch10-attend']} title="Write attention" code={attendCode} pre={SOFTMAX} rows={11} check={attendCheck} solution={attendSolution}
      hint="Each position has a weight and a value (a list). For place j of the answer, add up weight × value[j] over every position. zip(weights, values) pairs each weight with its value.">
      <p>One step is missing: blending the values. The first two steps are already there. Together they are lines 129–131 of <code>microgpt.py</code>.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <h2>The real thing</h2>
    <p>Now let's look at attention in the real 200-line model. Type a name. The picture shows, for each letter being read (rows), how its attention is shared among the earlier positions (columns), as percentages. Start with <strong>step 0</strong>, the untrained model, then compare with <strong>step 3,000</strong>. (The real model has four <em>heads</em>, or "attention units", working in parallel. For now, pick one and look at it. Next chapter, we'll see why there are four.)</p>
    <AttentionX mode="single" />
  </Beat>

  <Beat>
    <Aha title="Learning changed what it listens to">
      <p>At step 0 every position gets <strong>equal</strong> attention (a plain average: with six positions to look at, each gets 17%). After training, the heads have specialised, and they do <em>different</em> things. Averaged over 2,000 real names (from the third letter onward), <strong>head 1</strong> puts about 73% of its attention on the very first symbol (the start-of-name ⏎), <strong>head 2</strong> about 67%, while <strong>head 0</strong> (33%) and <strong>head 3</strong> (20%) spread their attention across the letters. Nobody designed that. Training found that it helps.</p>
    </Aha>
    <Sceptic q="Why would a head look at the start-of-name symbol? That isn't a letter.">
      <p>It's a good question and we don't have a one-line answer, but here's one plausible reading. The start symbol is always in the same place and always says the same thing, so it can serve as a <strong>fixed reference point</strong>, like a "home" position that tells the model how far into the name it is and gives it a steady signal to compare against. The same behaviour shows up in much larger models, where people call such positions "attention sinks". We should be honest that interpreting what a trained head is doing is an open research problem. What you can trust is the picture itself, which is exactly what the real model computes.</p>
    </Sceptic>
    <TimeTravel year="2014 to 2017" who="Bahdanau, then Vaswani and colleagues">
      <p>Attention was introduced by Dzmitry Bahdanau and colleagues in 2014 to help translation models focus on the right input words. In 2017 a team at Google showed, in a paper called <em>"Attention Is All You Need"</em>, that a model built <em>mainly</em> out of attention, with no other machinery for handling sequences, worked better and trained faster. That design is the <strong>transformer</strong>, the "T" in GPT (Generative Pre-trained Transformer).</p>
    </TimeTravel>
  </Beat>

  <Beat>
    <Pain title="One question per position isn't enough">
      <p>A single attention unit asks a <em>single</em> question per position and gets a single blended answer. But in the real model, different heads were doing noticeably different jobs. Is one question really enough to predict a letter? Probably not: you might want to ask <em>"what was the first letter?"</em> and <em>"what was the last vowel?"</em> and <em>"how long has the name been?"</em> all at once. That's the next chapter.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch10">
      <ul>
        <li><strong>Attention</strong> = a selective average: each position asks a <strong>query</strong>, every earlier position offers a <strong>key</strong> (its badge) and a <strong>value</strong> (its note).</li>
        <li>Three steps: <strong>match</strong> (dot product of query and key) → <strong>softmax</strong> into weights → <strong>blend</strong> the values.</li>
        <li>Dividing the match by <strong>√d</strong> keeps softmax from becoming overconfident.</li>
        <li>A position sees <strong>itself and the past</strong>, never the future. Keys and values are filed in a growing cabinet (lines 121–122).</li>
        <li>In the real model, untrained attention is a flat average; trained attention has <strong>specialised</strong>.</li>
        <li>The code: <code>q, k, v = linear(...)</code> (lines 118–120), keys and values filed (121–122), and score → softmax → blend (129–131).</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
