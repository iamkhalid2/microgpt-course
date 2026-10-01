<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import Predict from '../components/Predict.svelte';
  import Cell from '../components/Cell.svelte';
  import Bank from '../components/Bank.svelte';
  import AttentionX from '../widgets/AttentionX.svelte';
  import HeadSwitch from '../widgets/HeadSwitch.svelte';
  import { PLAIN } from './plain.js';

  const headsCode = `q = [round(0.1 * i, 1) for i in range(16)]      # pretend this is a query: 16 numbers
n_head = 4
head_dim = len(q) // n_head                       # 4 numbers per head

heads = []
for h in range(n_head):
    hs = h * head_dim                            # where this head's chunk starts
    chunk = None                                 # <- replace None: the numbers from position hs up to hs + head_dim
    heads.append(chunk)

x_attn = []
for part in heads:
    x_attn.extend(part)                          # glue the four chunks back into one list of 16

print(heads[1])
print(x_attn == q)
`;
  const headsCheck = `assert heads == [q[0:4], q[4:8], q[8:12], q[12:16]], "Each head should get its own 4-number chunk: q[0:4], q[4:8], q[8:12], q[12:16]. Got %r" % (heads,)
assert x_attn == q, "Gluing the chunks back together should give the original list."
print("Split into 4 heads and glued back. These are lines 124-128 and 132 of microgpt.py.")`;
  const headsSolution = `chunk = q[hs:hs + head_dim]`;
</script>

<Lesson id="ch11" num={11} title="Many Eyes" tagline="One question per position isn't enough. Ask several at once, and combine the answers.">
  <Beat>
    <Pain title="Where we left off">
      <p>Attention lets a position look back and blend what it finds. But a position gets just <strong>one</strong> blend, built from <strong>one</strong> question. Last chapter we saw the real model's four attention units behaving very differently from each other, which hints that a single question isn't enough.</p>
    </Pain>
    <p>Think about a hiring decision. One interviewer assessing everything would probably focus on whatever stood out most. A <strong>panel</strong> works better: one interviewer checks technical skill, another checks experience, another checks culture fit, another salary expectations. Each asks a <em>different</em> question of the same candidate, and then someone combines their reports.</p>
    <p>That's <strong>multi-head attention</strong>. Each "head" is one interviewer: it has its own query, its own matching, its own blend. And the model combines the four reports.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch11-cost" kind="choice" options={['Yes: four heads means four times as many dials', 'No: splitting costs no extra dials, because the 16 numbers are just shared out between the heads', 'It needs fewer dials, because each head is simpler']} answer={1}>
      {#snippet q()}<p>The model has 16 numbers per position. Suppose we run <strong>4 heads</strong>. <strong>Does that need four times as many dials</strong> as one head?</p>{/snippet}
      <p>No, and that's the elegant part. The query, key and value lists are still 16 numbers long, produced by the same three linear layers as before. We just <strong>cut each list into 4 chunks of 4 numbers</strong>, and hand chunk <em>h</em> to head <em>h</em>. Each head matches and blends using only its own 4 numbers. Same dials, four independent conversations. In <code>microgpt.py</code>: <code>head_dim = n_embd // n_head</code> = 16 ÷ 4 = 4.</p>
    </Predict>
  </Beat>

  <Beat gate>
    <h2>Cut a list into chunks</h2>
    <p>The only new programming idea is <strong>slicing</strong>: <code>q[2:5]</code> means "the part of the list q from position 2 up to, but not including, position 5". Write the loop that hands each head its own chunk, then glues the four answers back together afterwards.</p>
    <Cell id="ch11-heads" plain={PLAIN['ch11-heads']} title="Split into heads, and glue back" code={headsCode} rows={14} check={headsCheck} solution={headsSolution}
      hint="Head h owns the numbers from position hs up to hs + head_dim. In Python that slice is q[hs:hs + head_dim] (the end position is not included).">
      <p>One step is missing: the slice. These are lines 125–128 and 132 of <code>microgpt.py</code>.</p>
    </Cell>
  </Beat>

  <Beat gate>
    <h2>Four heads, one name</h2>
    <p>Here are all four heads of the real model reading the same name. Type different names and compare. Do the heads look alike? Try step 0 as well, where all four are identical blurs, and then step 3,000.</p>
    <AttentionX mode="all" initial="jonathan" />
  </Beat>

  <Beat>
    <Aha title="Four different jobs">
      <p>After training, the four heads' patterns are visibly different from each other: some mostly look back at the start of the name, others spread their attention across the letters. Each head works in its own small 4-number space, so it can learn its own habit without disturbing the others. That's the whole point of having several.</p>
    </Aha>
    <p>Finally the four answers must be combined. Each head hands back its blended list of 4 numbers; the model simply <strong>glues them side by side</strong> into one list of 16 (<code>x_attn.extend(head_out)</code>, line 132), then passes that through one more <strong>linear layer</strong>, <code>attn_wo</code> (line 133), which lets the four reports <em>mix</em>. That's the person who reads all four interview notes and forms one overall view.</p>
  </Beat>

  <Beat gate>
    <h2>Do all the heads earn their keep?</h2>
    <p>A fair question to ask of any panel. Let's find out with an experiment you can only do on a model you can open up: <strong>switch heads off, one at a time, in the real model, and measure how much worse it gets</strong>. (We compute the loss over 600 names, right here in your browser.)</p>
    <HeadSwitch />
  </Beat>

  <Beat>
    <p>What the experiment shows (on 600 names, fully trained model):</p>
    <ul>
      <li><strong>Attention helps, modestly.</strong> With all four heads off, the loss is 2.364 instead of 2.276, about <strong>4% worse</strong>.</li>
      <li><strong>Head 0 does most of the work.</strong> Switching off head 0 alone costs +0.027. Head 3 costs +0.012.</li>
      <li><strong>Heads 1 and 2 barely matter on their own.</strong> Those are the heads that stare at the start-of-name symbol. Removing either changes the loss by almost nothing (+0.002 and about zero).</li>
      <li><strong>Early in training, attention hadn't paid off yet.</strong> At step 1,000, switching <em>all</em> heads off made no difference at all (2.415 against 2.424). Attention takes a while to become useful.</li>
    </ul>
    <Sceptic q="If two of the heads barely help, why does the model have them?">
      <p>Two honest answers. First, nobody knows in advance which heads will turn out useful, so we give the model several and let training sort it out. Second, this is a <em>tiny</em> model on a <em>very short-range</em> task (names are about six letters long). Attention shows its strength on long texts, where what you need can be hundreds of words back. In big language models, heads matter far more, though even there researchers have found some can be removed with little loss. A mechanism's value depends on the problem it's given, and a name is a gentle test.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <Pain title="Attention only blends">
      <p>Look at what attention does: it <em>mixes</em> information from different positions, producing weighted averages of notes. Averages and mixes are all <em>linear</em> operations (sums and products of numbers). But predicting a name needs more than mixing: it needs <strong>decisions</strong> ("if the last letter was a vowel <em>and</em> the one before was a consonant, then…"). Mixing alone can't make such decisions. The model needs a step that <strong>thinks about</strong> what it has gathered, position by position. That's the next chapter.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch11">
      <ul>
        <li><strong>Multi-head attention</strong>: cut each query/key/value list into chunks (one per head); each head matches and blends within its own chunk, so it can ask its own question.</li>
        <li>Splitting is <strong>free</strong>: the same total number of dials, shared out between heads (<code>head_dim = n_embd // n_head</code>).</li>
        <li>Slicing a list: <code>q[hs:hs + head_dim]</code> (lines 125–128).</li>
        <li>The heads' answers are <strong>glued together</strong> (line 132) and <strong>mixed</strong> by one more linear layer, <code>attn_wo</code> (line 133).</li>
        <li>In the real model, heads specialise, but not all earn their keep: switching off all four makes it about 4% worse, mostly because of head 0.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
