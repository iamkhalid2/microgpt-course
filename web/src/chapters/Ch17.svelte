<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import Predict from '../components/Predict.svelte';
  import Cell from '../components/Cell.svelte';
  import Bank from '../components/Bank.svelte';
  import TempLab from '../widgets/TempLab.svelte';
  import { GPTFN } from './pysnips.js';
  import { PLAIN } from './plain.js';

  const genCode = `import random

def generate(temperature=0.5):
    keys, values = [], []                 # an empty filing cabinet
    token_id = BOS                        # start from the start symbol
    name = ''
    for pos_id in range(block_size):      # at most 8 symbols
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([None for l in logits])       # <- replace None: how should the temperature change each score?
        token_id = random.choices(range(len(probs)), weights=probs)[0]      # spin the weighted wheel
        if token_id == BOS:               # the wheel landed on "the name ends here"
            break
        name += uchars[token_id]
    return name

random.seed(1)
print([generate() for _ in range(8)])
`;
  const genCheck = `random.seed(5); cold = [generate(0.05) for _ in range(6)]
random.seed(5); hot = [generate(2.0) for _ in range(12)]
assert all(isinstance(n, str) for n in cold + hot), "generate should return a string."
assert len(set(cold)) <= 3, "At a very low temperature the model should keep writing the same few names, but I got %r. Is the temperature DIVIDING the scores?" % (cold,)
assert len(set(hot)) >= 10, "At a high temperature the names should almost all be different, but I got %r." % (hot,)
print("generate works. Cold: %s ... Hot: %s ..." % (cold[:3], hot[:3]))`;
  const genSolution = `l / temperature`;
</script>

<Lesson id="ch17" num={17} title="Speaking" tagline="A trained model holds probabilities. Turning them into names takes one more idea: how adventurous to be.">
  <Beat>
    <Pain title="Where we left off">
      <p>The trained model can say, for any name-so-far, how likely each next symbol is. That's a table of probabilities, not a name. To <em>write</em>, we have to <strong>choose</strong> a symbol, add it to the name, and ask again. And the choice is where all the personality comes from.</p>
    </Pain>
    <p>You've met this loop already. In Chapter 2 we spun a weighted wheel again and again, starting from ⏎ and stopping when the wheel said ⏎. It's the same loop now, but the wheel's wedges come from the transformer rather than a counting table. In <code>microgpt.py</code> it's the last block of the file, lines 186 to 200.</p>
    <p>Think about a restaurant. You could <strong>always order the most popular dish</strong>: safe, predictable, and a bit dull. Or you could <strong>pick at random from the whole menu</strong>: adventurous, but you'll often regret it. A good diner is somewhere in between. Language models have exactly that choice, and a single number controls it.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch17-cold" kind="choice" options={['It writes a different creative name every time', 'It writes the same few names over and over', 'It writes gibberish']} answer={1}>
      {#snippet q()}<p>Suppose we make the model <strong>always pick the single most likely symbol</strong> (the "safest" dish, every time). What happens to the names it writes?</p>{/snippet}
      <p>The same few names, over and over. Always taking the top choice means every run follows the same single path. There's nothing random left to make names different. We want <em>mostly</em> likely symbols, with some room to wander. That's what <strong>temperature</strong> controls.</p>
    </Predict>
  </Beat>

  <Beat>
    <h2>Temperature</h2>
    <p>The fix is a small edit to a line you know. Before softmax turns the model's scores into probabilities, <strong>divide every score by a number called the temperature</strong>:</p>
    <div class="calc">
      <p><strong>In words:</strong> divide the scores by the temperature, then softmax as usual</p>
      <p><strong>In numbers:</strong> scores (2, 1, 0). At temperature 1: probabilities 67%, 24%, 9%. At temperature 0.5 the scores become (4, 2, 0): <strong>87%, 12%, 2%</strong>. At temperature 2 they become (1, 0.5, 0): <strong>51%, 31%, 19%</strong></p>
      <p><strong>In code:</strong> <code>softmax([l / temperature for l in logits])</code></p>
    </div>
    <ul>
      <li><strong>Low temperature</strong> (below 1) <em>stretches</em> the gaps between scores, so the favourite takes nearly all the probability. Cautious and repetitive.</li>
      <li><strong>Temperature 1</strong> leaves the model's own probabilities untouched.</li>
      <li><strong>High temperature</strong> <em>squeezes</em> the gaps, so even unlikely symbols get a real chance. Adventurous, and eventually random.</li>
    </ul>
  </Beat>

  <Beat gate>
    <h2>Try it on the real model</h2>
    <p>The top of the picture shows how the real model's probabilities reshape as you move the dial (the pale bars are temperature 1). Below it, 300 names generated at that setting, with measurements. <strong>Let go of the slider</strong> to regenerate. Try very low, the file's default of 0.5, 1, and very high.</p>
    <TempLab />
  </Beat>

  <Beat>
    <p>What we measured, 300 names per setting, from the fully trained model:</p>
    <table class="tbl">
      <thead><tr><td>Temperature</td><td>Real names (in the dataset)</td><td>Different from each other</td><td>What it looks like</td></tr></thead>
      <tbody>
        <tr><td>0.1</td><td>66%</td><td>only 13%</td><td>aris, aris, arile, aris… the same few names</td></tr>
        <tr><td>0.3</td><td>56%</td><td>72%</td><td>alyan, garien, jandel</td></tr>
        <tr><td><strong>0.5</strong> (the file's choice)</td><td>34%</td><td>97%</td><td>jariel, kaman, zayala</td></tr>
        <tr><td>1.0</td><td>5%</td><td>100%</td><td>acynnryn, lolatnn, euhahes</td></tr>
        <tr><td>2.0</td><td>2%</td><td>98%</td><td>abzonlle, moidtnem, gle: mostly nonsense</td></tr>
      </tbody>
    </table>
    <Aha title="Temperature trades reliability for variety">
      <p>There's no "correct" temperature. Turn it down and you get <em>reliable</em> output (many are names that really exist) but little variety. Turn it up and you get variety but rapidly lose quality. The file picks 0.5 as a compromise for names. Chatbots expose the same dial: a factual question wants a low temperature, a brainstorm a higher one. In the file it's line 187: <code>temperature = 0.5</code>, with the comment "in (0, 1], control the creativity of generated text, low to high".</p>
    </Aha>
  </Beat>

  <Beat gate>
    <Cell id="ch17-generate" plain={PLAIN['ch17-generate']} title="Write the name generator" code={genCode} pre={GPTFN} rows={17} check={genCheck} solution={genSolution}
      hint="Chapter 8 said only the gaps between scores matter, and a smaller number makes the gaps bigger. To make low temperature sharpen the choice, divide each score l by the temperature.">
      <p>One step is missing: how temperature changes the scores. You've written the rest before: it's the generation loop from Chapter 2 with the trained model in place of the counting table (lines 190 to 199 of <code>microgpt.py</code>).</p>
    </Cell>
  </Beat>

  <Beat>
    <h2>Training changes the writing</h2>
    <p>In the widget above, the <strong>"Model after step"</strong> selector lets you hear the model at five moments of its training. At the same temperature of 0.5 and the same dice, our recordings produced:</p>
    <table class="tbl">
      <thead><tr><td>After step</td><td>Eight names it wrote</td></tr></thead>
      <tbody>
        <tr><td>0</td><td class="mono">ab, snllgpug, duofjioh, (nothing), gxfdeiov, bypqs…</td></tr>
        <tr><td>50</td><td class="mono">aaynili, jiiariy, ele, dieeen, mavili, kitaen…</td></tr>
        <tr><td>200</td><td class="mono">ahurin, enenan, bmani, eraman, karin, urirany…</td></tr>
        <tr><td>1,000</td><td class="mono">ahylen, alian, jalen, vayan, alia, jenzi…</td></tr>
      </tbody>
    </table>
    <p>Noise, then vowel-consonant rhythm, then name-shaped, then names that you could meet at a party. Same dice, same temperature: only the dials changed.</p>
    <Sceptic q="Are the names copied from the data?">
      <p>Mostly not. At the default temperature only about a third (34%) of what it writes are names that exist in the dataset (the rest are new inventions), and some of those are short, common names it would hit by chance anyway. The model hasn't memorised a list. It has learned the <em>habits</em> of names and writes new ones that follow them. That's generalisation, which is what the whole course has been working towards.</p>
    </Sceptic>
    <Aha title="Why chatbots 'hallucinate'">
      <p>Try the name <code>xqz</code> in the temperature widget. The model happily continues it, even though no such name exists. It was never trained to <em>check</em> whether anything is true; it was trained to produce <strong>plausible continuations</strong>. A language model does one thing: given the text so far, produce likely next symbols. Where it has seen good evidence, plausible and true coincide. Where it hasn't, it produces something that <em>sounds</em> right. That is what people call "hallucination", and it comes straight from this loop, with no separate fault to fix.</p>
    </Aha>
  </Beat>

  <Beat>
    <Pain title="You know every line">
      <p>Count what's lit up in the file. Every line of <code>microgpt.py</code>, from the data at the top to the names at the bottom, now has an explanation you built yourself. The next chapter does something simple and a little emotional: it shows you the whole file, with nothing blurred, and asks you to read it.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch17">
      <ul>
        <li><strong>Generation</strong>: start at ⏎, run the model, choose a symbol from its probabilities, add it, repeat until it chooses ⏎ (lines 190–199).</li>
        <li><strong>Temperature</strong>: divide the scores by it before softmax. Low sharpens the choice (safe, repetitive), high flattens it (varied, then nonsense). Line 187 and the <code>/ temperature</code> in line 195.</li>
        <li>Measured: at 0.1, 66% real names but only 13% distinct; at 0.5, 34% real and 97% distinct; at 1.0, 5% real.</li>
        <li>The model writes <strong>new</strong> names, not copies. Training changed the writing from noise to name-like.</li>
        <li>"Hallucination" is the same loop: a model produces <strong>plausible</strong> continuations, not verified ones.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>

<style>
  :global(.tbl) { width: 100%; border-collapse: collapse; font-family: var(--font-ui); font-size: 0.9rem; margin: 1rem 0; }
  :global(.tbl td) { padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--line); vertical-align: top; }
  :global(.tbl thead td) { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
</style>
