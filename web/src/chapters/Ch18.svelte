<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Predict from '../components/Predict.svelte';
  import Bank from '../components/Bank.svelte';
  import FileReader from '../widgets/FileReader.svelte';
  import { href } from '../lib/router.svelte.js';

  const path = [
    ['Computers can\'t read letters', 'Give every character a number (tokens), plus a start/end symbol', 1],
    ['Random letters aren\'t names', 'Count the habits of real names: probabilities, sampling, the bigram table', 2],
    ['Which model is better?', 'Surprise, logarithms, the loss', 3],
    ['A counting table can\'t improve', 'Parameters: dials, turned until the loss is lowest', 4],
    ['Which way should each dial turn?', 'Slopes, found by a wiggle; gradient descent', 5],
    ['Dials are buried deep in a chain', 'The chain rule: multiply the slopes along the way', 6],
    ['The chain rule by hand is hopeless', 'An engine of numbers that remember how they were made (autograd)', 7],
    ['Dials make scores, not probabilities', 'Softmax: exp, then shares of the total', 8],
    ['Letters are islands', 'Embeddings (learned coordinates), the dot product, linear layers', 9],
    ['One letter of memory is too little', 'Attention: query, key, value, blend', 10],
    ['One question isn\'t enough', 'Several heads, each in its own slice', 11],
    ['Mixing isn\'t deciding', 'The MLP, with a ReLU gate to stop layers collapsing', 12],
    ['Deep stacks lose or explode the signal', 'Skip lanes and RMSNorm', 13],
    ['Pieces on a table aren\'t a model', 'Assemble gpt(): 4,064 dials', 14],
    ['Plain descent can\'t cope with different dial scales', 'Adam: momentum and per-dial step sizes, with a decaying rate', 15],
    ['One prediction isn\'t a lesson', 'Predict the next symbol at every position; the training loop', 16],
    ['Probabilities aren\'t names', 'Sampling and temperature', 17],
  ];
</script>

<Lesson id="ch18" num={18} title="The Reveal" tagline="Here is microgpt.py. Read it. There's nothing new in this chapter, which is the point.">
  <Beat>
    <p>In Chapter 0, I showed you this file almost entirely blurred and told you it would light up as you earned it. Open the file now (top right: <strong>The File</strong>) and you'll find <strong>every line lit</strong>: all 160 lines of code, each one earned by something you built or ran.</p>
    <p>This chapter doesn't teach anything. It lets you <em>read</em> the whole thing, from top to bottom, and see that you can. Below is the complete file, with a plain-English note for each part and a pointer to the chapter that built it. Browse it. Pick a part you remember being confusing, and see how it reads now.</p>
  </Beat>

  <Beat gate>
    <FileReader mode="read" />
    <p class="muted">Click at least five different parts. The coloured filter buttons show which lines each chapter built.</p>
  </Beat>

  <Beat gate>
    <h2>Find the line</h2>
    <p>A fair test of whether you can really navigate it: I'll describe what a line does, and you click it. If you get one wrong, you'll be told which line you clicked and given a hint for where to look.</p>
    <FileReader mode="quiz" />
  </Beat>

  <Beat gate>
    <Aha title="Could you have come up with it?">
      <p>That's the question this course set out to answer. Look at the path you've just walked. Every idea arrived as the <strong>obvious fix to a problem you could see</strong>, and none of the fixes needed a leap of genius. Here's the whole chain:</p>
    </Aha>
    <table class="path wide">
      <thead><tr><td>The problem we hit</td><td>What we invented</td><td>Chapter</td></tr></thead>
      <tbody>
        {#each path as [problem, fix, ch]}
          <tr><td>{problem}</td><td>{fix}</td><td><a href={href(`ch/${ch}`)}>{ch}</a></td></tr>
        {/each}
      </tbody>
    </table>
    <Predict id="ch18-hard" kind="text" min={15} placeholder="e.g. attention, because… / the chain rule, because…">
      {#snippet q()}<p>Be honest. <strong>Which single idea in that list do you think you could have come up with yourself, given the problem in front of you? And which would have stumped you?</strong> One or two sentences.</p>{/snippet}
      <p>There's no right answer, but here's something worth noticing. For almost every row, <em>stating the problem</em> is most of the work: once you see that "letters are islands", the idea of giving them coordinates is close to obvious, and once you see that "stacked layers collapse", you start looking for a gate. The hard part of research is usually noticing the problem. And that's the part this course tried to give you: each chapter began with you <em>feeling</em> the pain, which made the fix feel like yours.</p>
      <p>If you named attention as the stumper, you're in good company: it took the field about three years from the first attention paper to the transformer, and people argued about it for years afterwards.</p>
    </Predict>
  </Beat>

  <Beat>
    <Pain title="One more test">
      <p>You've read the file and found your way around it. There's a harder test, and it's the one the course was named for: <strong>can you build it from a blank page?</strong> Not by copying. By starting with an empty file and a list of milestones, using what you've understood. The last chapter gives you that blank page, with hints at three levels if you get stuck, and then lets you change the model and watch what happens: new names, new sizes, new ideas.</p>
    </Pain>
  </Beat>

  <Beat last>
    <Bank chapter="ch18">
      <ul>
        <li>You can read <strong>all 160 lines of code</strong> of a complete GPT, and say what each part is for.</li>
        <li>Each idea was a <strong>fix for a problem you could see</strong>: the whole design is a chain of 17 such steps.</li>
        <li>The file's own description is true: this is the complete algorithm. What big models add is scale, not new ideas (next chapter).</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>

<style>
  .path { width: 100%; border-collapse: collapse; font-family: var(--font-ui); font-size: 0.88rem; margin: 1rem auto; }
  .path td { padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--line); vertical-align: top; }
  .path thead td { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .path td:last-child { text-align: center; }
</style>
