<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Pain from '../components/Pain.svelte';
  import Aha from '../components/Aha.svelte';
  import Sceptic from '../components/Sceptic.svelte';
  import Predict from '../components/Predict.svelte';
  import Bank from '../components/Bank.svelte';
  import Capstone from '../widgets/Capstone.svelte';
  import MutationLab from '../widgets/MutationLab.svelte';
  import { progress, litCount } from '../lib/progress.svelte.js';
  import { href } from '../lib/router.svelte.js';

  const nBanked = $derived(Object.keys(progress.banked).length);
  const nSolved = $derived(Object.keys(progress.solved).length);
  const nCap = $derived([1, 2, 3, 4, 5].filter((i) => progress.solved['capstone-' + i]).length);
  const nLit = $derived(litCount());
</script>

<Lesson id="ch19" num={19} title="Blank File" tagline="Write it from nothing. Then change it, and find out what changes.">
  <Beat>
    <p>You've read the whole file and recognised every line. Recognising isn't the same as producing, though, as anyone who has studied a language and then tried to speak it knows. So here's the real test.</p>
    <p>Below is an <strong>empty editor</strong> and five milestones. Your job is to write <code>microgpt.py</code> yourself: the data, the engine, the model, the training loop, and the voice. After each milestone you press <em>Check it</em> and a test runs your code against known answers. There's nothing to download and nobody watching.</p>
    <Pain title="You are allowed to get stuck">
      <p>Getting stuck is the normal state of writing code, even for professionals. Each milestone has three levels of help: <strong>the idea</strong> in plain words, <strong>an outline</strong> you can paste in as comments, and <strong>the real code</strong> for that part. Use the lowest level that unsticks you. Using level 3 isn't failing; typing out code you've understood is a fine way to learn it, and you can always go back and rewrite a part without looking.</p>
    </Pain>
  </Beat>

  <Beat gate>
    <Predict id="ch19-guess" kind="number" answer={160} tolerance={60}>
      {#snippet q()}<p>Before you start: <strong>how many lines of code do you think you'll write</strong> for the whole thing? (Blank lines and comments don't count.)</p>{/snippet}
      <p>The real file is 160 lines of code, plus comments. Roughly a quarter of it is the autograd engine, a quarter the model, and the rest data, training and sampling. Yours may be longer or shorter: that's fine, as long as the milestones pass.</p>
    </Predict>
  </Beat>

  <Beat>
    <Capstone />
    <p class="muted">Your file is saved in this browser as you type. The tests use the same names as the real file (<code>docs</code>, <code>uchars</code>, <code>BOS</code>, <code>vocab_size</code>, <code>Value</code>, <code>params</code>, <code>gpt</code>…) so they can find your work. Milestones 4 and 5 run a short 40-step training to check that your loop really learns; <em>Run my whole file</em> runs it exactly as written.</p>
    <Sceptic q="This is a lot. Can I skip it and come back?">
      <p>Yes. Everything on the page after this is available now, and your file stays saved. But try at least milestone 1 and a hint on milestone 2: the first time you write <code>class Value</code> from a blank page is when it stops being something you read and becomes something you can do.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <h2>Now make it yours</h2>
    <p>You've seen one program train on one dataset. A program that learns from <em>whatever you give it</em> is a stranger and more interesting thing. This lab runs the real <code>microgpt.py</code> in your browser, with the settings you choose on the data and the model, and keeps a table of your experiments so you can compare.</p>
    <p>Some experiments worth running:</p>
    <ul>
      <li><strong>Different data.</strong> Dinosaur names, or city names, or paste in your own list: pet names, brands, anything with a pattern, one per line. The code doesn't change at all; only the habits it learns.</li>
      <li><strong>Bigger or smaller.</strong> One layer or three. Width 8 or 32. Does the loss get lower? How much longer does it take?</li>
      <li><strong>The learning rate.</strong> Make it ten times smaller, and then ten times larger. Chapter 5 predicted both outcomes.</li>
    </ul>
    <MutationLab />
  </Beat>

  <Beat>
    <h2>Beyond</h2>
    <p>The model you built has 4,064 dials, reads 27 symbols, and remembers 8 positions. A modern assistant is this same program, with the numbers turned up. Here is what actually changes:</p>
    <table class="tbl">
      <thead><tr><td></td><td>microgpt</td><td>A large language model</td></tr></thead>
      <tbody>
        <tr><td><strong>Symbols</strong></td><td>26 letters + 1 start/end symbol</td><td>Around 100,000 <em>chunks</em> of text (word pieces such as "ing", " the", "tion"), found by counting which pairs occur most (byte-pair encoding). Same idea as Chapter 1: give every symbol a number.</td></tr>
        <tr><td><strong>Data</strong></td><td>32,033 names (about 190,000 letters), 500 used</td><td>Trillions of symbols from books, websites and code</td></tr>
        <tr><td><strong>Dials</strong></td><td>4,064</td><td>Hundreds of billions to trillions</td></tr>
        <tr><td><strong>Memory</strong></td><td>8 positions</td><td>Hundreds of thousands to millions</td></tr>
        <tr><td><strong>Layers / heads</strong></td><td>1 layer, 4 heads</td><td>Dozens to over a hundred layers, with many heads each</td></tr>
        <tr><td><strong>Hardware</strong></td><td>One ordinary CPU, one name at a time, written in plain Python</td><td>Thousands of GPUs, doing the same arithmetic on big blocks of numbers at once (the same dot products, in bulk)</td></tr>
        <tr><td><strong>After training</strong></td><td>Nothing: it just writes names</td><td>A second stage teaches it to follow instructions and be helpful, by training on human-written examples and on human judgements of its answers</td></tr>
      </tbody>
    </table>
    <Aha title="Scale, not a new idea">
      <p>The ingredients in that last column are engineering, and a lot of hard-won engineering: making things fast, stable and cheap enough to run at that size. But the <em>algorithm</em> is the one you wrote: predict the next symbol, measure the surprise, find each dial's slope with the chain rule, and nudge. The file's own description says it: <em>"This file is the complete algorithm. Everything else is just efficiency."</em></p>
    </Aha>
    <Sceptic q="So is a chatbot just predicting the next word?">
      <p>At the level of mechanism, yes: it writes one symbol at a time, from probabilities. But "just" hides how much a model must learn to do that well. To predict the next word in a mystery story's last chapter, it helps to have tracked who the murderer is. Nobody wrote a rule for that. Predicting well on a vast range of text <em>pays</em> for learning facts, grammar, and reasoning habits, and the training loop finds them without being told. Whether that counts as "understanding" is an argument the field is still having; what you can say with confidence is that you now know exactly what the machinery is, and where the mystery isn't.</p>
    </Sceptic>
  </Beat>

  <Beat>
    <h2>Where to go next</h2>
    <p>You have the foundation that every one of these builds on, and you can now read them without feeling lost:</p>
    <ul>
      <li><strong>Andrej Karpathy, "Neural Networks: Zero to Hero"</strong> (video series): micrograd (the engine from Chapter 7), makemore (names again, with bigger models) and "Let's build GPT" (the same model, with batching). The closest next step.</li>
      <li><strong>nanoGPT</strong> (code): this model rewritten in PyTorch, which does the autograd and the GPU work for you. You'll recognise every part.</li>
      <li><strong>"Attention Is All You Need"</strong> (the 2017 paper that introduced the transformer). With Chapters 9 to 14 behind you, the architecture diagram on page 3 is now readable.</li>
      <li><strong>Learn PyTorch</strong>, then try training this model on a dataset of your own, with ten thousand lines instead of a few hundred.</li>
    </ul>
    <p>And the one that matters most: <strong>keep rebuilding things from a blank page.</strong> The skill you've practised here, noticing a problem and asking "what's the simplest fix?", is the same skill the researchers used.</p>
  </Beat>

  <Beat>
    <div class="finale widget wide">
      <div class="widget-title">What you did</div>
      <div class="stats">
        <div><strong>{nBanked}</strong><span>chapters banked</span></div>
        <div><strong>{nLit}</strong><span>of 160 lines lit</span></div>
        <div><strong>{nSolved}</strong><span>exercises solved</span></div>
        <div><strong>{nCap}/5</strong><span>blank-page milestones</span></div>
      </div>
      <p>You started with a file of 200 lines that you couldn't read. You finish able to say what every line is for, <em>why</em> it's there, and what problem it solved, and you can rebuild it. Karpathy's file is a small thing in a very big field, and you now understand the whole of it. <a href={href('')}>Back to the map</a> or open <strong>The File</strong> at the top right to see it fully lit.</p>
    </div>
  </Beat>

  <Beat last>
    <Bank chapter="ch19">
      <ul>
        <li>You can <strong>rebuild a GPT from a blank page</strong>: data, engine, model, training loop, sampling.</li>
        <li>You've changed the data and the model and <strong>watched what changed</strong>.</li>
        <li>A large language model is this algorithm plus <strong>scale</strong>: more symbols, data, dials, layers, memory and hardware, then a stage of instruction-following.</li>
        <li>You could have come up with it. Now you have.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>

<style>
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.8rem; margin: 0.6rem 0 1rem; }
  .stats div { text-align: center; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 0.8rem 0.4rem; }
  .stats strong { display: block; font-family: var(--font-display, var(--font-ui)); font-size: 1.9rem; color: var(--accent); }
  .stats span { font-family: var(--font-ui); font-size: 0.75rem; color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.05em; }
</style>
