<script>
  import Lesson from '../components/Lesson.svelte';
  import Beat from '../components/Beat.svelte';
  import Aha from '../components/Aha.svelte';
  import Predict from '../components/Predict.svelte';
  import Bank from '../components/Bank.svelte';
  import Replay from '../widgets/Replay.svelte';
  import FilePeek from '../widgets/FilePeek.svelte';
</script>

<Lesson id="ch0" num={0} title="The Trick" tagline="Watch 200 lines of Python learn to invent names. Then we take it apart.">
  <Beat>
    <p>Here is a strange fact. The idea underneath ChatGPT, Claude and every chatbot you've met, the <strong>GPT</strong>, fits in about <strong>200 lines of plain Python</strong>. No libraries doing the hard part. Not even a maths package.</p>
    <p>Andrej Karpathy published exactly that file. It's called <code>microgpt.py</code>, and its first lines say it plainly: <em>"This file is the complete algorithm. Everything else is just efficiency."</em></p>
    <p>Its job here is deliberately small: read <strong>32,033 human names</strong>, then invent new ones. Before I explain a single thing, watch it learn. Press play, or drag the slider.</p>
  </Beat>

  <Beat gate>
    <Replay />
  </Beat>

  <Beat>
    <p>That's a real recording: one actual run of the real file, nothing faked. At step 0 it emits noise, like <code>slhvgtbo</code>. By step 50 the output is <em>name-shaped</em>. By step 500 it says things like <code>kalia</code>, <code>ameli</code> and <code>kanni</code>.</p>
    <Aha title="The unsettling part">
      <p>Nobody told it what a name is. No list of vowels, no grammar, no rule about which letters can follow which. It was shown names, and it kept adjusting <strong>4,064 numbers</strong> so that real names would surprise it a little less each time.</p>
    </Aha>
    <p>That's all "learning" is here: nudging thousands of numbers, over and over, in whatever direction makes the real data less surprising. The mystery isn't in the recipe. It's in <em>which</em> numbers, nudged <em>which</em> way, and how a computer can possibly work that out. That's what you're going to invent.</p>
  </Beat>

  <Beat gate>
    <p>Here is the file itself. Open it, and notice that almost all of it is blurred. <strong>That's the game.</strong> Every time you finish a chapter, the lines you've actually understood light up. The narrow strip down the right edge of every page is a map of the whole file, and at the end of the course all of it glows.</p>
    <FilePeek />
  </Beat>

  <Beat>
    <h2>The deal</h2>
    <p>You said you know a little Python and not much maths. Good. That's exactly who this is for, and it shapes how the course works:</p>
    <ul>
      <li><strong>Nothing arrives before you need it.</strong> Every idea shows up as the fix for something that just broke. If it feels like homework, I got it wrong.</li>
      <li><strong>You guess first.</strong> You'll commit a prediction, then see what happens. Being wrong first is what makes the answer stick.</li>
      <li><strong>You write the real code,</strong> in the page, and it runs. No installing anything.</li>
      <li><strong>Maths comes as numbers, then code, then symbols.</strong> You already read code, so symbols are the last thing we add, not the first.</li>
      <li><strong>You can't break it.</strong> Every toy resets. Break things on purpose; there's a whole step for that.</li>
    </ul>
    <p>Two controls up top: <strong>Best loss</strong> opens a scoreboard of every model you build (it fills up in Chapter 3), and the <strong>⋮</strong> button turns off the one-scene-at-a-time reveal if you'd rather scroll.</p>
  </Beat>

  <Beat gate>
    <Predict id="ch0-idea" kind="text" min={12} placeholder="e.g. give it rules about vowels… or have it copy and remix real names… or…">
      {#snippet q()}<p>Before we start: <strong>how would <em>you</em> make a machine invent names that sound real?</strong> You've seen what it does and nothing about how. One or two sentences. There are no wrong answers, and I'll ask you to revisit this.</p>{/snippet}
      <p>Hold on to that idea. People usually reach for one of three: <em>write rules</em> ("vowels then consonants"), <em>copy and remix</em> real names, or <em>count which letters tend to follow which</em>.</p>
      <p>The first breaks on the very first exception. The second can never invent anything new. The third is where we'll begin, and it turns out to be closer to the real thing than it sounds.</p>
    </Predict>
  </Beat>

  <Beat last>
    <Bank chapter="ch0">
      <ul>
        <li>What the project is: a tiny program that learns to invent names from nothing but examples.</li>
        <li>That "learning" means adjusting thousands of numbers (4,064 here) until real data surprises the model less.</li>
        <li>The rules of the game: pain, guess, play, build, break, bank.</li>
      </ul>
    </Bank>
  </Beat>
</Lesson>
