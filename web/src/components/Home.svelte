<script>
  import { onMount } from 'svelte';
  import { acts, chapters } from '../chapters/index.js';
  import { progress } from '../lib/progress.svelte.js';
  import { href } from '../lib/router.svelte.js';
  import { ui } from '../lib/ui.svelte.js';
  // Loaded on its own so that if the 3D piece can't load (a blocker, an old browser), the rest of the page still does.
  let Hero3D = $state(null);
  onMount(() => { import('./Hero3D.svelte').then((m) => { Hero3D = m.default; }).catch(() => {}); });

  // Hero ticker: real names from the recorded microgpt run, at the training step they were made.
  let replay = $state(null);
  let tick = $state({ step: 0, name: '' });
  onMount(() => {
    fetch(new URL('data/replay.json', document.baseURI)).then((r) => r.json()).then((d) => {
      replay = d;
      const steps = Object.keys(d.snaps).map(Number).sort((a, b) => a - b);
      let i = 0;
      const show = () => {
        const s = steps[i % steps.length];
        const list = d.snaps[String(s)].filter((n) => n.length > 1);
        tick = { step: s, name: list[Math.floor(Math.random() * list.length)] ?? '' };
        i++;
      };
      show();
      const id = setInterval(show, 1100);
      window.__tickId = id;
    });
    return () => clearInterval(window.__tickId);
  });

  const next = $derived(chapters.find((c) => c.ready && !progress.banked[c.id]) ?? chapters[0]);
  const started = $derived(Object.keys(progress.banked).length > 0 || Object.keys(progress.beat).length > 0);
  const loop = [
    ['Pain', 'Something just failed. You feel why it matters.'],
    ['Guess', 'Commit a prediction before you’re shown anything.'],
    ['Play', 'Poke an interactive toy until the idea clicks.'],
    ['Build', 'Write the real code, in the page, and it runs.'],
    ['Break', 'Sabotage the model to see why each part exists.'],
    ['Bank', 'Light up the matching lines of microgpt.py.'],
  ];
</script>

<div class="home">
  <section class="hero">
   <div class="htext">
    <div class="eyebrow">A first-principles course · no math degree · no Python mastery</div>
    <h1>Invent a GPT.</h1>
    <p class="lead">Andrej Karpathy wrote a complete language model in <strong>200 lines of plain Python</strong>. You're going to reinvent it, one broken idea at a time, without leaving this page. By the end you won't feel like you studied it. You'll feel like you could have come up with it.</p>

    <div class="ticker widget" aria-live="off">
      <div class="tlabel">A name the real 200-line model invents after</div>
      <div class="tstep"><span class="mono">{tick.step}</span> training step{tick.step === 1 ? '' : 's'}</div>
      <div class="tname mono" class:gib={tick.step < 25}>{tick.name || '…'}</div>
    </div>

    <div class="cta">
      <a class="btn primary big" href={href(`ch/${next.num}`)}>{started ? `Continue: ${next.title}` : 'Start with The Trick'} →</a>
      <button class="btn big" onclick={() => (ui.fileOpen = true)}>Peek at the 200 lines</button>
    </div>
   </div>
   <div class="hviz">{#if Hero3D}<Hero3D />{/if}</div>
  </section>

  <section class="loop">
    <h2>How every chapter works</h2>
    <div class="cards">
      {#each loop as [t, d], i}
        <div class="card"><div class="n">{i + 1}</div><h3>{t}</h3><p>{d}</p></div>
      {/each}
    </div>
  </section>

  <section class="map">
    <h2>The path</h2>
    <p class="sub">Each chapter is the answer to a problem the previous one left you with. Nothing is introduced before you need it.</p>
    {#each acts as a}
      <div class="act">
        <div class="actname">{a.name}<span>{a.blurb}</span></div>
        <ol class="chs">
          {#each chapters.filter((c) => c.act === a.id) as c}
            <li class:ready={c.ready} class:done={progress.banked[c.id]}>
              <svelte:element this={c.ready ? 'a' : 'div'} class="row" href={c.ready ? href(`ch/${c.num}`) : undefined}>
                <span class="dot">{progress.banked[c.id] ? '✓' : c.num}</span>
                <span class="meta">
                  <span class="t">{c.title}{#if !c.ready}<em>being built</em>{/if}</span>
                  <span class="pn">{c.pain}</span>
                  <span class="iv">{c.invent}</span>
                </span>
              </svelte:element>
            </li>
          {/each}
        </ol>
      </div>
    {/each}
  </section>

  <footer class="foot">
    <p>Based on <a href="https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95">microgpt.py</a> by Andrej Karpathy. Progress is saved only in this browser. Every number on this site is computed by code that actually ran, and the JavaScript that powers the toys is tested against independent Python.</p>
  </footer>
</div>

<style>
  .home { max-width: 62rem; margin: 0 auto; padding: 3.2rem 1.25rem 6rem; }
  .hero { display: grid; grid-template-columns: minmax(0, 1fr); gap: 2.4rem; align-items: start; }
  .hviz { width: 100%; max-width: 30rem; margin-inline: auto; }
  @media (min-width: 960px) { .hero { grid-template-columns: minmax(0, 1fr) minmax(0, 25.5rem); gap: 2.2rem; } .hviz { margin: 1.4rem 0 0; max-width: none; } }
  .eyebrow { font-family: var(--font-ui); font-size: 0.74rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3); font-weight: 600; margin-bottom: 1rem; }
  h1 { font-size: clamp(3rem, 9vw, 5.6rem); margin-bottom: 0.3em; }
  .lead { font-size: 1.35rem; max-width: 41rem; color: var(--ink-2); }
  .ticker { display: inline-block; margin: 1.4rem 0 1.6rem; min-width: min(20rem, 100%); text-align: center; padding: 1.2rem 2rem; }
  .tlabel { color: var(--ink-3); font-size: 0.78rem; }
  .tstep { font-size: 0.85rem; color: var(--ink-2); margin: 0.2rem 0 0.4rem; }
  .tname { font-size: 2.6rem; font-weight: 500; letter-spacing: 0.02em; min-height: 1.3em; color: var(--ink); }
  .tname.gib { color: var(--pain); }
  .cta { display: flex; gap: 0.7rem; flex-wrap: wrap; }
  .big { font-size: 0.95rem; padding: 0.75rem 1.3rem; text-decoration: none; display: inline-block; }
  .loop, .map { margin-top: 4.5rem; }
  .sub { color: var(--ink-2); max-width: 40rem; }
  .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin-top: 1.2rem; }
  @media (max-width: 720px) { .cards { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 420px) { .cards { grid-template-columns: 1fr; } }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 1rem 1.1rem; font-family: var(--font-ui); }
  .card .n { font-size: 0.72rem; color: var(--accent-strong); font-weight: 700; }
  .card h3 { font-size: 1.15rem; margin: 0.1rem 0 0.3rem; }
  .card p { font-family: var(--font-body); font-size: 0.95rem; color: var(--ink-2); margin: 0; line-height: 1.45; }
  .act { margin-top: 2.2rem; }
  .actname { font-family: var(--font-display); font-weight: 700; font-size: 1.25rem; display: flex; gap: 0.8rem; align-items: baseline; flex-wrap: wrap; }
  .actname span { font-family: var(--font-body); font-weight: 400; font-style: italic; font-size: 1rem; color: var(--ink-3); }
  .chs { list-style: none; margin: 0.7rem 0 0; padding: 0 0 0 1.05rem; border-left: 2px solid var(--line); }
  .chs li { position: relative; margin-bottom: 0.3rem; }
  .row { display: flex; gap: 0.9rem; padding: 0.6rem 0.8rem; border-radius: 12px; text-decoration: none; color: inherit; margin-left: 0.6rem; }
  li.ready .row:hover { background: var(--surface); box-shadow: var(--shadow); }
  li:not(.ready) { opacity: 0.55; }
  .dot { flex: none; width: 2rem; height: 2rem; border-radius: 50%; display: grid; place-items: center; font-family: var(--font-ui); font-weight: 700; font-size: 0.85rem; background: var(--surface); border: 2px solid var(--line-strong); margin-left: -3.45rem; margin-right: 0.3rem; position: relative; }
  li.ready .dot { border-color: var(--accent); color: var(--accent-strong); }
  li.done .dot { background: var(--good); border-color: var(--good); color: #fff; }
  .meta { display: flex; flex-direction: column; }
  .t { font-family: var(--font-display); font-weight: 700; font-size: 1.1rem; }
  .t em { font-family: var(--font-ui); font-style: normal; font-weight: 500; font-size: 0.68rem; color: var(--ink-3); border: 1px solid var(--line-strong); border-radius: 999px; padding: 0.05rem 0.5rem; margin-left: 0.6rem; vertical-align: middle; }
  .pn { color: var(--ink-2); font-size: 0.98rem; }
  .iv { color: var(--ink-3); font-size: 0.9rem; font-style: italic; }
  .foot { margin-top: 5rem; color: var(--ink-3); font-size: 0.9rem; max-width: 42rem; }
</style>
