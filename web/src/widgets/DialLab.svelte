<script>
  import { getContext } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { lossVC, bestVC } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';
  import Landscape from './Landscape.svelte';

  // Be the optimiser. Two dials, a loss computed from the REAL names, and one job: make the loss as low as you can.
  const beat = getContext('beat');
  let a = $state(0.5), b = $state(0.5);
  let trail = $state([{ a: 0.5, b: 0.5 }]);
  let turns = $state(0);
  let reveal = $state(false);
  let found = $state(false);

  const best = $derived(bestVC(data.vc));
  const minL = $derived(lossVC(data.vc, best.a, best.b));
  const loss = $derived(lossVC(data.vc, a, b));
  const goal = $derived(minL + 0.003);
  const meter = $derived(Math.max(0, Math.min(1, 1 - (loss - minL) / (Math.log(2) - minL))));

  function moved() {
    turns++;
    trail = [...trail.slice(-300), { a, b }];
    if (!found && loss < goal) { found = true; reveal = true; beat?.complete(); }
  }
  function pick(x, y) { a = +x.toFixed(3); b = +y.toFixed(3); moved(); }
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">The dial lab · two dials, one score</div>
    <div class="grid">
      <div class="left">
        <div class="dial">
          <label for="da"><strong>Dial A</strong> <span>chance the next letter is a vowel, when the last letter was a <em>vowel</em></span></label>
          <div class="row"><input id="da" type="range" min="0.01" max="0.99" step="0.005" bind:value={a} oninput={moved} /><output>{(a * 100).toFixed(1)}%</output></div>
        </div>
        <div class="dial">
          <label for="db"><strong>Dial B</strong> <span>chance the next letter is a vowel, when the last letter was a <em>consonant</em></span></label>
          <div class="row"><input id="db" type="range" min="0.01" max="0.99" step="0.005" bind:value={b} oninput={moved} /><output>{(b * 100).toFixed(1)}%</output></div>
        </div>

        <div class="score">
          <div class="lab">Loss (average surprise per letter pair, lower is better)</div>
          <div class="big">{loss.toFixed(4)}</div>
          <div class="meter" aria-hidden="true"><span style="width:{meter * 100}%"></span></div>
          <div class="lab">Starting point (both dials at 50%, pure coin-flipping): {Math.log(2).toFixed(4)}. Goal: get under <strong>{goal.toFixed(4)}</strong>.</div>
        </div>
        <div class="foot">
          <span class="muted">dial turns: {turns}</span>
          <button class="btn ghost" onclick={() => (reveal = !reveal)}>{reveal ? 'Hide the map' : 'Show me the map'}</button>
        </div>
        {#if found}
          <div class="win"><strong>Found it.</strong> Your dials: A = {(a * 100).toFixed(1)}%, B = {(b * 100).toFixed(1)}%. The best possible: {(best.a * 100).toFixed(1)}% and {(best.b * 100).toFixed(1)}%.</div>
        {/if}
      </div>
      <div class="right">
        <Landscape c={data.vc} {reveal} {trail} cur={{ a, b }} onpick={pick} />
        <div class="cap">{reveal ? 'Light = low loss. The valley is where you want to be. (Click the map to jump there.)' : 'Your path so far. Turn the dials, or click the map.'}</div>
      </div>
    </div>
  </div>
</DataGate>

<style>
  .grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 1.4rem; align-items: start; }
  @media (max-width: 820px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .dial { margin-bottom: 0.9rem; }
  .dial label { display: block; margin-bottom: 0.2rem; }
  .dial label span { color: var(--ink-2); font-size: 0.85rem; }
  .row { display: flex; gap: 0.8rem; align-items: center; }
  .row input { flex: 1; accent-color: var(--accent); }
  output { min-width: 3.8rem; text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }
  .score { background: var(--surface-2); border-radius: 12px; padding: 0.8rem 1rem; }
  .lab { color: var(--ink-3); font-size: 0.78rem; }
  .big { font-family: var(--font-display); font-size: 2.4rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
  .meter { height: 8px; background: var(--surface); border-radius: 4px; margin: 0.4rem 0 0.5rem; overflow: hidden; }
  .meter span { display: block; height: 100%; background: var(--series-3); border-radius: 4px; transition: width 0.15s; }
  .foot { display: flex; justify-content: space-between; align-items: center; margin-top: 0.6rem; }
  .win { margin-top: 0.7rem; padding: 0.6rem 0.8rem; border-radius: 10px; background: var(--good-wash); }
  .cap { color: var(--ink-3); font-size: 0.78rem; margin-top: 0.3rem; }
</style>
