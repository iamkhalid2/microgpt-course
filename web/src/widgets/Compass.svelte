<script>
  import { getContext } from 'svelte';
  import { data } from '../lib/data.svelte.js';
  import { wiggleVC } from '../lib/stats.js';
  import DataGate from './DataGate.svelte';
  import Landscape from './Landscape.svelte';

  // At any setting of the two dials, the slopes say which way is downhill. Click the map to ask.
  const beat = getContext('beat');
  let at = $state({ a: 0.5, b: 0.5 });
  let field = $state(false);
  let asked = new Set(['0.5,0.5']);

  const g = $derived(wiggleVC(data.vc, at.a, at.b));
  const arrows = $derived.by(() => {
    const out = [{ a: at.a, b: at.b, da: g.da, db: g.db, big: true }];
    if (field) for (let i = 1; i <= 8; i++) for (let j = 1; j <= 8; j++) {
      const a = i / 9, b = j / 9; const w = wiggleVC(data.vc, a, b);
      out.push({ a, b, da: w.da, db: w.db });
    }
    return out;
  });
  function pick(a, b) { at = { a, b }; asked.add(`${a.toFixed(1)},${b.toFixed(1)}`); if (asked.size >= 4) beat?.complete(); }
  const word = (s, name) => Math.abs(s) < 0.02 ? `${name}: nearly flat. Leave it alone.` : s > 0 ? `${name}: raising it raises the loss, so turn it DOWN.` : `${name}: raising it lowers the loss, so turn it UP.`;
</script>

<DataGate>
  <div class="widget wide">
    <div class="widget-title">The compass · click anywhere on the map</div>
    <div class="grid">
      <Landscape c={data.vc} reveal={true} cur={at} onpick={pick} {arrows} />
      <div>
        <div class="at">Dials at <strong>A = {(at.a * 100).toFixed(0)}%</strong>, <strong>B = {(at.b * 100).toFixed(0)}%</strong></div>
        <div class="sl"><span class="n mono">{g.da >= 0 ? '+' : ''}{g.da.toFixed(3)}</span><span>{word(g.da, 'Dial A')}</span></div>
        <div class="sl"><span class="n mono">{g.db >= 0 ? '+' : ''}{g.db.toFixed(3)}</span><span>{word(g.db, 'Dial B')}</span></div>
        <div class="note">The black arrow points <strong>downhill</strong>: the direction that lowers the loss fastest. Its length is how steep it is here.</div>
        <label class="tg"><input type="checkbox" bind:checked={field} /> show the arrow at many spots</label>
        <div class="note">Notice how the arrows all lean toward the valley, and shrink to nothing at the bottom. <strong>A slope of zero means flat: you've arrived.</strong></div>
      </div>
    </div>
  </div>
</DataGate>

<style>
  .grid { display: grid; grid-template-columns: minmax(240px, 340px) 1fr; gap: 1.5rem; align-items: start; }
  @media (max-width: 760px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .at { margin-bottom: 0.6rem; }
  .sl { display: flex; gap: 0.7rem; align-items: baseline; background: var(--surface-2); border-radius: 10px; padding: 0.5rem 0.7rem; margin-bottom: 0.4rem; }
  .n { font-weight: 700; min-width: 4.4rem; background: none; padding: 0; font-size: 1rem; }
  .note { color: var(--ink-2); font-size: 0.85rem; margin: 0.6rem 0; }
  .tg { display: flex; gap: 0.4rem; align-items: center; color: var(--ink-2); }
</style>
