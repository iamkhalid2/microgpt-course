<script>
  // The whole data path of microgpt's gpt() function, as a picture. Boxes light up when named in `lit`.
  let { lit = [], title = 'The path of one letter through the model' } = $props();
  const on = (k) => lit.includes(k);
  const boxes = [
    { k: 'emb', x: 10, y: 70, w: 100, t1: 'letter + position', t2: 'lines 109–111' },
    { k: 'norm0', x: 126, y: 70, w: 70, t1: 'RMSNorm', t2: 'line 112' },
    { k: 'attn', x: 230, y: 24, w: 130, t1: 'RMSNorm → attention', t2: 'lines 117–133' },
    { k: 'mlp', x: 410, y: 24, w: 130, t1: 'RMSNorm → MLP', t2: 'lines 137–140' },
    { k: 'head', x: 590, y: 70, w: 100, t1: 'lm_head → scores', t2: 'line 143' },
  ];
</script>
<div class="bd">
  <div class="t">{title}</div>
  <svg viewBox="0 0 700 150" role="img" aria-label="Flow: embedding, norm, attention block, MLP block, output head, with skip lanes">
    <defs><marker id="bda" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" style="fill: var(--ink-3);" /></marker></defs>
    <!-- main lane -->
    <path d="M110 92 H126 M196 92 H214 L214 48 H230 M360 48 H376 L376 92 H394 L394 48 H410 M540 48 H556 L556 92 H590" fill="none" stroke="var(--line-strong)" stroke-width="1.6" marker-end="url(#bda)" />
    <!-- skip lanes -->
    <path d="M214 92 H368 Q380 92 380 100 H382" fill="none" style="stroke: {on('skip') ? 'var(--series-2)' : 'var(--line-strong)'}; stroke-width: {on('skip') ? 2.6 : 1.6}; stroke-dasharray: 5 4;" />
    <path d="M394 112 H548 Q556 112 556 104 V96" fill="none" style="stroke: {on('skip') ? 'var(--series-2)' : 'var(--line-strong)'}; stroke-width: {on('skip') ? 2.6 : 1.6}; stroke-dasharray: 5 4;" />
    <circle cx="380" cy="92" r="9" style="fill: var(--surface); stroke: {on('skip') ? 'var(--series-2)' : 'var(--line-strong)'}; stroke-width: 2;" /><text x="380" y="96.5" text-anchor="middle" class="plus">+</text>
    <circle cx="556" cy="92" r="9" style="fill: var(--surface); stroke: {on('skip') ? 'var(--series-2)' : 'var(--line-strong)'}; stroke-width: 2;" /><text x="556" y="96.5" text-anchor="middle" class="plus">+</text>
    {#each boxes as b}
      <rect x={b.x} y={b.y} width={b.w} height="44" rx="9" style="fill: {on(b.k) ? 'var(--accent-wash)' : 'var(--surface)'}; stroke: {on(b.k) ? 'var(--accent)' : 'var(--line-strong)'}; stroke-width: {on(b.k) ? 2.4 : 1.4};" />
      <text x={b.x + b.w / 2} y={b.y + 19} text-anchor="middle" class="b1">{b.t1}</text><text x={b.x + b.w / 2} y={b.y + 34} text-anchor="middle" class="b2">{b.t2}</text>
    {/each}
    <text x="290" y="132" class="b2" text-anchor="middle">skip lane: the block's output is ADDED to what was already there</text>
  </svg>
</div>
<style>
  .bd { background: var(--surface-2); border-radius: 12px; padding: 0.6rem 0.8rem; } .t { font-size: 0.78rem; color: var(--ink-2); font-weight: 600; margin-bottom: 0.2rem; }
  svg { width: 100%; height: auto; display: block; } .b1 { font-family: var(--font-ui); font-size: 11px; font-weight: 600; fill: var(--ink); } .b2 { font-family: var(--font-mono); font-size: 9.5px; fill: var(--ink-3); } .plus { font-family: var(--font-ui); font-size: 14px; font-weight: 700; fill: var(--ink); }
</style>
