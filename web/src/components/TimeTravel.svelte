<script> let { year, who, children } = $props(); </script>
<aside class="tt">
  <div class="year">{year}</div>
  <div>
    <div class="who">Time travel · {who}</div>
    <div class="body">{@render children()}</div>
  </div>
</aside>
<style>
  .tt { display: grid; grid-template-columns: auto 1fr; gap: 1.1rem; align-items: start; background: var(--gold-wash); border: 1px solid color-mix(in srgb, var(--gold) 40%, transparent); padding: 1rem 1.2rem; border-radius: var(--radius); margin-bottom: 1.4rem; font-size: 1rem; }
  .year { font-family: var(--font-display); font-weight: 700; font-size: 2rem; color: var(--gold); line-height: 1; padding-top: 0.15rem; font-variant-numeric: lining-nums; }
  .who { font-family: var(--font-ui); font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); font-weight: 600; margin-bottom: 0.3rem; }
  .body :global(p:last-child) { margin-bottom: 0; }
  @media (max-width: 520px) { .tt { grid-template-columns: 1fr; gap: 0.4rem; } }
</style>
