<script> let { title = 'Aha', children } = $props(); </script>
<aside class="aha">
  <div class="lab">{title}</div>
  <div class="body">{@render children()}</div>
</aside>
<style>
  .aha { border-left: 4px solid var(--accent); background: var(--accent-wash); padding: 0.9rem 1.2rem; border-radius: 0 var(--radius) var(--radius) 0; margin-bottom: 1.4rem; }
  .lab { font-family: var(--font-ui); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent-strong); margin-bottom: 0.3rem; }
  .body :global(p:last-child) { margin-bottom: 0; }
</style>
