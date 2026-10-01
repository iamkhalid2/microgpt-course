<script> let { title = 'The pain', children } = $props(); </script>
<aside class="pain">
  <div class="lab">{title}</div>
  <div class="body">{@render children()}</div>
</aside>
<style>
  .pain { border-left: 4px solid var(--pain); background: var(--pain-wash); padding: 0.9rem 1.2rem; border-radius: 0 var(--radius) var(--radius) 0; margin-bottom: 1.4rem; }
  .lab { font-family: var(--font-ui); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--pain); margin-bottom: 0.3rem; }
  .body :global(p:last-child) { margin-bottom: 0; }
</style>
