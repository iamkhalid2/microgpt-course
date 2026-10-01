<script>
  import { getContext } from 'svelte';
  import { ui } from '../lib/ui.svelte.js';
  import { litCount } from '../lib/progress.svelte.js';
  import { CODE_LINES } from '../lib/fileMap.js';

  const beat = getContext('beat');
  function open() { ui.fileOpen = true; beat?.complete(); }
</script>
<div class="widget wide peek">
  <div>
    <div class="big">{litCount()} <span>of {CODE_LINES.length} lines earned</span></div>
    <div class="muted">Everything else is waiting for its chapter.</div>
  </div>
  <button class="btn primary" onclick={open}>Open the file</button>
</div>
<style>
  .peek { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }
  .big { font-size: 2rem; font-weight: 700; font-family: var(--font-display); }
  .big span { font-family: var(--font-ui); font-size: 0.9rem; font-weight: 400; color: var(--ink-2); }
</style>
