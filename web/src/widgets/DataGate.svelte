<script>
  import { data } from '../lib/data.svelte.js';
  let { children } = $props();
</script>
{#if data.status === 'ready'}
  {@render children()}
{:else if data.status === 'error'}
  <div class="widget wide">Couldn't load the names file ({data.error}). If you opened this page straight from disk, serve it with a web server instead.</div>
{:else}
  <div class="widget wide muted">Loading the names…</div>
{/if}
