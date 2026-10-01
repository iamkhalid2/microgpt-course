<script>
  // A text alternative for a chart: the same numbers as a table, tucked in a disclosure.
  let { caption = 'Table view', cols, rows } = $props();
</script>
<details class="tv">
  <summary>{caption}</summary>
  <div class="scroll">
    <table>
      <thead><tr>{#each cols as c}<th>{c}</th>{/each}</tr></thead>
      <tbody>{#each rows as r}<tr>{#each r as v}<td>{v}</td>{/each}</tr>{/each}</tbody>
    </table>
  </div>
</details>
<style>
  .tv { margin-top: 0.4rem; font-family: var(--font-ui); font-size: 0.78rem; color: var(--ink-2); }
  summary { cursor: pointer; color: var(--ink-3); }
  .scroll { max-height: 180px; overflow: auto; margin-top: 0.3rem; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 0.15rem 0.6rem 0.15rem 0; border-bottom: 1px solid var(--line); } th { color: var(--ink-3); font-weight: 500; }
</style>
