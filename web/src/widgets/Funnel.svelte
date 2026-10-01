<script>
  import { getContext } from 'svelte';

  // A business funnel: spend -> clicks -> signups -> customers -> revenue. If you know each stage's conversion,
  // what does one extra pound of spend do to revenue? Multiply the stage ratios. That is the chain rule.
  const beat = getContext('beat');
  let spend = $state(1000);
  let r = $state({ clicks: 0.8, signups: 0.10, customers: 0.25, revenue: 40 });
  let moved = 0;

  const stages = $derived.by(() => {
    const clicks = spend * r.clicks, signups = clicks * r.signups, customers = signups * r.customers, revenue = customers * r.revenue;
    return [
      { key: 'spend', name: 'Ad spend', val: spend, fmt: (v) => '£' + Math.round(v).toLocaleString() },
      { key: 'clicks', name: 'Clicks', val: clicks, fmt: (v) => v.toFixed(0) },
      { key: 'signups', name: 'Signups', val: signups, fmt: (v) => v.toFixed(1) },
      { key: 'customers', name: 'Paying customers', val: customers, fmt: (v) => v.toFixed(2) },
      { key: 'revenue', name: 'Revenue', val: revenue, fmt: (v) => '£' + v.toFixed(0) },
    ];
  });
  const sens = $derived(r.clicks * r.signups * r.customers * r.revenue);
  // the wiggle test: spend one extra pound, recompute the whole funnel, see what revenue did
  const wig = $derived(((spend + 1) * r.clicks * r.signups * r.customers * r.revenue) - stages[4].val);
  const touch = () => { if (++moved >= 3) beat?.complete(); };
  const ctl = [
    ['clicks', 'clicks per £ of spend', 0.2, 2, 0.05],
    ['signups', 'share of clicks who sign up', 0.02, 0.4, 0.01],
    ['customers', 'share of signups who pay', 0.05, 0.6, 0.01],
    ['revenue', '£ per paying customer', 10, 100, 1],
  ];
</script>

<div class="widget wide">
  <div class="widget-title">The funnel · one extra pound, all the way down</div>
  <div class="flow">
    {#each stages as s, i}
      <div class="stage"><div class="nm">{s.name}</div><div class="v">{s.fmt(s.val)}</div></div>
      {#if i < stages.length - 1}
        <div class="arrow"><span>× {(i === 0 ? r.clicks : i === 1 ? r.signups : i === 2 ? r.customers : r.revenue).toFixed(i === 3 ? 0 : 2)}</span></div>
      {/if}
    {/each}
  </div>

  <div class="sliders">
    <label class="sl">Ad spend <input type="range" min="200" max="3000" step="50" bind:value={spend} oninput={touch} /> <strong>£{spend.toLocaleString()}</strong></label>
    {#each ctl as [k, label, lo, hi, st]}
      <label class="sl">{label} <input type="range" min={lo} max={hi} step={st} bind:value={r[k]} oninput={touch} /> <strong>{k === 'revenue' ? '£' + r[k] : r[k].toFixed(2)}</strong></label>
    {/each}
  </div>

  <div class="res">
    <div class="cascade">
      <div class="h">Spend one <strong>extra £1</strong> and it cascades:</div>
      <div class="casc mono">
        +£1 → +{r.clicks.toFixed(2)} clicks → +{(r.clicks * r.signups).toFixed(3)} signups → +{(r.clicks * r.signups * r.customers).toFixed(4)} customers → <strong>+£{sens.toFixed(2)} revenue</strong>
      </div>
    </div>
    <div class="check">
      <div><span class="l">multiply the four ratios</span><strong>{r.clicks.toFixed(2)} × {r.signups.toFixed(2)} × {r.customers.toFixed(2)} × {r.revenue} = £{sens.toFixed(3)}</strong></div>
      <div><span class="l">wiggle test: really add £1 and recompute everything</span><strong>£{wig.toFixed(3)}</strong></div>
    </div>
  </div>
</div>

<style>
  .flow { display: flex; align-items: stretch; gap: 0.3rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .stage { background: var(--code-bg); border-radius: 10px; padding: 0.5rem 0.8rem; min-width: 6.4rem; text-align: center; }
  .nm { color: var(--ink-3); font-size: 0.72rem; }
  .v { font-size: 1.15rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .arrow { display: flex; align-items: center; color: var(--accent-strong); font-weight: 700; }
  .arrow span { background: var(--accent-wash); border-radius: 999px; padding: 0.15rem 0.6rem; font-size: 0.82rem; }
  .sliders { display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); gap: 0.2rem 1.2rem; margin-bottom: 0.8rem; }
  .sl { display: grid; grid-template-columns: 1fr auto; gap: 0.1rem 0.6rem; align-items: center; color: var(--ink-2); font-size: 0.85rem; }
  .sl input { grid-column: 1 / -1; accent-color: var(--accent); }
  .res { background: var(--surface-2); border-radius: 12px; padding: 0.8rem 1rem; }
  .h { margin-bottom: 0.3rem; }
  .casc { font-size: 0.85rem; overflow-x: auto; white-space: nowrap; padding-bottom: 0.2rem; }
  .check { display: grid; gap: 0.3rem; margin-top: 0.6rem; padding-top: 0.6rem; border-top: 1px solid var(--line-strong); font-variant-numeric: tabular-nums; }
  .check div { display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
  .l { color: var(--ink-3); font-size: 0.8rem; }
</style>
