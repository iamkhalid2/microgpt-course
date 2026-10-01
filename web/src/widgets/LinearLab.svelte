<script>
  import { getContext } from 'svelte';

  // A linear layer is a stack of scorecards. Each row of weights scores the same inputs a different way.
  const beat = getContext('beat');
  let x = $state([7, 4, 9]);
  let w = $state([[0.5, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.1, 0.7]]);
  const feats = ['years of experience', 'interview score', 'teamwork rating'];
  const roles = ['Engineer', 'Analyst', 'Manager'];
  let edits = 0;
  const touch = () => { if (++edits >= 3) beat?.complete(); };
  const scores = $derived(w.map((row) => row.reduce((s, wi, i) => s + wi * x[i], 0)));
  const best = $derived(scores.indexOf(Math.max(...scores)));
</script>

<div class="widget wide">
  <div class="widget-title">A scorecard for every role · that is a linear layer</div>
  <p class="q">A candidate has three numbers (the <strong>input</strong>). Each role has its own row of <strong>weights</strong>: how much it values each input. A role's score is the dot product of the candidate with its row. Do that for every row and you have a <em>linear layer</em>.</p>
  <div class="tbl">
    <div class="r head"><span></span>{#each feats as f, i}<span>{f}<br /><input type="range" min="0" max="10" step="1" bind:value={x[i]} oninput={touch} aria-label={f} /><b>{x[i]}</b></span>{/each}<span>score</span></div>
    {#each roles as role, r}
      <div class="r" class:best={r === best}>
        <span class="role">{role}</span>
        {#each feats as f, i}<span><input class="wi" type="number" step="0.1" bind:value={w[r][i]} oninput={touch} aria-label="Weight of {f} for {role}" /></span>{/each}
        <span class="sc mono">{scores[r].toFixed(2)}</span>
      </div>
    {/each}
  </div>
  <div class="code mono">scores = [sum(w * x for w, x in zip(row, inputs)) for row in weights]   # ← the same line is `linear` in microgpt.py</div>
  <div class="note">Best fit for this candidate: <strong>{roles[best]}</strong>. Change a weight and the ranking can flip. In a real model nobody sets these weights by hand: they are dials, trained by slopes.</div>
</div>

<style>
  .q { color: var(--ink-2); margin: 0 0 0.7rem; }
  .tbl { display: grid; gap: 0.3rem; }
  .r { display: grid; grid-template-columns: 6rem repeat(3, minmax(0, 1fr)) 5rem; gap: 0.6rem; align-items: center; padding: 0.35rem 0.6rem; border-radius: 10px; background: var(--surface-2); }
  .r.head { background: none; font-size: 0.78rem; color: var(--ink-3); align-items: end; }
  .r > * { min-width: 0; }
  .r.head span { display: flex; flex-direction: column; gap: 0.15rem; }
  .r.head input { accent-color: var(--accent); width: 100%; } .r.head b { color: var(--ink); font-size: 0.95rem; }
  .r.best { background: var(--good-wash); }
  .role { font-weight: 600; } .wi { width: 100%; max-width: 5rem; padding: 0.25rem 0.4rem; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink); }
  .sc { text-align: right; font-weight: 700; }
  .code { margin-top: 0.8rem; font-size: 0.74rem; background: var(--code-bg); padding: 0.45rem 0.7rem; border-radius: 8px; overflow-x: auto; white-space: nowrap; }
  .note { margin-top: 0.7rem; color: var(--ink-2); font-size: 0.85rem; }
  @media (max-width: 640px) { .r { grid-template-columns: 4.2rem repeat(3, minmax(0, 1fr)) 3rem; gap: 0.3rem; padding: 0.35rem 0.4rem; } .r.head span { font-size: 0.66rem; line-height: 1.2; overflow-wrap: anywhere; } .wi { padding: 0.25rem 0.2rem; } }
</style>
