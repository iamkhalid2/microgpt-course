// Runs every browser test in order and reports which passed. Needs the dev server (npm run dev) and Google Chrome.
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const here = dirname(fileURLToPath(import.meta.url));
const num = (f) => parseInt(f.match(/^ch(\d+)/)?.[1] ?? '999');
const files = readdirSync(here).filter((f) => f.endsWith('.mjs') && !['lib.mjs', 'all.mjs', 'widgets.mjs'].includes(f)).sort((a, b) => num(a) - num(b) || a.localeCompare(b));
const results = [];
for (const f of files) {
  const t = Date.now();
  const r = spawnSync('node', [`${here}/${f}`], { encoding: 'utf8', timeout: 900000 });
  const ok = r.status === 0;
  results.push([f, ok, ((Date.now() - t) / 1000).toFixed(0) + 's']);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${f}  ${results.at(-1)[2]}`);
  if (!ok) console.log((r.stdout + r.stderr).split('\n').slice(-12).join('\n'));
}
process.exit(results.every((r) => r[1]) ? 0 : 1);
