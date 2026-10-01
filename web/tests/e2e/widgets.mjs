// Visual review helper: turns on reader mode (every scene visible) and photographs each widget on its own.
import { launch, BASE, SHOTS } from './lib.mjs';
import { mkdirSync } from 'node:fs';
mkdirSync(SHOTS, { recursive: true });
const dark = process.argv.includes('--dark');
const { browser, page, problems } = await launch({ width: 1100, height: 900, dark });
await page.goto(BASE);
await page.evaluate(() => localStorage.setItem('igpt.progress.v1', JSON.stringify({ reader: true })));
const list = [
  [1, 'The raw material', 'w-names'], [1, 'Tokenizer toy', 'w-tokens'], [1, 'Real or random letters', 'w-fake'],
  [2, 'Letter counts', 'w-letters'], [2, 'The letter wheel', 'w-wheel'], [2, 'The bigram table', 'w-grid'],
  [2, 'Walk the chain', 'w-chain'], [2, 'The wall', 'w-wall'],
  [3, 'How likely did the model', 'w-prob'], [3, 'The surprise machine', 'w-logs'], [3, 'The scoreboard', 'w-score'],
  [4, 'The dial lab', 'w-dials'], [4, 'The random hill-climber', 'w-climb'], [4, 'What happens with more dials', 'w-scale'],
  [5, 'The wiggle test', 'w-wiggle'], [5, 'The compass', 'w-compass'], [5, 'Follow the slope', 'w-descent'],
  [6, 'The funnel', 'w-funnel'], [6, 'The chain rule, running backwards', 'w-graph'],
];
let cur = -1;
for (const [ch, text, name] of list) {
  if (cur !== ch) { await page.goto(BASE + `#/ch/${ch}`); await page.reload(); await page.waitForTimeout(1200); cur = ch; }
  const w = page.locator('.widget', { hasText: text }).first();
  await w.scrollIntoViewIfNeeded();
  await page.waitForTimeout(700);
  if (name === 'w-grid') { await page.locator('svg.grid rect').nth(17 * 27 + 20).click(); await page.waitForTimeout(200); }
  if (name === 'w-chain') { await page.locator('button:has-text("Write a whole name")').click(); await page.waitForTimeout(3500); }
  if (name === 'w-dials') { await page.locator('button:has-text("Show me the map")').click(); await page.waitForTimeout(200); }
  if (name === 'w-climb') { await page.locator('button:has-text("Try 25")').click(); await page.locator('button:has-text("Try 25")').click(); await page.waitForTimeout(200); }
  if (name === 'w-descent') { for (let i = 0; i < 4; i++) await page.locator('button:has-text("Take one step")').click(); }
  if (name === 'w-graph') { const b = page.locator('button:has-text("Next step")'); while (await b.isEnabled()) await b.click(); }
  if (name === 'w-score') { await page.locator('.grp button:has-text("never counted")').click(); await page.waitForTimeout(300); }
  await w.screenshot({ path: `${SHOTS}/${name}${dark ? '-dark' : ''}.png` });
}
console.log('problems:', problems);
await browser.close();
