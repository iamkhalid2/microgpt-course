import { launch, BASE, SHOTS, cont, shot, freshChapter } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import assert from 'node:assert/strict';
mkdirSync(SHOTS, { recursive: true });
const { browser, page, problems } = await launch({ height: 1000 });
await freshChapter(page, 4, 'plain');
const slide = (loc, v) => loc.evaluate((el, v) => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); }, v);

await cont(page); await cont(page);
await page.locator('.opt:has-text("Dial B")').click();                    // predict
await cont(page);
// dial lab: be the optimiser. Set both dials to roughly the best.
await shot(page, 'ch4-dials-start');
await slide(page.locator('#da'), '0.3'); await slide(page.locator('#db'), '0.5');
await slide(page.locator('#da'), '0.18'); await slide(page.locator('#db'), '0.67');
await page.locator('.win').waitFor();
await shot(page, 'ch4-dials-found');
console.log('found:', (await page.locator('.win').innerText()).replace(/\n/g, ' '));
await cont(page); await cont(page);
// hill climber: 50 attempts
await page.locator('button:has-text("Try 25")').click(); await page.locator('button:has-text("Try 25")').click();
await shot(page, 'ch4-climb');
console.log('stats:', (await page.locator('.stats').innerText()).replace(/\n+/g, ' '));
await cont(page); await cont(page);
// plain-English exercise
const cell = page.locator('.widget.cell');
await cell.locator('.po:has-text("Only if it gives a lower loss")').click();
await cell.locator('.recipe .bar button.btn.primary').click();
await cell.locator('.oktxt').waitFor({ timeout: 90000 });
console.log('climb:', (await cell.locator('.oktxt').innerText()).trim());
await cont(page); await cont(page);
await page.locator('button:has-text("Run the hill-climber")').click();
await shot(page, 'ch4-scale');
console.log('scale rows:', (await page.locator('.row .n').allInnerTexts()).join(' '));
await cont(page); await cont(page); await cont(page);
await page.locator('button:has-text("Light up")').click();
console.log('problems:', problems);
assert.equal(problems.length, 0);
await browser.close();
console.log('CH4 OK');
