import { launch, BASE, SHOTS, cont, shot, freshChapter } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import assert from 'node:assert/strict';
mkdirSync(SHOTS, { recursive: true });
const { browser, page, problems } = await launch({ height: 1000 });
await freshChapter(page, 17, 'plain');
const setT = (v) => page.locator('input[aria-label="Temperature"]').evaluate((el, v) => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); }, v);

await cont(page);
await page.locator('.opt:has-text("same few names")').click();
await cont(page); await cont(page);
for (const t of ['0.1', '1', '2']) { await setT(t); await page.waitForTimeout(900); console.log('T', t, (await page.locator('.stats').innerText()).replace(/\n+/g, ' ')); if (t === '0.1') await shot(page, 'ch17-cold'); }
await setT('0.5'); await page.waitForTimeout(900);
await shot(page, 'ch17-temp');
console.log('names:', (await page.locator('.names').first().innerText()).replace(/\n/g, ', ').slice(0, 100));
await cont(page); await cont(page);
const c = page.locator('.widget.cell').first();
await c.locator('.po:has-text("Divide every score by the temperature")').click();
await c.locator('.recipe .bar button.btn.primary').click();
await c.locator('.oktxt').waitFor({ timeout: 120000 });
console.log('generate:', (await c.locator('.out').innerText()).trim().split('\n')[0]);
await cont(page); await cont(page); await cont(page);
await page.locator('button:has-text("Light up")').click();
console.log('problems:', problems);
assert.equal(problems.length, 0);
await browser.close();
console.log('CH17 OK');
