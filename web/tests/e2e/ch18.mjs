import { launch, BASE, SHOTS, cont, shot, freshChapter } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import assert from 'node:assert/strict';
mkdirSync(SHOTS, { recursive: true });
const { browser, page, problems } = await launch({ height: 1000 });
await freshChapter(page, 18, 'plain');

await cont(page);
for (const n of [30, 72, 98, 129, 181, 195]) await page.locator('.fr .ln').nth(n - 1).click();
await shot(page, 'ch18-reader');
console.log('note:', (await page.locator('.note').innerText()).replace(/\n+/g, ' | ').slice(0, 150));
await page.locator('.chip:has-text("Ch 13")').click();
await cont(page);
// the quiz: answer every question with the right line (retry once wrongly first)
const answers = [166, 134, 181, 168, 182, 195, 72, 24, 197, 81];
const q = page.locator('.fr:has(.quiz) .ln');
await q.nth(10).click();    // a wrong guess on purpose
console.log('wrong feedback:', (await page.locator('.fb').innerText()).slice(0, 110));
for (const [i, n] of answers.entries()) {
  await q.nth(n - 1).click();
  await page.locator('.fb.good').waitFor();
  await page.locator('.fb.good button').click();
}
await shot(page, 'ch18-quiz');
await cont(page);
await page.locator('textarea').fill('Attention, because it is not obvious how a position would choose what to listen to.');
await page.locator('button:has-text("Lock in my idea")').click();
await cont(page); await cont(page);
await page.locator('button:has-text("Mark this chapter complete")').click();
console.log('problems:', problems);
assert.equal(problems.length, 0);
await browser.close();
console.log('CH18 OK');
