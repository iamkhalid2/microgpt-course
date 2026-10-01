import { launch, BASE, SHOTS, cont, shot, freshChapter } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import assert from 'node:assert/strict';
mkdirSync(SHOTS, { recursive: true });
const { browser, page, problems } = await launch({ height: 1000 });
await freshChapter(page, 7, 'plain');
const cell = (n) => page.locator('.widget.cell').nth(n);
const runPlain = async (c, opts = []) => {
  for (const o of opts) await c.locator(`.po:has-text("${o}")`).click();
  await c.locator('.recipe .bar button.btn.primary').click();
};

await cont(page);                                                                  // 0 -> predict
await page.locator('textarea').fill('Its value, and which cards it came from, and how sensitive it is.');
await page.locator('button:has-text("Lock in my idea")').click();
await cont(page);                                                                  // 1 -> card builder
const sel = page.locator('.form select');
await sel.nth(0).selectOption('mul');                                              // c = a x b  (a, b default)
await page.locator('button:has-text("Make the card")').click();
await sel.nth(0).selectOption('add');                                              // d = c + 1
await sel.nth(2).selectOption('#');
await page.locator('.form input.num').fill('1');
await page.locator('button:has-text("Make the card")').click();
await sel.nth(0).selectOption('pow');                                              // e = d ^ 2
await page.locator('button:has-text("Make the card")').click();
await page.locator('button:has-text("Run backward")').click();
await page.waitForTimeout(300);
await shot(page, 'ch7-cardlab');
console.log('cardlab:', (await page.locator('.xray').first().innerText()).replace(/\s+/g, ' ').slice(0, 200));
await cont(page);                                                                  // 2 -> cards cell
await runPlain(cell(0), ["OTHER one"]);
await cell(0).locator('.oktxt').waitFor({ timeout: 90000 });
console.log('cards:', (await cell(0).locator('.out').innerText()).trim().split('\n')[0]);
await cont(page);                                                                  // 3 -> order puzzle
for (const t of ['a (input 2)', 'b (input 3)', 'the number 1', 'c = a × b', 'd = c + 1', 'loss = d²']) await page.locator(`.pool .chip:has-text("${t}")`).click();
await page.locator('.win').waitFor();
await shot(page, 'ch7-order');
await cont(page);                                                                  // 4 -> topo walk
const nx = page.locator('button:has-text("Next ▶")');
while (await nx.isEnabled()) await nx.click();
await shot(page, 'ch7-topowalk');
await cont(page);                                                                  // 5 -> topo cell
await runPlain(cell(1), ["Add it to the list"]);
await cell(1).locator('.oktxt').waitFor({ timeout: 90000 });
await cont(page);                                                                  // 6 -> backward cell
await runPlain(cell(2), ["Add (local slope × this card's slope) to it"]);
await cell(2).locator('.oktxt').waitFor({ timeout: 90000 });
console.log('backward:', (await cell(2).locator('.oktxt').innerText()).trim());
await cont(page);                                                                  // 7 -> show cell
await cell(3).locator('.recipe .bar button.btn.primary').click();
await cell(3).locator('.xray').waitFor({ timeout: 90000 });
await shot(page, 'ch7-xray');
console.log('show:', (await cell(3).locator('.out').innerText()).trim());
await cont(page); await cont(page);                                                // 8, 9 -> train cell
await runPlain(cell(4), ["Subtract (learning rate × its slope)"]);
await cell(4).locator('.oktxt').waitFor({ timeout: 120000 });
console.log('train:', (await cell(4).locator('.out').innerText()).trim().split('\n').join(' | '));
await cont(page);                                                                  // 10 -> predict reset
await page.locator('.opt:has-text("old slopes would still be")').click();
await cont(page);                                                                  // 11 -> break cell
await cell(5).locator('.recipe .bar button.btn.primary').click();
await cell(5).locator('.errtxt').waitFor({ timeout: 120000 });
console.log('break:', (await cell(5).locator('.out').innerText()).trim().split('\n').slice(-3).join(' | '));
await cont(page); await cont(page);                                                // 12 -> clothes, 13 -> class cell
await shot(page, 'ch7-translate');
await runPlain(cell(6), ["(other.data, self.data)"]);
await cell(6).locator('.oktxt').waitFor({ timeout: 90000 });
console.log('class:', (await cell(6).locator('.out').innerText()).trim().split('\n').join(' | '));
await cont(page); await cont(page);
await page.locator('button:has-text("Light up")').click();
await page.waitForTimeout(500);
console.log('file pill:', (await page.locator('.pill').nth(1).innerText()).replace(/\n/g, ' '));
console.log('problems:', problems);
assert.equal(problems.length, 0);
await browser.close();
console.log('CH7 OK');
