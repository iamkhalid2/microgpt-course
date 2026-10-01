import { launch, SHOTS, cont, shot, freshChapter } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import assert from 'node:assert/strict';
mkdirSync(SHOTS, { recursive: true });
const { browser, page, problems } = await launch({ height: 1100 });
await freshChapter(page, 19, 'plain');
await cont(page);
await page.locator('input[type=number]:not([disabled])').fill('180');
await page.locator('button:has-text("Lock in guess")').click();
await cont(page);

const ms = page.locator('.cap .m');
// 1. an empty file fails with a readable message
await ms.nth(1).locator('button:has-text("Check it")').click();
await page.locator('.cap .m.fail').waitFor({ timeout: 60000 });
const failMsg = await ms.nth(1).locator('.msg').innerText();
console.log('empty-file feedback:', failMsg.slice(0, 120));
assert.ok(/name|defined|Value/i.test(failMsg));
// 2. add each milestone's real code through the hint ladder, and check it
for (let i = 0; i < 5; i++) {
  const m = ms.nth(i);
  await m.locator('button:has-text("Hints")').click();
  if (i === 0) { await m.locator('.lv button').nth(0).click(); await shot(page, 'ch19-hint1'); await m.locator('.lv button').nth(1).click(); }
  await m.locator('.lv button').nth(2).click();
  await m.locator('button:has-text("Add to my file")').click();
  await m.locator('button:has-text("Check it")').click();
  await page.waitForFunction((n) => { const e = document.querySelectorAll('.cap .m')[n]; return e.classList.contains('pass') || e.classList.contains('fail'); }, i, { timeout: 180000 });
  const txt = await m.locator('.msg').innerText();
  console.log(`milestone ${i + 1}:`, txt.slice(0, 140).replace(/\n/g, ' '));
  assert.ok(await m.evaluate((e) => e.classList.contains('pass')), `milestone ${i + 1} should pass: ${txt}`);
}
await shot(page, 'ch19-capstone');
assert.ok(await page.locator('.win').isVisible());
// 3. run the whole file (shorten to 20 steps by editing num_steps)
await page.locator('.cap .cm-content').click();
await page.keyboard.press('ControlOrMeta+End');
await page.keyboard.insertText('\n');
await page.locator('button:has-text("Run my whole file")').click();
await page.locator('.cap .out:has-text("sample")').waitFor({ timeout: 300000 });
console.log('whole file tail:', (await page.locator('.cap .out').innerText()).trim().split('\n').slice(-2).join(' | '));
await shot(page, 'ch19-run');
// 4. the mutation lab: tiny custom run
await cont(page);
await page.locator('.cap').scrollIntoViewIfNeeded();
console.log('lab present:', await page.locator('.widget-title:has-text("mutation lab")').count());
const lab = page.locator('.widget:has(.widget-title:has-text("mutation lab"))');
await lab.scrollIntoViewIfNeeded();
await lab.locator('select').nth(0).selectOption('dino');
await lab.locator('select').nth(3).selectOption('100');
await lab.locator('button:has-text("Train this model")').click();
await lab.locator('.tbl .r:not(.head)').first().waitFor({ timeout: 240000 });
console.log('lab run:', (await lab.locator('.tbl .r:not(.head)').first().innerText()).replace(/\n+/g, ' | ').slice(0, 200));
// own data
await lab.locator('select').nth(0).selectOption('own');
await lab.locator('textarea').fill('ab\n'.repeat(5));
await lab.locator('button:has-text("Train this model")').click();
console.log('too-short feedback:', await lab.locator('.bad').innerText());
await shot(page, 'ch19-lab');
console.log('problems:', problems);
assert.equal(problems.length, 0);
await browser.close();
console.log('CH19 OK');
