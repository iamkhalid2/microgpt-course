import { launch, BASE, SHOTS } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import assert from 'node:assert/strict';
mkdirSync(SHOTS, { recursive: true });
const { browser, page, problems } = await launch();
await page.goto(BASE + '#/ch/0');
await page.evaluate(() => localStorage.clear()); await page.reload();
const cont = () => page.locator('button.btn.primary:has-text("Continue"):not([disabled])').last().click();

await cont();                                               // beat 0 -> replay
await page.locator('text=A real training run').waitFor();
assert.ok(await page.locator('button:has-text("Continue")').last().isDisabled(), 'replay beat should be gated');
await page.locator('button:has-text("Play the training")').click();
await page.waitForTimeout(1300);
await page.screenshot({ path: SHOTS + '/ch0-replay.png', fullPage: false });
await page.locator('input[type=range]').evaluate((el) => { el.value = el.max; el.dispatchEvent(new Event('input', { bubbles: true })); });
await page.waitForTimeout(300);
await page.locator('.widget.wide').first().scrollIntoViewIfNeeded();
await page.screenshot({ path: SHOTS + '/ch0-replay-end.png' });
await cont();                                               // -> explanation
await cont();                                               // -> file peek (gated)
await page.locator('button:has-text("Open the file")').click();
await page.waitForTimeout(500);
await page.screenshot({ path: SHOTS + '/ch0-file.png' });
await page.keyboard.press('Escape');
await cont();                                               // -> the deal
await cont();                                               // -> predict text
await page.locator('textarea').fill('Count which letters tend to follow other letters.');
await page.locator('button:has-text("Lock in my idea")').click();
await page.waitForTimeout(300);
await cont();                                               // -> bank
await page.locator('button:has-text("Light up")').click();
await page.waitForTimeout(900);
await page.screenshot({ path: SHOTS + '/ch0-banked.png' });
const pill = await page.locator('.pill').nth(1).innerText();
console.log('file pill after banking:', pill.replace(/\n/g, ' '));
assert.ok(!pill.includes('0/'), 'lines should be lit after banking');
console.log('problems:', problems);
assert.equal(problems.length, 0);
await browser.close();
console.log('CH0 OK');
