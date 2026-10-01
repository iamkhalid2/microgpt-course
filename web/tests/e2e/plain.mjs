// The Plain English route: pick the right step, run the recipe, and the same hidden check passes.
import { launch, BASE, SHOTS, cont, shot, freshChapter } from './lib.mjs';
import assert from 'node:assert/strict';
const { browser, page, problems } = await launch({ height: 1000 });
await freshChapter(page, 1, 'plain');
await page.evaluate(() => localStorage.setItem('igpt.progress.v1', JSON.stringify({ mode: 'plain', beat: { ch1: 6 } })));
await page.reload();
const cells = () => page.locator('.widget.cell');

// load cell in plain mode: no blanks, run immediately
const load = cells().nth(0);
await load.locator('.recipe').waitFor();
assert.equal(await load.locator('.cm-editor').isVisible(), false, 'editor should be hidden in the plain route');
await load.locator('button:has-text("Run this recipe")').click();
await load.locator('.out:has-text("32033 names")').waitFor({ timeout: 90000 });
await cont(page);

// vocab: two blanks. First try a wrong answer, expect an explanation.
const vocab = cells().nth(1);
const run = vocab.locator('.recipe .bar button.btn.primary');
assert.ok(await run.isDisabled(), 'run should be locked until the blanks are answered');
await vocab.locator('.po:has-text("just the first name")').click();
assert.ok(await vocab.locator('.why').first().isVisible(), 'wrong answer should explain itself');
await shot(page, 'plain-wrong');
await vocab.locator('.po:has-text("Glue all the names")').click();
await vocab.locator('.po:has-text("next free number")').click();
await shot(page, 'plain-solved');
await run.click();
await vocab.locator('.oktxt').waitFor({ timeout: 60000 });
console.log('plain vocab:', (await vocab.locator('.oktxt').innerText()).trim());
await cont(page);

// Python-route reading panel, and that switching routes keeps the exercise state
const enc = cells().nth(2);
await enc.locator('.mode button:has-text("Python")').click();
await enc.locator('details.read summary').click();
assert.ok(await enc.locator('details.read').isVisible());
await shot(page, 'plain-read-panel');
await enc.locator('.mode button:has-text("Plain English")').click();
await enc.locator('.po:has-text("start symbol, then the number")').click();
await enc.locator('.recipe .bar button.btn.primary').click();
await enc.locator('.oktxt').waitFor({ timeout: 60000 });
console.log('plain encode:', (await enc.locator('.oktxt').innerText()).trim());
console.log('problems:', problems);
assert.equal(problems.length, 0);
await browser.close();
console.log('PLAIN OK');
