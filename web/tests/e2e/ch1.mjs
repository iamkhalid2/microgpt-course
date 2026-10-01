import { launch, BASE, SHOTS } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import assert from 'node:assert/strict';
mkdirSync(SHOTS, { recursive: true });
const { browser, page, problems } = await launch({ height: 1000 });
await page.goto(BASE + '#/ch/1');
await page.evaluate(() => { localStorage.clear(); localStorage.setItem('igpt.progress.v1', JSON.stringify({ mode: 'code' })); }); await page.reload();
const cont = () => page.locator('button.btn.primary:has-text("Continue"):not([disabled])').last().click();
const shot = (n) => page.screenshot({ path: `${SHOTS}/ch1-${n}.png` });

// beat 0: name browser, gated on two shuffles
await page.locator('button:has-text("Show 24 others")').click();
await page.locator('button:has-text("Show 24 others")').click();
await shot('browser');
await cont();
// predict number
await page.locator('input[type=number]').fill('14');
await page.locator('button:has-text("Lock in guess")').click();
await cont();
// fake-or-fact warm-up: 3 answers
for (let i = 0; i < 3; i++) {
  await page.locator('.opt').first().click();
  await page.locator('button:has-text("Next")').click();
}
await shot('fake');
await cont();   // prose
await cont();   // tokenizer
const inp = page.locator('input[maxlength="18"]');
for (const t of ['emm', 'emma', 'anna']) { await inp.fill(t); await inp.dispatchEvent('input'); }
await shot('tokens');
await cont();   // -> how-to-read-code primer (gated: step to the end)
for (let i = 0; i < 6; i++) await page.locator('button:has-text("Next line")').click();
await shot('stepper');
await cont();   // load cell
// run the first python cell (real Pyodide from CDN)
const cell1 = page.locator('.widget.cell').first();
await cell1.locator('button:has-text("Run")').click();
await page.locator('.out:has-text("32033 names")').waitFor({ timeout: 90000 });
console.log('load cell output:', (await cell1.locator('.out').innerText()).split('\n').slice(0, 2).join(' | '));
await shot('cell1');
await cont();   // vocab cell
const setCode = async (cell, code) => {
  await cell.locator('.cm-content').click();
  await page.keyboard.press('ControlOrMeta+a');
  await page.keyboard.insertText(code);
};
const cell2 = page.locator('.widget.cell').nth(1);
await cell2.locator('button:has-text("Run")').click();          // untouched: should fail the check politely
await cell2.locator('.out').waitFor({ timeout: 30000 });
console.log('vocab (untouched):', (await cell2.locator('.out').innerText()).trim().split('\n').pop());
await setCode(cell2, "uchars = sorted(set(''.join(docs)))\nBOS = len(uchars)\nprint(uchars)\nprint('BOS =', BOS)");
await cell2.locator('button:has-text("Run")').click();
await cell2.locator('.oktxt').waitFor({ timeout: 30000 });
console.log('vocab (solved):', (await cell2.locator('.oktxt').innerText()).trim());
await cont();   // encode
const cell3 = page.locator('.widget.cell').nth(2);
await setCode(cell3, "def encode(doc):\n    return [BOS] + [uchars.index(ch) for ch in doc] + [BOS]\n\nprint(encode('emma'))");
await cell3.locator('button:has-text("Run")').click();
await cell3.locator('.oktxt').waitFor({ timeout: 30000 });
console.log('encode:', (await cell3.locator('.oktxt').innerText()).trim());
await cont();   // predict avg length
await page.locator('input[type=number]:not([disabled])').fill('26');
await page.locator('button:has-text("Lock in guess")').last().click();
await cont();   // babble cell
const cell4 = page.locator('.widget.cell').nth(3);
await cell4.locator('button:has-text("Run")').click();
await cell4.locator('.out').waitFor({ timeout: 30000 });
console.log('babble (untouched) ->', (await cell4.locator('.out').innerText()).trim().split('\n').pop());
await setCode(cell4, `import random\n\ndef babble():\n    name = ''\n    while True:\n        t = random.randrange(len(uchars) + 1)\n        if t == BOS:\n            break\n        name += uchars[t]\n    return name\n\nfor _ in range(8):\n    print(babble())\n\nprint('average length:', sum(len(babble()) for _ in range(2000)) / 2000)`);
await cell4.locator('button:has-text("Run")').click();
await cell4.locator('.oktxt').waitFor({ timeout: 30000 });
console.log('babble:', (await cell4.locator('.out').innerText()).trim().split('\n').slice(-3).join(' | '));
await shot('babble');
await cont();
await cont();
await cont();
await page.locator('button:has-text("Light up")').click();
await page.waitForTimeout(800);
await shot('banked');
console.log('problems:', problems);
assert.equal(problems.length, 0);
await browser.close();
console.log('CH1 OK');
