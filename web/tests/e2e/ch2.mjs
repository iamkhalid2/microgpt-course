import { launch, BASE, SHOTS, cont, shot, solve, freshChapter } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import assert from 'node:assert/strict';
mkdirSync(SHOTS, { recursive: true });
const { browser, page, problems } = await launch({ height: 1000 });
await freshChapter(page, 2);
const cells = () => page.locator('.widget.cell');

await page.locator('.opt:has-text("a")').nth(1).click();        // predict: most common letter = 'a'
await cont(page);
await page.locator('button:has-text("Most common first")').click();
await page.locator('button:has-text("Share of all")').click();
await shot(page, 'ch2-bars');
await cont(page); await cont(page);
console.log('count:', await solve(page, cells().nth(0), `counts = {ch: 0 for ch in uchars}\nfor doc in docs:\n    for ch in doc:\n        counts[ch] += 1\nprint(counts['a'], counts['e'], counts['q'])`));
await cont(page);
for (let i = 0; i < 3; i++) { await page.locator('.side button.btn.primary').click(); await page.waitForTimeout(1700); }
await shot(page, 'ch2-wheel');
await cont(page);
await page.locator('.opt:has-text("Better than random")').click();
await cont(page);
await page.locator('button:has-text("more")').first().click();
await shot(page, 'ch2-unigram');
await cont(page);
await cells().nth(1).locator('button:has-text("Run")').click();
await cells().nth(1).locator('.out').waitFor({ timeout: 60000 });
console.log('choices:', (await cells().nth(1).locator('.out').innerText()).trim());
await cont(page); await cont(page);
const rects = page.locator('svg.grid rect');
for (const i of [0 * 27 + 16, 1 * 27 + 4, 5 * 27 + 6]) { await rects.nth(i).click(); }
await rects.nth(3).hover();
await shot(page, 'ch2-grid');
await cont(page); await cont(page);
console.log('table:', await solve(page, cells().nth(2), `V = len(uchars) + 1\ntable = [[0] * V for _ in range(V)]\nfor doc in docs:\n    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]\n    for prev, nxt in zip(tokens, tokens[1:]):\n        table[prev][nxt] += 1\nprint(table[BOS][0])`));
await cont(page);
await page.locator('button:has-text("Pick the next letter")').click(); await page.waitForTimeout(900);
await shot(page, 'ch2-chain');
await page.locator('button:has-text("Write a whole name")').click(); await page.waitForTimeout(6000);
await page.locator('button:text-is("New name")').click();
await page.locator('button:has-text("Write a whole name")').click(); await page.waitForTimeout(6000);
await cont(page);
console.log('make:', await solve(page, cells().nth(3), `import random\ndef make_name():\n    tok = BOS\n    name = ''\n    while True:\n        tok = random.choices(range(V), weights=table[tok])[0]\n        if tok == BOS:\n            break\n        name += uchars[tok]\n    return name\nfor _ in range(6):\n    print(make_name())`));
await cont(page);
for (let i = 0; i < 6; i++) { await page.locator('.opt.mono:not([disabled])').first().click(); await page.locator('button.btn.primary:has-text("Next →")').click(); }
await shot(page, 'ch2-fake');
await cont(page); await cont(page);
const slider = page.locator('input[type=range]').last();
for (const v of ['3', '5', '6', '4']) { await slider.evaluate((el, v) => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); }, v); }
await page.waitForTimeout(500);
await shot(page, 'ch2-wall');
await cont(page); await cont(page);
await page.locator('button:has-text("Light up")').click();
console.log('problems:', problems);
assert.equal(problems.length, 0);
await browser.close();
console.log('CH2 OK');
