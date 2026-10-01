import { launch, BASE, SHOTS, cont, shot } from './lib.mjs';
import assert from 'node:assert/strict';
const { browser, page, problems } = await launch({ width: 390, height: 800 });
await page.goto(BASE);
await page.waitForTimeout(1200);
const overflow = async (label) => {
  const o = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth }));
  console.log(label, o.sw <= o.iw ? 'fits' : `OVERFLOW ${o.sw} > ${o.iw}`);
  return o.sw <= o.iw;
};
const ok = [await overflow('home')];
await shot(page, 'm-home');
await page.evaluate(() => localStorage.setItem('igpt.progress.v1', JSON.stringify({ reader: true })));
for (const n of [0, 1, 2, 3]) {
  await page.goto(BASE + `#/ch/${n}`); await page.reload(); await page.waitForTimeout(1500);
  ok.push(await overflow('ch' + n));
  await page.screenshot({ path: `${SHOTS}/m-ch${n}.png`, fullPage: false });
}
await page.goto(BASE + '#/ch/2'); await page.reload(); await page.waitForTimeout(1000);
await page.locator('.widget', { hasText: 'The bigram table' }).first().scrollIntoViewIfNeeded();
await page.waitForTimeout(400);
await shot(page, 'm-grid');
await page.locator('.pill').nth(1).click(); await page.waitForTimeout(400);
await shot(page, 'm-file');
console.log('problems:', problems);
assert.ok(ok.every(Boolean), 'horizontal overflow on mobile');
await browser.close();
console.log('MOBILE OK');
