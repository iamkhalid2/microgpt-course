// Opens every chapter in reader mode (all scenes visible) at desktop and phone width, in light and dark,
// and checks: no console errors, no horizontal page scroll, no element wider than the viewport. Saves screenshots.
import { launch, BASE, SHOTS } from './lib.mjs';
import { mkdirSync } from 'node:fs';
mkdirSync(SHOTS, { recursive: true });
const only = process.argv.slice(2).map(Number);
const chapters = only.length ? only : [...Array(20).keys()];
let bad = 0;
for (const [label, width, dark] of [['desk', 1280, false], ['phone', 390, false], ['phone-dark', 390, true], ['desk-dark', 1280, true]]) {
  const { browser, page, problems } = await launch({ width, height: 900, dark });
  await page.goto(BASE);
  await page.evaluate(() => localStorage.setItem('igpt.progress.v1', JSON.stringify({ reader: true, mode: 'plain' })));
  for (const n of chapters) {
    problems.length = 0;
    await page.goto(BASE + `#/ch/${n}`); await page.reload();
    await page.locator('.beat').first().waitFor({ timeout: 20000 });
    await page.waitForTimeout(700);
    const over = await page.evaluate(() => {
      const vw = document.documentElement.clientWidth, out = [];
      if (document.documentElement.scrollWidth > vw + 1) out.push(`page scrolls horizontally (${document.documentElement.scrollWidth} > ${vw})`);
      for (const e of document.querySelectorAll('.beat *')) {
        const r = e.getBoundingClientRect();
        if (r.width > 0 && (r.right > vw + 2) && !e.closest('.cm-scroller, pre, .scroll, table, .fr, .tbl, .tv, .mono, .names, svg, .hs, .wscroll, .minimap') && getComputedStyle(e).position !== 'fixed') { out.push(`${e.tagName}.${(e.className?.baseVal ?? e.className)} right=${Math.round(r.right)}`); if (out.length > 4) break; }
      }
      return out;
    });
    const h = await page.evaluate(() => document.documentElement.scrollHeight);
    await page.screenshot({ path: `${SHOTS}/sweep-${label}-ch${n}.png`, fullPage: true });
    const ok = !over.length && !problems.length;
    if (!ok) bad++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${label} ch${n} (${h}px)`, ...(over.length ? [over] : []), ...(problems.length ? [problems.slice(0, 3)] : []));
  }
  await browser.close();
}
console.log(bad ? `${bad} problems` : 'SWEEP OK');
process.exit(bad ? 1 : 0);
