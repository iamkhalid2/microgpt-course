// Shared helpers for browser tests. Uses the locally installed Google Chrome (no browser download needed).
import { chromium } from 'playwright-core';
export const CHROME = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
export const BASE = process.env.BASE_URL || 'http://localhost:5173/';
export const SHOTS = process.env.SHOTS || '/tmp/igpt-shots';

export async function launch({ width = 1280, height = 900, dark = false } = {}) {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: dark ? 'dark' : 'light' });
  const page = await ctx.newPage();
  const problems = [];
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') problems.push('console.error: ' + m.text()); });
  return { browser, ctx, page, problems };
}

export const cont = (page) => page.locator('button.btn.primary:has-text("Continue"):not([disabled])').last().click();
export const shot = (page, name) => page.screenshot({ path: `${SHOTS}/${name}.png` });
export async function setCode(page, cell, code) {
  await cell.locator('.cm-content').click();
  await page.keyboard.press('ControlOrMeta+a');
  await page.keyboard.insertText(code);
}
// Run a cell, wait for the check to pass, return its output text.
export async function solve(page, cell, code) {
  await setCode(page, cell, code);
  await cell.locator('button:has-text("Run")').click();
  await cell.locator('.oktxt').waitFor({ timeout: 60000 });
  return (await cell.locator('.out').innerText()).trim();
}
export async function freshChapter(page, n, mode = 'code') {
  await page.goto(BASE + `#/ch/${n}`);
  // Existing tests drive the Python editor, so start them in the Python route (the site's default is Plain English).
  await page.evaluate((m) => { localStorage.clear(); localStorage.setItem('igpt.progress.v1', JSON.stringify({ mode: m })); }, mode);
  await page.reload();
}
