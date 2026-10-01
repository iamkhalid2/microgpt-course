// Copies the dataset from the repo root so the site serves the exact file microgpt.py uses.
import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, '../../input.txt');
const dst = resolve(here, '../public/data/input.txt');
mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
copyFileSync(resolve(here, '../../microgpt.py'), resolve(here, '../public/data/microgpt.py'));
console.log('synced input.txt and microgpt.py');
