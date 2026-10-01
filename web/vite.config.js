import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  base: './',
  plugins: [svelte({
    // Props like `id` and `chapter` are fixed for a component's whole life by design, so this warning is just noise here.
    onwarn: (w, handler) => { if (w.code === 'state_referenced_locally') return; handler(w); },
  })],
  server: { fs: { allow: ['..', '../..'] } }, // microgpt.py lives one folder up
  build: { target: 'es2022', chunkSizeWarningLimit: 900 },
});
