// Python in the browser (Pyodide) running inside a Web Worker, so the page never freezes
// and "Stop" can really stop it. One namespace per chapter; earlier cells feed later cells like a notebook.
class PyRuntime {
  status = $state('idle');      // idle | loading | ready | running
  worker = null;
  readyPromise = null;
  pending = new Map();
  nextId = 1;
  session = 'default';
  setup = '';
  setupDone = new Set();

  #boot() {
    if (this.readyPromise) return this.readyPromise;
    this.status = 'loading';
    this.readyPromise = new Promise((resolve, reject) => {
      const w = new Worker(new URL('py-worker.js', document.baseURI));
      this.worker = w;
      w.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') { this.status = 'ready'; resolve(); }
        else if (m.type === 'fatal') { this.status = 'idle'; reject(new Error(m.error)); }
        else if (m.type === 'out') this.pending.get(m.id)?.onOut?.(m.text, !!m.err);
        else if (m.type === 'done') { const p = this.pending.get(m.id); this.pending.delete(m.id); if (this.pending.size === 0 && this.status === 'running') this.status = 'ready'; p?.resolve(m); }
      };
      w.onerror = (e) => { this.status = 'idle'; reject(new Error(e.message || 'worker failed')); };
      w.postMessage({ cmd: 'init', dataUrl: new URL('data/input.txt', document.baseURI).href, weightsUrl: new URL('data/weights.json', document.baseURI).href });
    });
    return this.readyPromise;
  }

  warm() { this.#boot().catch(() => {}); }

  // Called by a lesson on mount: same id keeps its variables, a new id starts a fresh namespace.
  startSession(id, setup = '') { this.session = id; this.setup = setup; }

  async #exec(code, onOut, session = this.session) {
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.pending.set(id, { resolve, onOut });
      this.worker.postMessage({ cmd: 'run', id, code, session });
    });
  }

  // opts.session: run in a separate namespace (e.g. live training), leaving the chapter's variables untouched.
  async run(code, onOut, opts = {}) {
    try { await this.#boot(); } catch (e) { return { ok: false, error: 'Python could not start: ' + e.message }; }
    this.status = 'running';
    if (opts.session) {
      const r = await this.#exec(code, onOut, opts.session);
      if (this.pending.size === 0) this.status = 'ready';
      return r;
    }
    if (this.setup && !this.setupDone.has(this.session)) {
      const r = await this.#exec(this.setup, () => {});
      if (!r.ok) { this.status = 'ready'; return { ok: false, error: 'Chapter setup failed:\n' + r.error }; }
      this.setupDone.add(this.session);
    }
    const r = await this.#exec(code, onOut);
    if (this.pending.size === 0) this.status = 'ready';
    return r;
  }

  // Free a throwaway namespace (used by the capstone's tests so memory doesn't pile up).
  drop(session) { this.worker?.postMessage({ cmd: 'reset', session }); }

  // Forget this chapter's variables and start from the setup again.
  reset() { this.setupDone.delete(this.session); this.worker?.postMessage({ cmd: 'reset', session: this.session }); }

  // Kill a runaway program. Loses state in every chapter, which is fine: setups re-run on demand.
  stop() {
    this.worker?.terminate();
    for (const p of this.pending.values()) p.resolve({ ok: false, error: 'Stopped.' });
    this.pending.clear();
    this.worker = null; this.readyPromise = null; this.setupDone.clear(); this.status = 'idle';
  }
}
export const py = new PyRuntime();
