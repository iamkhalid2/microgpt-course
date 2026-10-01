// Loads the real trained weights (exported by tools/train_export.py) and exposes the model at each checkpoint.
// `mdl.status` is reactive; the heavy numbers live in the plain object `M` (no proxies, so forward() stays fast).
import { makeModel } from './transformer.js';

export const mdl = $state({ status: 'idle', error: null });
export const M = { cfg: null, models: {}, steps: [], ref: null };
let promise = null;

export function loadModel() {
  if (promise) return promise;
  mdl.status = 'loading';
  promise = fetch(new URL('data/weights.json', document.baseURI))
    .then((r) => { if (!r.ok) throw new Error('could not load the model weights'); return r.json(); })
    .then((d) => {
      const models = {};
      for (const [step, w] of Object.entries(d.checkpoints)) models[step] = makeModel(w, d.config);
      Object.assign(M, { cfg: d.config, models, steps: Object.keys(models).map(Number).sort((a, b) => a - b), ref: d.ref });
      mdl.status = 'ready';
      return M;
    })
    .catch((e) => { mdl.error = String(e); mdl.status = 'error'; throw e; });
  return promise;
}
