/* Runs Python (Pyodide) off the main thread. Loaded as a classic worker. */
importScripts('https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.js');

let pyodide = null;
let current = null;
const sessions = new Map();

// Keep only the frames from the learner's own code, and renumber so "line 3" means line 3 of their cell.
function tidy(msg) {
  const lines = String(msg).split('\n');
  const first = lines.findIndex((l) => l.includes('File "<exec>"'));
  if (first < 0) return lines.slice(-3).join('\n').trim();
  const body = lines.slice(first).filter((l) => !l.includes('_pyodide/') && !l.includes('pyodide/_base'));
  return ('Traceback (most recent call last):\n' + body.join('\n')).replace(/File "<exec>", /g, '');
}

async function init(dataUrl, weightsUrl) {
  try {
    pyodide = await loadPyodide();
    const text = await (await fetch(dataUrl)).text();
    pyodide.FS.writeFile('input.txt', text); // so open('input.txt') works exactly as in microgpt.py
    try { // the real trained weights, for exercises that use them
      const wtext = await (await fetch(weightsUrl)).text();
      pyodide.FS.writeFile('weights.json', wtext);
    } catch (e) { /* optional */ }
    pyodide.setStdout({ batched: (s) => postMessage({ type: 'out', id: current, text: s + '\n' }) });
    pyodide.setStderr({ batched: (s) => postMessage({ type: 'out', id: current, text: s + '\n', err: true }) });
    postMessage({ type: 'ready' });
  } catch (e) {
    postMessage({ type: 'fatal', error: String(e) });
  }
}

onmessage = async (e) => {
  const m = e.data;
  if (m.cmd === 'init') return init(m.dataUrl, m.weightsUrl);
  if (m.cmd === 'reset') { sessions.delete(m.session); return; }
  if (m.cmd === 'run') {
    current = m.id;
    let ns = sessions.get(m.session);
    if (!ns) { ns = pyodide.globals.get('dict')(); sessions.set(m.session, ns); }
    try {
      await pyodide.runPythonAsync(m.code, { globals: ns });
      postMessage({ type: 'done', id: m.id, ok: true });
    } catch (err) {
      postMessage({ type: 'done', id: m.id, ok: false, error: tidy(err.message) });
    }
  }
};
