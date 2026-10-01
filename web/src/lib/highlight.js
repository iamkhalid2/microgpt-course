// Very small Python highlighter for the read-only File panel (the editable cells use CodeMirror).
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const KW = new Set(['import', 'from', 'def', 'class', 'return', 'for', 'in', 'if', 'else', 'elif', 'while', 'not', 'and', 'or', 'is', 'lambda', 'break', 'continue', 'None', 'True', 'False', 'isinstance', 'with', 'as']);

export function highlight(line) {
  const hash = (() => { let q = null; for (let i = 0; i < line.length; i++) { const c = line[i]; if (q) { if (c === q) q = null; } else if (c === '"' || c === "'") q = c; else if (c === '#') return i; } return -1; })();
  const code = hash >= 0 ? line.slice(0, hash) : line;
  const comment = hash >= 0 ? line.slice(hash) : '';
  const html = esc(code).replace(/("[^"]*"|'[^']*')|\b(\d+\.?\d*(?:e-?\d+)?)\b|\b([A-Za-z_]\w*)\b/g, (m, str, num, word) => {
    if (str) return `<i class="s">${str}</i>`;
    if (num) return `<i class="n">${num}</i>`;
    if (word && KW.has(word)) return `<i class="k">${word}</i>`;
    return m;
  });
  return html + (comment ? `<i class="c">${esc(comment)}</i>` : '');
}
