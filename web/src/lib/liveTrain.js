// Turns the real microgpt.py into a version that reports progress and can be reconfigured, WITHOUT changing its logic:
// it only adds print lines and swaps a few constants. Used by the live-training and mutation-lab widgets.
const LOG_LINE = '    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")';

export const snapCode = (n, k) => `
SNAPS = set(range(0, ${n} + 1, ${k})) | {${n}}
def _snap(tag):
    _st = random.getstate()
    _out = []
    for _ in range(8):
        _k, _v = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        _t = BOS; _s = []
        for _p in range(block_size):
            _lg = gpt(_t, _p, _k, _v)
            _pr = softmax([l / 0.5 for l in _lg])
            _t = random.choices(range(vocab_size), weights=[x.data for x in _pr])[0]
            if _t == BOS: break
            _s.append(uchars[_t])
        _out.append(''.join(_s))
    print('@@S %d %s' % (tag, ','.join(_out)))
    random.setstate(_st)
_snap(0)
`;

// overrides: { n_embd, n_head, n_layer, block_size, lr, temperature }, dataFile: name of a file in the Python filesystem
export function patchSource(src, { steps = 500, overrides = {}, dataFile = null } = {}) {
  let out = src;
  const rep = (from, to) => {
    if (!out.includes(from)) throw new Error(`microgpt.py has changed: could not find "${from}"`);
    out = out.replace(from, to);
  };
  rep('num_steps = 500', `num_steps = ${steps}`);
  if (overrides.n_embd) rep('n_embd = 16', `n_embd = ${overrides.n_embd}`);
  if (overrides.n_head) rep('n_head = 4', `n_head = ${overrides.n_head}`);
  if (overrides.n_layer) rep('n_layer = 1', `n_layer = ${overrides.n_layer}`);
  if (overrides.block_size) rep('block_size = 8', `block_size = ${overrides.block_size}`);
  if (overrides.lr) rep('learning_rate, beta1, beta2, eps_adam = 1e-2,', `learning_rate, beta1, beta2, eps_adam = ${overrides.lr},`);
  if (dataFile) out = out.split("'input.txt'").join(`'${dataFile}'`);
  const k = Math.max(10, Math.round(steps / 100) * 10);
  rep('# Repeat in sequence', snapCode(steps, k) + '\n# Repeat in sequence');
  rep(LOG_LINE, '    print(f"@@L {step+1} {loss.data:.4f}")\n    if (step+1) in SNAPS: _snap(step+1)');
  return out;
}

// Parse one line of the patched program's output.
export function parseLine(line, h) {
  let m;
  if ((m = line.match(/^@@L (\d+) ([\d.]+)/))) h.loss?.(+m[1], +m[2]);
  else if ((m = line.match(/^@@S (\d+) (.*)$/))) h.snap?.(+m[1], m[2].split(','));
  else if ((m = line.match(/^sample\s+\d+: (.*)$/))) h.final?.(m[1]);
  else if (/^(num|vocab)/.test(line)) h.info?.(line.trim());
}

export async function loadSource() {
  const r = await fetch(new URL('data/microgpt.py', document.baseURI));
  if (!r.ok) throw new Error('could not load microgpt.py');
  return r.text();
}
