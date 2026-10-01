"""
Trains the UNMODIFIED ../microgpt.py for 3000 steps and exports its weights at several checkpoints, plus reference
outputs computed by the original Python, so the site's JavaScript port can be tested against it.

    python3 tools/train_export.py        (about 10 minutes)

Writes public/data/weights.json
"""
import json, os, shutil, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "public", "data", "weights.json")

src = open(os.path.join(ROOT, "microgpt.py")).read()
assert "num_steps = 500" in src
src = src.replace("num_steps = 500", "num_steps = 3000")

CHECKPOINTS = [0, 50, 200, 1000, 3000]
setup = '''
CKPT = {}
def _save(step):
    CKPT[str(step)] = {k: [[round(p.data, 6) for p in row] for row in mat] for k, mat in state_dict.items()}
_save(0)
'''
assert "# Repeat in sequence" in src
src = src.replace("# Repeat in sequence", setup + "\n# Repeat in sequence")
log_line = '    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")'
assert log_line in src
src = src.replace(log_line, "    if (step + 1) in %r: _save(step + 1)\n" % CHECKPOINTS + log_line)

tail = '''
# reference outputs from the ORIGINAL Python gpt(), final weights
REF = {}
for prefix in ["", "k", "ka", "mar", "anna", "emma", "xqz"]:
    ids = [BOS] + [uchars.index(c) for c in prefix]
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    logits = None
    for pos, t in enumerate(ids):
        logits = gpt(t, pos, keys, values)
    REF[prefix] = [l.data for l in logits]

import json
json.dump({
    "config": {"n_embd": n_embd, "n_head": n_head, "n_layer": n_layer, "block_size": block_size, "vocab_size": vocab_size, "uchars": uchars, "num_params": len(params)},
    "checkpoints": CKPT,
    "ref": REF,
}, open("weights.json", "w"), separators=(",", ":"))
'''
src += tail
work = tempfile.mkdtemp()
shutil.copy(os.path.join(ROOT, "input.txt"), work)
os.chdir(work)
exec(compile(src, "microgpt_export", "exec"), {"__name__": "__main__"})
shutil.copy(os.path.join(work, "weights.json"), OUT)
print("wrote", OUT)
