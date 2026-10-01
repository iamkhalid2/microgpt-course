"""
Records a REAL training run of ../microgpt.py so the site can replay it.

Nothing is simulated: we exec the unmodified source with a few extra lines
spliced in that (a) remember every step's loss and (b) sample 16 names at
chosen steps. The output is public/data/replay.json.

    python3 tools/record_replay.py        (takes a minute or two)
"""
import json, os, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "public", "data", "replay.json")

src = open(os.path.join(ROOT, "microgpt.py")).read()

snap_fn = '''
SNAP_STEPS = {0, 1, 5, 10, 25, 50, 100, 200, 350, 500}
SNAPS = {}
LOSSES = []
def _snap(tag):
    _st = random.getstate()
    _out = []
    for _ in range(16):
        _k, _v = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        _t = BOS; _s = []
        for _p in range(block_size):
            _lg = gpt(_t, _p, _k, _v)
            _pr = softmax([l / 0.5 for l in _lg])
            _t = random.choices(range(vocab_size), weights=[p.data for p in _pr])[0]
            if _t == BOS: break
            _s.append(uchars[_t])
        _out.append(''.join(_s))
    SNAPS[tag] = _out
    random.setstate(_st)
_snap(0)
'''
assert "# Repeat in sequence" in src
src = src.replace("# Repeat in sequence", snap_fn + "\n# Repeat in sequence")
log_line = '    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")'
assert log_line in src
src = src.replace(log_line, log_line + "\n    LOSSES.append(loss.data)\n    if (step+1) in SNAP_STEPS: _snap(step+1)")
src += '''
import json
json.dump({"snaps": SNAPS, "losses": LOSSES, "num_params": len(params),
           "vocab_size": vocab_size, "uchars": uchars, "temperature": 0.5},
          open("replay.json", "w"))
'''

work = tempfile.mkdtemp()
shutil.copy(os.path.join(ROOT, "input.txt"), work)
os.chdir(work)
exec(compile(src, "microgpt_recorded", "exec"), {"__name__": "__main__"})
shutil.copy(os.path.join(work, "replay.json"), OUT)
print("wrote", OUT)
