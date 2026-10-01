"""
Runs microgpt.py for 3000 steps (instead of 500) and records the mean loss per 250-step window.
Takes ~10 minutes. Output: public/data/long_run.json

    python3 tools/record_long_run.py
"""
import json, os, shutil, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "public", "data", "long_run.json")

src = open(os.path.join(ROOT, "microgpt.py")).read()
assert "num_steps = 500" in src
src = src.replace("num_steps = 500", "num_steps = 3000")
log_line = '    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")'
assert log_line in src
src = src.replace(log_line, "    LOSSES.append(loss.data)")
src = src.replace("# Repeat in sequence", "LOSSES = []\n# Repeat in sequence")
src += '''
import json
W = 250
json.dump({"note": "Mean training loss over each window of 250 steps, from one unmodified run of microgpt.py with num_steps=3000 (seed 42). Every step studies a name it has not seen before, so this doubles as an honest held-out estimate.",
           "window": W,
           "steps": list(range(W, 3001, W)),
           "meanLoss": [round(sum(LOSSES[i - W:i]) / W, 4) for i in range(W, 3001, W)]},
          open("long_run.json", "w"), indent=2)
'''
work = tempfile.mkdtemp()
shutil.copy(os.path.join(ROOT, "input.txt"), work)
os.chdir(work)
exec(compile(src, "microgpt_long", "exec"), {"__name__": "__main__"})
shutil.copy(os.path.join(work, "long_run.json"), OUT)
print("wrote", OUT)
