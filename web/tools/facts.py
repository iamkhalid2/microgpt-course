"""Ground truth for the JS stats library: the same numbers computed independently in Python."""
import json, math, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
docs = [l.strip() for l in open(os.path.join(ROOT, "input.txt")).read().strip().split("\n") if l.strip()]
uchars = sorted(set("".join(docs)))
BOS = len(uchars); V = BOS + 1

def toks(d): return [BOS] + [uchars.index(c) for c in d] + [BOS]
def pairs(ds):
    for d in ds:
        t = toks(d)
        for i in range(len(t) - 1): yield t[i], t[i+1]

def counts1(ds):
    c = [0] * V
    for _, n in pairs(ds): c[n] += 1
    return c
def counts2(ds):
    c = [[0] * V for _ in range(V)]
    for p, n in pairs(ds): c[p][n] += 1
    return c

def loss(ds, probof):
    tot = 0.0; n = 0
    for p, nx in pairs(ds):
        pr = probof(p, nx)
        tot += -math.log(pr) if pr > 0 else math.inf
        n += 1
    return tot / n

train = [d for i, d in enumerate(docs) if i % 10 != 0]
test = [d for i, d in enumerate(docs) if i % 10 == 0]
c1 = counts1(docs); s1 = sum(c1)
c2 = counts2(docs)
r2 = [[x / sum(row) if sum(row) else 0 for x in row] for row in c2]
c1t = counts1(train); c2t = counts2(train)
VOW = "aeiou"
vv = vc = cv = cc = 0
for d in docs:
    for x, y in zip(d, d[1:]):
        a, b = x in VOW, y in VOW
        if a and b: vv += 1
        elif a: vc += 1
        elif b: cv += 1
        else: cc += 1
_N = vv + vc + cv + cc
_a, _b = vv / (vv + vc), cv / (cv + cc)
vc_min = -(vv * math.log(_a) + vc * math.log(1 - _a) + cv * math.log(_b) + cc * math.log(1 - _b)) / _N

out = {
    "vowelCounts": [vv, vc, cv, cc], "vowelBest": [_a, _b], "vowelMinLoss": vc_min,
    "nDocs": len(docs), "uchars": "".join(uchars), "V": V,
    "totalEvents": sum(1 for _ in pairs(docs)),
    "unigramCounts": c1,
    "bigramRow_BOS": c2[BOS],
    "lossUniform": math.log(V),
    "lossUnigram": loss(docs, lambda p, n: c1[n] / s1),
    "lossBigram": loss(docs, lambda p, n: r2[p][n]),
    # raw counts: a never-seen pair means probability 0 => infinite surprise (JSON has no inf, so null)
    "lossBigramHeldOut": (lambda v: None if math.isinf(v) else v)(loss(test, lambda p, n: c2t[p][n] / sum(c2t[p]))),
    "lossBigramHeldOutSmooth1": loss(test, lambda p, n: (c2t[p][n] + 1) / (sum(c2t[p]) + V)),
    "lossUnigramHeldOut": loss(test, lambda p, n: c1t[n] / sum(c1t)),
    "longestName": max(len(d) for d in docs),
    "meanNameLen": sum(len(d) for d in docs) / len(docs),
}
print(json.dumps(out))
