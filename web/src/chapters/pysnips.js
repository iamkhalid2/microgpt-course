// Hidden, idempotent Python prerequisites ("pre") so any cell works on its own, even after a page reload.
export const LOAD_DOCS = `import random, math
if 'docs' not in globals():
    docs = [l.strip() for l in open('input.txt').read().strip().split('\\n') if l.strip()]
`;

export const VOCAB = LOAD_DOCS + `if 'BOS' not in globals() or 'uchars' not in globals():
    uchars = sorted(set(''.join(docs)))
    BOS = len(uchars)
`;

export const BIGRAM = VOCAB + `if 'P' not in globals():
    _V = len(uchars) + 1
    _counts = [[0] * _V for _ in range(_V)]
    for _d in docs:
        _t = [BOS] + [uchars.index(c) for c in _d] + [BOS]
        for _a, _b in zip(_t, _t[1:]):
            _counts[_a][_b] += 1
    def P(prev, nxt):
        """How likely is token nxt right after token prev? (counts / row total)"""
        return _counts[prev][nxt] / sum(_counts[prev])
`;

export const COUNTS = VOCAB + `if 'counts' not in globals():
    counts = {ch: sum(d.count(ch) for d in docs) for ch in uchars}
`;

export const TABLE = VOCAB + `if 'table' not in globals():
    V = len(uchars) + 1
    table = [[0] * V for _ in range(V)]
    for _d in docs:
        _t = [BOS] + [uchars.index(c) for c in _d] + [BOS]
        for _a, _b in zip(_t, _t[1:]):
            table[_a][_b] += 1
`;

export const LOSSFN = BIGRAM + `if 'name_loss' not in globals():
    def name_loss(name):
        tokens = [BOS] + [uchars.index(ch) for ch in name] + [BOS]
        losses = []
        for prev, nxt in zip(tokens, tokens[1:]):
            losses.append(-math.log(P(prev, nxt)))
        return sum(losses) / len(losses)
`;

// Chapter 4: the two-dial vowel model. loss(a, b) is graded on the real names, just like in the page's toys.
export const VC = LOAD_DOCS + `if 'loss' not in globals():
    _n = {'vv': 0, 'vc': 0, 'cv': 0, 'cc': 0}
    for _d in docs:
        for _x, _y in zip(_d, _d[1:]):
            _n[('v' if _x in 'aeiou' else 'c') + ('v' if _y in 'aeiou' else 'c')] += 1
    _N = sum(_n.values())
    def loss(a, b):
        """Average surprise of the two-dial model. a = chance of a vowel after a vowel, b = chance of a vowel after a consonant."""
        if not (0 < a < 1 and 0 < b < 1):
            return float('inf')
        return -(_n['vv'] * math.log(a) + _n['vc'] * math.log(1 - a) + _n['cv'] * math.log(b) + _n['cc'] * math.log(1 - b)) / _N
`;

// Chapter 6: the two backward rules (used by the run-only cell even if the learner skipped the exercise).
export const BACKFNS = `if 'mul_backward' not in globals():
    def add_backward(upstream):
        return upstream, upstream
    def mul_backward(a, b, upstream):
        return b * upstream, a * upstream
`;

// Chapter 7: the card engine. Each piece is defined only if the learner's own version isn't already there,
// so any cell works on its own, and a learner's own (correct) code is never overwritten.
export const ENGINE = `import math, json
if 'card' not in globals():
    def card(data, children=(), local_grads=(), label=''):
        return {'data': data, 'grad': 0, 'children': children, 'local_grads': local_grads, 'label': label}
if 'add' not in globals():
    def add(a, b):
        return card(a['data'] + b['data'], (a, b), (1, 1), '+')
if 'mul' not in globals():
    def mul(a, b):
        return card(a['data'] * b['data'], (a, b), (b['data'], a['data']), '×')
if 'power' not in globals():
    def power(a, k):
        return card(a['data'] ** k, (a,), (k * a['data'] ** (k - 1),), '^' + str(k))
if 'log_' not in globals():
    def log_(a):
        return card(math.log(a['data']), (a,), (1 / a['data'],), 'log')
if 'neg' not in globals():
    def neg(a):
        return card(-a['data'], (a,), (-1,), '−')
if 'sub' not in globals():
    def sub(a, b):
        return add(a, neg(b))
if 'topo_order' not in globals():
    def topo_order(root):
        order, seen = [], set()
        def visit(c):
            if id(c) not in seen:
                seen.add(id(c))
                for parent in c['children']:
                    visit(parent)
                order.append(c)
        visit(root)
        return order
if 'backward' not in globals():
    def backward(root):
        order = topo_order(root)
        root['grad'] = 1
        for c in reversed(order):
            for parent, local in zip(c['children'], c['local_grads']):
                parent['grad'] += local * c['grad']
if 'show' not in globals():
    def show(root):
        """Draw the graph that root was built from. (The page turns this line into a picture.)"""
        order = topo_order(root)
        ids = {id(c): i for i, c in enumerate(order)}
        nodes = [{'id': ids[id(c)], 'label': c.get('label', ''), 'data': round(c['data'], 4), 'grad': round(c['grad'], 4)} for c in order]
        edges = [[ids[id(p)], ids[id(c)]] for c in order for p in c['children']]
        print('@@GRAPH ' + json.dumps({'nodes': nodes, 'edges': edges}))
`;

// ...plus the real two-dial loss, built out of cards, so the engine can train something genuine.
export const ENGINE_VC = LOAD_DOCS + `if '_n' not in globals():
    _n = {'vv': 0, 'vc': 0, 'cv': 0, 'cc': 0}
    for _d in docs:
        for _x, _y in zip(_d, _d[1:]):
            _n[('v' if _x in 'aeiou' else 'c') + ('v' if _y in 'aeiou' else 'c')] += 1
    _N = sum(_n.values())
` + ENGINE + `if 'vowel_loss' not in globals():
    def vowel_loss(a, b):
        """The Chapter 4 loss, built from cards. a and b are cards for the two dials."""
        one = card(1.0)
        t1 = mul(card(-_n['vv'] / _N), log_(a))
        t2 = mul(card(-_n['vc'] / _N), log_(sub(one, a)))
        t3 = mul(card(-_n['cv'] / _N), log_(b))
        t4 = mul(card(-_n['cc'] / _N), log_(sub(one, b)))
        return add(add(t1, t2), add(t3, t4))
`;


// Chapter 8: plain-list softmax, so the error-signal exercise works even if the learner skipped ahead.
export const SOFTMAX = `import math
if 'softmax' not in globals():
    def softmax(logits):
        biggest = max(logits)
        exps = [math.exp(x - biggest) for x in logits]
        total = sum(exps)
        return [e / total for e in exps]
`;


// Chapter 9's linear(), available to later exercises even if the learner skipped ahead.
export const LINEAR = `if 'linear' not in globals():
    def linear(x, w):
        return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]
`;


// Chapter 14: the real trained weights and the small helpers (all built in earlier chapters), so the learner can focus on assembly.
export const GPTPRE = `import json, math
if 'cfg' not in globals():
    _D = json.load(open('weights.json'))
    W = _D['checkpoints']['3000']
    cfg = _D['config']
    n_head = cfg['n_head']
    head_dim = cfg['n_embd'] // n_head
    BOS = len(cfg['uchars'])
if 'linear' not in globals():
    def linear(x, w):
        return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]
if 'softmax' not in globals():
    def softmax(logits):
        biggest = max(logits)
        exps = [math.exp(v - biggest) for v in logits]
        total = sum(exps)
        return [e / total for e in exps]
if 'rmsnorm' not in globals():
    def rmsnorm(x):
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]
`;


// Chapter 15: Adam for ONE dial, available to the race cell even if the learner skipped the exercise.
export const ADAMFN = `if 'adam_step' not in globals():
    def adam_step(p, grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.95, eps=1e-8):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        return p - lr * m_hat / (v_hat ** 0.5 + eps), m, v
`;


// Chapter 14's finished gpt(), so later exercises can run the real model even if the learner skipped the assembly cell.
export const GPTFN = GPTPRE + `if 'gpt' not in globals():
    block_size = cfg['block_size']
    uchars = cfg['uchars']
    def gpt(token_id, pos_id, keys, values):
        x = [t + p for t, p in zip(W['wte'][token_id], W['wpe'][pos_id])]
        x = rmsnorm(x)
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, W['layer0.attn_wq'])
        k = linear(x, W['layer0.attn_wk'])
        v = linear(x, W['layer0.attn_wv'])
        keys.append(k)
        values.append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [ki[hs:hs + head_dim] for ki in keys]
            v_h = [vi[hs:hs + head_dim] for vi in values]
            scores = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim ** 0.5 for t in range(len(k_h))]
            weights = softmax(scores)
            x_attn.extend([sum(weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)])
        x = linear(x_attn, W['layer0.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, W['layer0.mlp_fc1'])
        x = [max(0, v) ** 2 for v in x]
        x = linear(x, W['layer0.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
        return linear(x, W['lm_head'])
`;
