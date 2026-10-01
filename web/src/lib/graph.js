// A tiny computation graph with reverse-mode slopes: the same idea as the Value class in microgpt.py,
// but built so a page can show it step by step. Pure functions, no DOM, tested against the wiggle test.
//
// A graph is a list of nodes in order. Each node: { id, op, inputs: [ids], value?: number (for inputs/consts) }.
// Ops: 'input' | 'add' | 'mul' | 'pow' (exponent in node.k) | 'log' | 'neg' | 'const'

const local = {
  // value of the node, and the slope of the node with respect to each input (the "local slopes")
  add: (x) => ({ v: x[0] + x[1], d: [1, 1] }),
  mul: (x) => ({ v: x[0] * x[1], d: [x[1], x[0]] }),
  pow: (x, k) => ({ v: x[0] ** k, d: [k * x[0] ** (k - 1)] }),
  log: (x) => ({ v: Math.log(x[0]), d: [1 / x[0]] }),
  neg: (x) => ({ v: -x[0], d: [-1] }),
};

export function forward(nodes, inputs = {}) {
  const val = {}, loc = {};
  for (const n of nodes) {
    if (n.op === 'input') { val[n.id] = inputs[n.id] ?? n.value; continue; }
    if (n.op === 'const') { val[n.id] = n.value; continue; }
    const r = local[n.op](n.inputs.map((i) => val[i]), n.k);
    val[n.id] = r.v; loc[n.id] = r.d;
  }
  return { val, loc };
}

// Backward pass. Returns the slope of the final node's value with respect to every node, plus a step-by-step trace.
export function backward(nodes, inputs = {}) {
  const { val, loc } = forward(nodes, inputs);
  const out = nodes[nodes.length - 1].id;
  const grad = Object.fromEntries(nodes.map((n) => [n.id, 0]));
  grad[out] = 1;
  const trace = [{ node: out, note: 'start', grad: { ...grad } }];
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i];
    if (!loc[n.id]) continue;
    n.inputs.forEach((inp, j) => { grad[inp] += loc[n.id][j] * grad[n.id]; });
    trace.push({ node: n.id, upstream: grad[n.id], local: loc[n.id], inputs: n.inputs, grad: { ...grad } });
  }
  return { val, grad, loc, trace };
}

// The wiggle test, for checking: nudge one input by h and see how the final value moves.
export function wiggle(nodes, inputs, name, h = 1e-6) {
  const base = forward(nodes, inputs).val;
  const out = nodes[nodes.length - 1].id;
  const bumped = forward(nodes, { ...inputs, [name]: (inputs[name] ?? nodes.find((n) => n.id === name).value) + h }).val;
  return (bumped[out] - base[out]) / h;
}

export const PRESETS = {
  chain: {
    title: 'A chain: loss = (a × b + 1)²',
    nodes: [
      { id: 'a', op: 'input', value: 2 }, { id: 'b', op: 'input', value: 3 },
      { id: 'c', op: 'mul', inputs: ['a', 'b'], label: 'c = a × b' },
      { id: 'd', op: 'add', inputs: ['c', 'one'], label: 'd = c + 1' },
      { id: 'loss', op: 'pow', k: 2, inputs: ['d'], label: 'loss = d²' },
    ],
    consts: [{ id: 'one', op: 'const', value: 1 }],
    pos: { a: [60, 70], b: [60, 190], one: [360, 232], c: [200, 130], d: [360, 130], loss: [520, 130] },
    edit: ['a', 'b'],
  },
  twice: {
    title: 'One input used twice: loss = a × a',
    nodes: [
      { id: 'a', op: 'input', value: 3 },
      { id: 'loss', op: 'mul', inputs: ['a', 'a'], label: 'loss = a × a' },
    ],
    consts: [],
    pos: { a: [90, 130], loss: [400, 130] },
    edit: ['a'],
  },
  surprise: {
    title: 'A baby version of our real loss: loss = −log(p × q)',
    nodes: [
      { id: 'p', op: 'input', value: 0.4 }, { id: 'q', op: 'input', value: 0.5 },
      { id: 'm', op: 'mul', inputs: ['p', 'q'], label: 'm = p × q' },
      { id: 'l', op: 'log', inputs: ['m'], label: 'l = log(m)' },
      { id: 'loss', op: 'neg', inputs: ['l'], label: 'loss = −l' },
    ],
    consts: [],
    pos: { p: [60, 70], q: [60, 190], m: [200, 130], l: [360, 130], loss: [520, 130] },
    edit: ['p', 'q'],
  },
};

// Full node list for a preset, with constants first so evaluation order works.
export const nodesOf = (preset) => [...preset.consts, ...preset.nodes];

// Trace of the recursive "visit" used to line cards up so every card comes AFTER the cards it was made from.
// Events: enter (a call starts), skip (already seen), add (all ingredients done, card joins the list), leave.
export function topoTrace(nodes, rootId) {
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const seen = new Set(), order = [], events = [], stack = [];
  function visit(id) {
    if (seen.has(id)) { events.push({ kind: 'skip', id, stack: [...stack], order: [...order] }); return; }
    seen.add(id);
    stack.push(id);
    events.push({ kind: 'enter', id, stack: [...stack], order: [...order] });
    for (const p of byId[id].inputs ?? []) visit(p);
    order.push(id);
    events.push({ kind: 'add', id, stack: [...stack], order: [...order] });
    stack.pop();
  }
  visit(rootId);
  return { events, order };
}
