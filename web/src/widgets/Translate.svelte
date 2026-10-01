<script>
  // The same engine, twice: plain cards-and-functions on the left, Python's "class" clothing on the right.
  let pick = $state(0);
  const tabs = [
    { name: 'A card', en: 'Both make a record with four boxes. A class is a template for such records: “Value” is the template, and each Value is one filled-in card.',
      l: "def card(data, children=(), local_grads=()):\n    return {'data': data, 'grad': 0,\n            'children': children,\n            'local_grads': local_grads}",
      r: "class Value:\n    def __init__(self, data, children=(), local_grads=()):\n        self.data = data\n        self.grad = 0\n        self._children = children\n        self._local_grads = local_grads" },
    { name: 'Adding', en: 'On the left you call add(a, b). On the right Python lets you write a + b and calls __add__ for you: “__add__” is simply what + means for a Value.',
      l: "def add(a, b):\n    return card(a['data'] + b['data'],\n                (a, b), (1, 1))",
      r: "def __add__(self, other):\n    other = other if isinstance(other, Value) else Value(other)\n    return Value(self.data + other.data,\n                 (self, other), (1, 1))" },
    { name: 'Multiplying', en: 'Same story for ×. The extra line “other = …” just wraps a plain number such as 3 into a card first, so a * 3 works as well as a * b.',
      l: "def mul(a, b):\n    return card(a['data'] * b['data'],\n                (a, b), (b['data'], a['data']))",
      r: "def __mul__(self, other):\n    other = other if isinstance(other, Value) else Value(other)\n    return Value(self.data * other.data,\n                 (self, other), (other.data, self.data))" },
    { name: 'Using it', en: 'This is the whole reason for the class: arithmetic that reads like arithmetic. The engine underneath is identical.',
      l: "loss = power(add(mul(a, b), card(1.0)), 2)\nbackward(loss)\nprint(a['grad'])",
      r: "loss = (a * b + 1) ** 2\nloss.backward()\nprint(a.grad)" },
  ];
</script>

<div class="widget wide">
  <div class="widget-title">Same engine, two dialects</div>
  <div class="tabs">{#each tabs as t, i}<button class:on={pick === i} onclick={() => (pick = i)}>{t.name}</button>{/each}</div>
  <div class="cols">
    <div><div class="h">Cards and functions (what you wrote)</div><pre class="mono">{tabs[pick].l}</pre></div>
    <div><div class="h">A Python class (what microgpt.py uses)</div><pre class="mono">{tabs[pick].r}</pre></div>
  </div>
  <div class="en">{tabs[pick].en}</div>
</div>

<style>
  .tabs { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.7rem; }
  .tabs button { padding: 0.3rem 0.8rem; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); font-size: 0.8rem; }
  .tabs button.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1rem; }
  @media (max-width: 760px) { .cols { grid-template-columns: minmax(0, 1fr); } }
  .h { color: var(--ink-3); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin-bottom: 0.3rem; }
  pre { margin: 0; background: var(--code-bg); border-radius: 10px; padding: 0.7rem 0.9rem; font-size: 0.78rem; overflow-x: auto; min-height: 7.5rem; }
  .en { margin-top: 0.8rem; padding: 0.6rem 0.8rem; background: var(--accent-wash); border-radius: 10px; font-family: var(--font-body); font-size: 1rem; }
</style>
