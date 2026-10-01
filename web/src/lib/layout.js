// Auto-layout for small computation graphs: columns by depth (longest path from an input), rows spread within a column.
export function layoutGraph(g, { gapX = 150, gapY = 68, w = 112, h = 50, pad = 14 } = {}) {
  const parents = {};
  for (const [f, t] of g.edges) (parents[t] ??= []).push(f);
  const depth = {};
  const d = (id) => (id in depth ? depth[id] : (depth[id] = parents[id]?.length ? 1 + Math.max(...parents[id].map(d)) : 0));
  g.nodes.forEach((n) => d(n.id));
  const layers = [];
  g.nodes.forEach((n) => (layers[depth[n.id]] ??= []).push(n));
  const rows = Math.max(...layers.map((l) => l.length), 1);
  const height = Math.max(rows * gapY + pad * 2, 120);
  const pos = {};
  layers.forEach((layer, x) => {
    layer.forEach((n, i) => { pos[n.id] = [pad + w / 2 + x * gapX, (height / layer.length) * (i + 0.5)]; });
  });
  return { pos, width: pad * 2 + w + (layers.length - 1) * gapX, height, w, h };
}
