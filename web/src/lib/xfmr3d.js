// A small, dependency-free 3D renderer for the hero: the real microgpt, drawn as a stack of sheets you can turn.
// Each sheet is the list of 16 numbers (one stripe each) that a letter carries at that stage; the arcs are the real
// attention weights. It uses a plain 2D canvas and a hand-rolled camera, so it costs a few KB instead of a 3D library.

import { LAYERS, blend, sameShape, makeCamera, pointInPolygon } from './xfmr3d-data.js';

const WD = 4.8, DD = 3.4, THICK = 0.12;                 // sheet width (positions) and depth (the 16 numbers)
const TOP = { tokens: 0.8, embed: 1.5, res1: 3.7, mlp: 5.3, res2: 6.9, out: 8.5 };   // height of each sheet's top face
const LABEL_Y = [0.55, 1.5, 2.6, 3.7, 5.3, 6.9, 8.7];     // where each stage's label hangs, by LAYERS order
const CENTER_Y = 5.35;
const TRAY_Z = DD / 2 + 0.72;                           // the input tiles sit on a tray in front of the first sheet
const BASE_YAW = 0.52, BASE_PITCH = 0.58;
const TOUR_MS = 2800;
const N = 16, HID = 64, SUB = 4;                        // numbers per letter, MLP neurons, and neurons per stripe

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const ease = (t) => t * t * (3 - 2 * t);

export class XfmrScene {
  constructor(canvas, { onActive = () => {}, reduced = false } = {}) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.onActive = onActive; this.reduced = reduced;
    this.snap = null; this.vals = {}; this.from = null; this.tw = 1; this.fade = 1;
    this.yawBase = BASE_YAW; this.pitch = BASE_PITCH; this.vel = 0; this.swayAmp = reduced ? 0 : 0.7; this.phase = 0;
    this.hover = -1; this.pinned = -1; this.tour = 0; this.tourAt = 0; this.tourPaused = false;
    this.pulseY = 0.55; this.lastTouch = 0; this.drag = null;
    this.hit = []; this.labelHit = []; this.W = 0; this.H = 0; this.dpr = 1;
    this.running = false; this.visible = true; this.t0 = performance.now(); this.last = this.t0; this.lastActive = -1;
    this.readTheme();
    this.resize();
    this.bind();
  }

  // ── setup ──
  readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    this.c = {
      surface: v('--surface'), plate: v('--surface-2'), ink: v('--ink'), ink2: v('--ink-2'), ink3: v('--ink-3'),
      line: v('--line-strong'), accent: v('--accent'), accentStrong: v('--accent-strong'),
      pos: v('--series-2'), neg: v('--series-1'), hid: v('--series-3'), gold: v('--gold'),
      heads: [v('--series-1'), v('--series-2'), v('--series-3'), v('--series-4')],
    };
    this.ui = v('--font-ui') || 'system-ui, sans-serif';
    this.mono = v('--font-mono') || 'ui-monospace, monospace';
    this.invalidate();
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.W = Math.max(1, Math.round(r.width)); this.H = Math.max(1, Math.round(r.height));
    this.canvas.width = Math.round(this.W * this.dpr); this.canvas.height = Math.round(this.H * this.dpr);
    this.invalidate();
  }

  bind() {
    const cv = this.canvas;
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(cv);
    this.io = new IntersectionObserver((e) => { this.visible = e[0].isIntersecting; if (this.visible) this.kick(); }, { threshold: 0.05 });
    this.io.observe(cv);
    this.mo = new MutationObserver(() => this.readTheme());
    this.mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    this.mq = matchMedia('(prefers-color-scheme: dark)'); this.mqf = () => this.readTheme(); this.mq.addEventListener('change', this.mqf);
    this.onDown = (e) => this.down(e); this.onMove = (e) => this.move(e); this.onUp = (e) => this.up(e); this.onLeave = () => { if (!this.drag) this.setHover(-1); };
    this.onKey = (e) => this.key(e);
    cv.addEventListener('pointerdown', this.onDown); cv.addEventListener('pointermove', this.onMove);
    cv.addEventListener('pointerup', this.onUp); cv.addEventListener('pointercancel', this.onUp); cv.addEventListener('pointerleave', this.onLeave);
    cv.addEventListener('keydown', this.onKey);
    this.onVis = () => { if (!document.hidden) this.kick(); };
    document.addEventListener('visibilitychange', this.onVis);
    this.kick();
  }

  destroy() {
    this.running = false; this.dead = true;
    const cv = this.canvas;
    this.ro.disconnect(); this.io.disconnect(); this.mo.disconnect(); this.mq.removeEventListener('change', this.mqf);
    cv.removeEventListener('pointerdown', this.onDown); cv.removeEventListener('pointermove', this.onMove);
    cv.removeEventListener('pointerup', this.onUp); cv.removeEventListener('pointercancel', this.onUp); cv.removeEventListener('pointerleave', this.onLeave);
    cv.removeEventListener('keydown', this.onKey);
    document.removeEventListener('visibilitychange', this.onVis);
  }

  // ── data ──
  setSnapshot(snap) {
    const old = this.snap;
    if (old && sameShape(old, snap) && !this.reduced) {
      this.from = {}; for (const k in this.vals) this.from[k] = Float32Array.from(this.vals[k]);
      this.tw = 0;
    } else {
      this.from = null; this.tw = 1; this.fade = old ? 0.25 : 0;
      this.vals = {}; blend(snap, snap, 1, this.vals);
    }
    this.snap = snap;
    this.invalidate();
  }

  // ── what is highlighted ──
  get active() { return this.hover >= 0 ? this.hover : this.pinned >= 0 ? this.pinned : this.tour; }
  setHover(i) { if (i !== this.hover) { this.hover = i; this.emit(); this.kick(); } }
  pin(i) { this.pinned = i; this.emit(); this.kick(); }
  pauseTour(p) { this.tourPaused = p; this.tourAt = performance.now(); }
  emit() { const a = this.active, key = a * 10 + (this.pinned >= 0 ? 1 : 0); if (key !== this.lastActive) { this.lastActive = key; this.onActive(a, this.pinned >= 0); } }

  // ── input ──
  local(e) { const r = this.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  pick(x, y) {
    for (let i = this.labelHit.length - 1; i >= 0; i--) { const r = this.labelHit[i]; if (x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3]) return i; }
    for (let i = this.hit.length - 1; i >= 0; i--) if (this.hit[i] && pointInPolygon(x, y, this.hit[i])) return i;
    return -1;
  }
  down(e) {
    if (e.button > 0) return;
    const [x, y] = this.local(e);
    this.drag = { x, y, px: x, py: y, moved: false, id: e.pointerId, touch: e.pointerType === 'touch' };
    this.yawBase += this.swayAmp * Math.sin(this.phase); this.swayAmp = 0; this.vel = 0;
    try { this.canvas.setPointerCapture(e.pointerId); } catch {}
  }
  move(e) {
    const [x, y] = this.local(e);
    const d = this.drag;
    if (d) {
      const dx = x - d.px, dy = y - d.py; d.px = x; d.py = y;
      if (Math.hypot(x - d.x, y - d.y) > 5) d.moved = true;
      if (d.moved) { this.yawBase += dx * 0.009; this.pitch = clamp(this.pitch + dy * 0.005, 0.28, 1.1); this.vel = dx * 0.009; this.lastTouch = performance.now(); }
      this.kick();
    } else if (e.pointerType !== 'touch') this.setHover(this.pick(x, y));
  }
  up(e) {
    const d = this.drag; this.drag = null;
    try { this.canvas.releasePointerCapture(e.pointerId); } catch {}
    if (!d) return;
    this.lastTouch = performance.now();
    if (!d.moved && e.type === 'pointerup') { const i = this.pick(d.x, d.y); this.pin(i >= 0 && i !== this.pinned ? i : -1); this.hover = d.touch ? -1 : this.hover; this.emit(); }
    this.kick();
  }
  key(e) {
    const n = LAYERS.length;
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') { e.preventDefault(); this.pin(((this.pinned < 0 ? this.tour : this.pinned) + 1) % n); }
    else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') { e.preventDefault(); this.pin(((this.pinned < 0 ? this.tour : this.pinned) + n - 1) % n); }
    else if (e.key === 'Escape') this.pin(-1);
  }

  // ── loop ──
  invalidate() { this.dirty = true; this.kick(); }
  kick() {
    if (this.running || this.dead) return;
    this.running = true; this.last = performance.now();
    requestAnimationFrame((t) => this.frame(t));
  }
  animating() {
    return !this.reduced || this.tw < 1 || this.fade < 1 || !!this.drag || Math.abs(this.vel) > 1e-4 || Math.abs(this.pulseY - this.pulseTarget()) > 0.01;
  }
  frame(now) {
    if (this.dead) return;
    const dt = Math.min(0.05, (now - this.last) / 1000); this.last = now;
    this.update(now, dt);
    if (this.visible && !document.hidden) { this.draw(now); this.dirty = false; }
    if ((this.animating() || this.dirty) && this.visible && !document.hidden) requestAnimationFrame((t) => this.frame(t));
    else this.running = false;
  }
  pulseTarget() { const a = this.active; return a === 2 ? 2.6 : LABEL_Y[a]; }
  update(now, dt) {
    if (this.tw < 1) { this.tw = Math.min(1, this.tw + dt / 0.8); blend(this.from, this.snap, ease(this.tw), this.vals); }
    if (this.fade < 1) this.fade = Math.min(1, this.fade + dt / 0.45);
    if (!this.drag) {
      this.yawBase += this.vel; this.vel *= Math.pow(0.04, dt);
      if (!this.reduced) {
        const idle = now - this.lastTouch > 5000;
        this.swayAmp += ((idle ? 0.7 : 0) - this.swayAmp) * Math.min(1, dt * (idle ? 0.6 : 3));
        this.phase += dt * 0.42;
      }
    }
    if (!this.tourPaused && this.pinned < 0 && this.hover < 0 && !this.reduced && now - this.tourAt > TOUR_MS) { this.tourAt = now; this.tour = (this.tour + 1) % LAYERS.length; }
    this.emit();
    this.pulseY += (this.pulseTarget() - this.pulseY) * Math.min(1, dt * 4);
  }

  // ── drawing ──
  draw(now) {
    const { ctx, W, H, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const s = this.snap; if (!s) return;
    const labelCol = W < 360 ? 90 : 102;
    const k = Math.min((H - 16) / 11.5, (W - labelCol - 10) / 7.0);
    const yaw = this.yawBase + this.swayAmp * Math.sin(this.phase);
    const cam = this.cam = makeCamera({ yaw, pitch: this.pitch, cx: (W - labelCol) / 2 + 4, cy: H * 0.43 + 2, k, centerY: CENTER_Y });
    this.k = k;
    const act = this.active, v = this.vals, c = this.c;
    this.hit = new Array(LAYERS.length).fill(null);

    ctx.save(); ctx.globalAlpha = this.fade;
    this.drawTokens(cam, s, act === 0);
    this.drawSheet16(cam, TOP.embed, v.emb, v.scales[0], s, 1, act === 1);
    this.drawAttention(cam, s, v.attn, act === 2);
    this.drawLane(cam, TOP.embed, TOP.res1, now);
    this.drawSheet16(cam, TOP.res1, v.res1, v.scales[1], s, 3, act === 3);
    this.drawSheetHid(cam, TOP.mlp, v.hid, v.scales[2], s, 4, act === 4);
    this.drawLane(cam, TOP.res1, TOP.res2, now);
    this.drawSheet16(cam, TOP.res2, v.res2, v.scales[3], s, 5, act === 5);
    this.drawOutput(cam, s, v.probs, 6, act === 6);
    this.drawPulse(cam, s);
    ctx.restore();
    this.drawLabels(act);
  }

  poly(pts, fill, alpha = 1) {
    const ctx = this.ctx;
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    if (fill) { ctx.globalAlpha = alpha * this.fade; ctx.fillStyle = fill; ctx.fill(); ctx.globalAlpha = this.fade; }
  }
  P(cam, x, y, z) { const p = cam.project(x, y, z); return [p[0], p[1], p[2]]; }

  // A box with its top and the sides that face us (the camera is always above).
  box(cam, x0, x1, y0, y1, z0, z1, top, side, edge, alpha = 1) {
    const ctx = this.ctx, P = (x, y, z) => this.P(cam, x, y, z);
    const b = [P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1), P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)];
    const sides = [[0, 1, 5, 4, 0.2], [1, 2, 6, 5, 0.34], [2, 3, 7, 6, 0.2], [3, 0, 4, 7, 0.34]]
      .map((f) => ({ f, d: (b[f[0]][2] + b[f[1]][2] + b[f[2]][2] + b[f[3]][2]) / 4 })).sort((p, q) => p.d - q.d);
    for (const { f } of sides) {
      const pts = [b[f[0]], b[f[1]], b[f[2]], b[f[3]]];
      this.poly(pts, side, alpha);
      ctx.globalAlpha = f[4] * alpha * this.fade; ctx.fillStyle = '#000'; ctx.fill(); ctx.globalAlpha = this.fade;
    }
    const tp = [b[4], b[5], b[6], b[7]];
    this.poly(tp, top, alpha);
    if (edge) { ctx.strokeStyle = edge.color; ctx.lineWidth = edge.w; ctx.lineJoin = 'round'; ctx.stroke(); }
    return tp;
  }

  text(cam, str, x, y, z, size, color, weight = 500, font = this.ui, align = 'center') {
    const p = cam.project(x, y, z), ctx = this.ctx;
    ctx.font = `${weight} ${size}px ${font}`; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillStyle = color;
    ctx.fillText(str, p[0], p[1]);
  }

  glow(on, color) { const ctx = this.ctx; if (on) { ctx.shadowColor = color; ctx.shadowBlur = 11; } else ctx.shadowBlur = 0; }

  plateEdge(active) { return active ? { color: this.c.accent, w: 2 } : { color: this.c.line, w: 1 }; }

  drawTokens(cam, s, active) {
    const { c } = this, P = s.P, gx = WD / P, w = Math.min(0.34, gx * 0.4), y0 = TOP.tokens - 0.3, z = TRAY_Z;
    this.glow(active, c.accent);
    this.box(cam, -WD / 2 - 0.16, WD / 2 + 0.16, y0 - 0.1, y0, z - 0.52, z + 0.52, c.plate, c.plate, this.plateEdge(active), 0.9);
    this.glow(false);
    for (let i = 0; i < P; i++) {
      const x = -WD / 2 + (i + 0.5) * gx, last = i === P - 1;
      this.box(cam, x - w, x + w, y0, TOP.tokens, z - w, z + w, c.surface, c.plate, last ? { color: c.accent, w: 1.6 } : { color: c.line, w: 1 }, 1);
      this.text(cam, s.tokens[i], x, TOP.tokens + 0.01, z, Math.max(12, this.k * 0.36), last ? c.accentStrong : c.ink, 600, this.mono);
    }
    this.hit[0] = [this.P(cam, -WD / 2 - 0.2, y0, z - 0.6), this.P(cam, WD / 2 + 0.2, y0, z - 0.6), this.P(cam, WD / 2 + 0.2, y0, z + 0.6), this.P(cam, -WD / 2 - 0.2, y0, z + 0.6)].map((p) => [p[0], p[1]]);
  }

  plate(cam, y, active) {
    const { c } = this;
    this.glow(active, c.accent);
    const tp = this.box(cam, -WD / 2 - 0.16, WD / 2 + 0.16, y - THICK, y, -DD / 2 - 0.14, DD / 2 + 0.14, c.plate, c.plate, this.plateEdge(active), 0.94);
    this.glow(false);
    return tp.map((p) => [p[0], p[1]]);
  }

  cellQuad(cam, x0, x1, z0, z1, y) {
    return [this.P(cam, x0, y, z0), this.P(cam, x1, y, z0), this.P(cam, x1, y, z1), this.P(cam, x0, y, z1)];
  }

  lastColumn(cam, y, P, active) {
    const gx = WD / P, x0 = -WD / 2 + (P - 1) * gx, c = this.c, ctx = this.ctx;
    this.poly(this.cellQuad(cam, x0 + gx * 0.04, x0 + gx * 0.96, -DD / 2, DD / 2, y + 0.002), null);
    ctx.strokeStyle = c.accent; ctx.globalAlpha = active ? 1 : 0.7; ctx.lineWidth = active ? 1.8 : 1.3; ctx.stroke(); ctx.globalAlpha = this.fade;
  }

  drawSheet16(cam, y, arr, scale, s, idx, active) {
    const { c } = this, P = s.P, gx = WD / P, cz = DD / N;
    this.hit[idx] = this.plate(cam, y, active);
    for (let i = 0; i < P; i++) {
      const xa = -WD / 2 + i * gx + gx * 0.1, xb = -WD / 2 + (i + 1) * gx - gx * 0.1;
      for (let d = 0; d < N; d++) {
        const z0 = -DD / 2 + d * cz + cz * 0.1, z1 = -DD / 2 + (d + 1) * cz - cz * 0.1;
        const q = this.cellQuad(cam, xa, xb, z0, z1, y + 0.001);
        this.poly(q, c.surface, 0.5);
        const val = arr[i * N + d], a = 0.12 + 0.88 * Math.min(1, Math.abs(val) / (scale || 1));
        this.poly(q, val >= 0 ? c.pos : c.neg, a);
      }
    }
    this.lastColumn(cam, y, P, active);
  }

  drawSheetHid(cam, y, arr, scale, s, idx, active) {
    const { c } = this, P = s.P, gx = WD / P, cz = DD / N, sx = (gx * 0.8) / SUB;
    this.hit[idx] = this.plate(cam, y, active);
    for (let i = 0; i < P; i++) {
      const xs = -WD / 2 + i * gx + gx * 0.1;
      for (let d = 0; d < HID; d++) {
        const sub = d % SUB, row = (d / SUB) | 0;
        const xa = xs + sub * sx + sx * 0.1, xb = xs + (sub + 1) * sx - sx * 0.1;
        const z0 = -DD / 2 + row * cz + cz * 0.1, z1 = -DD / 2 + (row + 1) * cz - cz * 0.1;
        const q = this.cellQuad(cam, xa, xb, z0, z1, y + 0.001);
        this.poly(q, c.surface, 0.45);
        const val = arr[i * HID + d];
        if (val > 1e-5) this.poly(q, c.hid, 0.2 + 0.8 * Math.sqrt(Math.min(1, val / (scale || 1))));
      }
    }
    this.lastColumn(cam, y, P, active);
  }

  drawAttention(cam, s, attn, active) {
    const { c, ctx } = this, P = s.P, gx = WD / P, y0 = TOP.embed + 0.06, u = this.k / 38;
    const zone = [this.P(cam, -WD / 2, y0 + 1.0, -DD / 2), this.P(cam, WD / 2, y0 + 1.0, -DD / 2), this.P(cam, WD / 2, y0 + 1.0, DD / 2), this.P(cam, -WD / 2, y0 + 1.0, DD / 2)];
    this.hit[2] = zone.map((p) => [p[0], p[1]]);
    if (active) {
      this.poly(zone, c.accent, 0.07);
      ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5; ctx.stroke(); ctx.restore();
    }
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (const pass of [0, 1]) {                         // other positions first, then the last one on top
      for (let h = 0; h < s.H; h++) {
        const zc = -DD / 2 + (h + 0.5) * (DD / s.H);
        for (let i = 0; i < P; i++) {
          if ((i === P - 1) !== (pass === 1)) continue;
          const xi = -WD / 2 + (i + 0.5) * gx;
          for (let j = 0; j <= i; j++) {
            const w = attn[(h * P + i) * P + j];
            if (w < 0.035) continue;
            const xj = -WD / 2 + (j + 0.5) * gx, lastQ = i === P - 1;
            ctx.strokeStyle = c.heads[h % 4]; ctx.lineWidth = (0.7 + 5.2 * w) * u * (lastQ ? 1 : 0.7);
            ctx.globalAlpha = this.fade * (lastQ ? 0.4 + 0.6 * Math.sqrt(w) : 0.1 + 0.45 * Math.sqrt(w));
            ctx.beginPath();
            if (j === i) {                                // looking at itself: a short stem
              const a = this.cam.project(xi, y0, zc), b = this.cam.project(xi, y0 + 0.25 + 0.5 * w, zc);
              ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
            } else {
              const apex = 0.3 + 0.27 * (i - j), STEPS = 16;
              for (let t = 0; t <= STEPS; t++) {
                const f = t / STEPS, x = xj + (xi - xj) * f, y = y0 + 4 * apex * f * (1 - f);
                const p = this.cam.project(x, y, zc);
                if (t === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
              }
            }
            ctx.stroke();
          }
        }
      }
    }
    ctx.globalAlpha = this.fade; ctx.lineCap = 'butt';
  }

  // The dashed "skip lane": the stream runs up beside the block and the block's output is added back onto it.
  drawLane(cam, yA, yB, now) {
    const { c, ctx } = this, xo = -WD / 2 - 0.16, xf = -WD / 2 - 0.72, u = this.k / 38;
    const pts = [[xo, yA - THICK / 2, 0], [xf, yA - THICK / 2, 0], [xf, yB - THICK / 2, 0], [xo, yB - THICK / 2, 0]].map((p) => cam.project(p[0], p[1], p[2]).slice());
    ctx.save();
    ctx.strokeStyle = c.gold; ctx.lineWidth = 1.8 * u; ctx.lineJoin = 'round'; ctx.setLineDash([5 * u, 4 * u]);
    ctx.lineDashOffset = this.reduced ? 0 : -((now / 60) % 9) * u;
    ctx.globalAlpha = 0.9 * this.fade;
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = c.surface; ctx.lineWidth = 1.6; ctx.beginPath();
    const e = pts[3], r = 6.5 * u; ctx.arc(e[0] - r * 1.15, e[1], r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.restore();
    ctx.font = `700 ${11 * u}px ${this.ui}`; ctx.fillStyle = c.gold; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('+', e[0] - 6.5 * u * 1.15, e[1] + 0.5);
  }

  drawOutput(cam, s, probs, idx, active) {
    const { c } = this, V = probs.length, gx = WD / V, y = TOP.out;
    this.hit[idx] = this.plate(cam, y, active);
    let top = 0; for (let i = 1; i < V; i++) if (probs[i] > probs[top]) top = i;
    const bw = gx * 0.34, order = [...Array(V).keys()];
    // paint the bars back to front so nearer ones overlap the farther ones correctly
    const depth = (i) => cam.project(-WD / 2 + (i + 0.5) * gx, y, 0)[2];
    order.sort((a, b) => depth(a) - depth(b));
    for (const i of order) {
      const x = -WD / 2 + (i + 0.5) * gx, h = Math.min(2.0, 0.05 + 5 * probs[i]);
      this.box(cam, x - bw, x + bw, y, y + h, -0.2, 0.2, i === top ? c.accent : c.neg, i === top ? c.accent : c.neg, null, i === top ? 1 : 0.78);
    }
    const sorted = [...Array(V).keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 3);
    const label = (i) => (i === V - 1 ? '⏎' : String.fromCharCode(97 + i));
    for (let i = 0; i < V; i++) this.text(cam, label(i), -WD / 2 + (i + 0.5) * gx, y, 0.72, Math.max(7, this.k * 0.2), i === top ? c.accentStrong : c.ink3, i === top ? 700 : 500, this.mono);
    for (const i of sorted) {
      const x = -WD / 2 + (i + 0.5) * gx, h = Math.min(2.0, 0.05 + 5 * probs[i]);
      if (probs[i] < 0.07) continue;
      this.text(cam, `${label(i)} ${Math.round(probs[i] * 100)}%`, x, y + h + 0.3, 0, Math.max(10, this.k * 0.27), c.ink, 700, this.ui);
    }
  }

  drawPulse(cam, s) {
    const { c, ctx } = this, P = s.P, gx = WD / P, x = -WD / 2 + (P - 0.5) * gx, z = TRAY_Z;
    const a = cam.project(x, 0.55, z), b = cam.project(x, TOP.out, z);
    ctx.save();
    ctx.strokeStyle = c.accent; ctx.globalAlpha = 0.28 * this.fade; ctx.lineWidth = 1.2; ctx.setLineDash([2, 4]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.restore();
    const p = cam.project(x, this.pulseY, z);
    ctx.save(); ctx.shadowColor = c.accent; ctx.shadowBlur = 14; ctx.fillStyle = c.accent;
    ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(4, this.k * 0.11), 0, Math.PI * 2); ctx.fill(); ctx.restore();
  }

  drawLabels(act) {
    const { ctx, W, c } = this, cam = this.cam;
    this.labelHit = [];
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    LAYERS.forEach((L, i) => {
      const p = cam.project(0, LABEL_Y[i], 0), on = act === i, size = W < 360 ? 11 : 12;
      ctx.font = `${on ? 700 : 500} ${size}px ${this.ui}`;
      const wd = ctx.measureText(L.short).width, x = W - 8, y = p[1];
      ctx.fillStyle = on ? c.ink : c.ink3; ctx.fillText(L.short, x, y);
      ctx.fillStyle = on ? c.accent : c.line; ctx.beginPath(); ctx.arc(x - wd - 10, y, on ? 4 : 2.5, 0, Math.PI * 2); ctx.fill();
      this.labelHit[i] = [x - wd - 20, y - 12, W, y + 12];
    });
  }
}
