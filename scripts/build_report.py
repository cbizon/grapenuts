"""Build the charts page from results/validation.json and
results/layer_pilot.json.

    uv run python scripts/build_report.py OUTPUT.html
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

PAGE = r"""<title>Grapenuts Contact Checks</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>
/* layout: one reading column of findings, each followed by a grid of small charts */
:root {
  --bg: #fcfcfb; --panel: #f4f4f1; --fg: #0b0b0b; --fg2: #52514e; --muted: #84837d;
  --rule: #e2e1dc; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --ref: #52514e;
  --font: Inter, system-ui, sans-serif; --display: "Space Grotesk", Inter, system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #1a1a19; --panel: #232321; --fg: #f0efec; --fg2: #c3c2b7; --muted: #8f8e86;
    --rule: #363531; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --ref: #c3c2b7;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #1a1a19; --panel: #232321; --fg: #f0efec; --fg2: #c3c2b7; --muted: #8f8e86;
  --rule: #363531; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --ref: #c3c2b7;
  color-scheme: dark;
}
body { background: var(--bg); color: var(--fg); font-family: var(--font); font-size: 15px; line-height: 1.5; }
main { max-width: 1080px; margin: 0 auto; padding-block: 32px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-family: var(--display); font-size: 30px; font-weight: 600; margin: 0 0 4px; text-wrap: balance; }
h2 { font-family: var(--display); font-size: 21px; font-weight: 600; margin: 48px 0 8px; text-wrap: balance; }
h3 { font-size: 14px; font-weight: 600; margin: 0; }
p, ul { max-width: 68ch; margin: 8px 0; }
ul { padding-left: 20px; }
li { margin: 4px 0; }
.lede { color: var(--fg2); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 28px 32px; margin-top: 20px; }
figure { margin: 0; min-width: 0; }
figure p { color: var(--fg2); font-size: 13px; margin: 2px 0 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--fg2); margin: 4px 0; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key { width: 16px; height: 0; border-top: 2px solid; display: inline-block; }
.key.dot { width: 8px; height: 8px; border: 0; border-radius: 50%; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-family: var(--font); font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .axis-title { fill: var(--fg2); }
.plot { position: relative; }
.tip { position: absolute; pointer-events: none; background: var(--fg); color: var(--bg); font-size: 12px;
  padding: 5px 8px; border-radius: 4px; white-space: nowrap; transform: translate(-50%, -130%); font-variant-numeric: tabular-nums; }
.tip b { font-weight: 600; }
details { font-size: 12px; color: var(--fg2); margin-top: 4px; }
summary { cursor: pointer; }
summary:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; margin-top: 8px; }
th, td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { font-weight: 600; color: var(--fg2); font-size: 12px; }
details table { font-size: 12px; }
.note { background: var(--panel); padding: 12px 16px; border-radius: 6px; max-width: 68ch; margin-top: 16px; }
.note p { margin: 4px 0; }
</style>
<main>
<h1>Grapenuts contact checks</h1>
<p class="lede">Spring contacts in the event-driven engine, checked against Walton's hard-sphere collision rule, against the spring-dashpot laws they should follow, and in a small 2D layer. Units are sphere diameter D, gravity g and sphere mass. Measured __DATE__ on branch <code>engine-springs</code>.</p>

<div class="note">
__SUMMARY__
</div>

<h2>Against Walton's collision rule</h2>
<p>Two unit spheres collide once, with the line of centres fixed at the moment of touch. The horizontal axis is the incoming tangential surface speed divided by the normal closing speed; the vertical axis is the same ratio after the collision. Negative means the surface velocity reversed. Lines are the hard (Walton) collision in the engine; dots are one spring contact. A spring now delivers Walton's tangential impulse during the collision and uses its tangential spring only if the contact outlasts one.</p>
<div class="grid" id="walton"></div>

<h2>Against the spring-dashpot laws</h2>
<p>A linear spring with a dashpot has a known contact time, peak overlap and restitution. Grey lines are those laws; dots are measured in the engine.</p>
<div class="grid" id="laws"></div>

<h2>A small 2D layer, hard against hybrid</h2>
<p>Width 100 D, three seeds, 20 cycles to settle and 100 measured, sampled at 20 phases per cycle. "tc" makes a contact a spring when either sphere collided within the last contact time or already holds a spring.</p>
<div class="grid" id="layer"></div>
<h3 style="margin-top:28px">Condition the hard engine runs: Γ = 3.5, f* = 0.5, 5 layers</h3>
<div class="scroll" id="table-runs"></div>
<h3 style="margin-top:28px">Condition the hard engine stalls at: Γ = 5, f* = 0.3, 10 layers</h3>
<div class="scroll" id="table-stalls"></div>
</main>
<script>
const V = __VALIDATION__;
const PILOT = __PILOT__;
const EXTRA = __EXTRA__;
const COLORS = ['var(--s1)', 'var(--s2)', 'var(--s3)'];
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function h(tag, cls, parent, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function fmt(x) {
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e4 || a < 1e-3) return x.toExponential(1).replace('e+', 'e').replace('.0e', 'e');
  return String(+x.toPrecision(4));
}
function niceTicks(lo, hi, n) {
  const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n + 0.5);
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}
function logTicks(lo, hi) {
  const out = [];
  for (let p = Math.ceil(Math.log10(lo) - 1e-9); p <= Math.floor(Math.log10(hi) + 1e-9); p++) out.push(Math.pow(10, p));
  return out;
}
// spec: {title, note, xlabel, ylabel, xlog, ylog, xticks, series:[{name, color, mark:'line'|'dots', pts:[[x,y]]}]}
function chart(parent, spec) {
  const fig = h('figure', '', parent);
  h('h3', '', fig, spec.title);
  if (spec.note) h('p', '', fig, spec.note);
  if (spec.series.length > 1) {
    const lg = h('div', 'legend', fig);
    for (const s of spec.series) {
      const item = h('span', '', lg);
      const key = h('i', 'key' + (s.mark === 'dots' ? ' dot' : ''), item);
      if (s.mark === 'dots') key.style.background = s.color; else key.style.borderTopColor = s.color;
      h('span', '', item, s.name);
    }
  }
  const W = 460, H = 300, m = {l: 58, r: 14, t: 10, b: 40};
  const all = spec.series.flatMap(s => s.pts);
  let x0 = Math.min(...all.map(p => p[0])), x1 = Math.max(...all.map(p => p[0]));
  let y0 = Math.min(...all.map(p => p[1])), y1 = Math.max(...all.map(p => p[1]));
  const tx = spec.xlog ? Math.log10 : (v => v), ty = spec.ylog ? Math.log10 : (v => v);
  let xt, yt;
  if (spec.xticks) { xt = spec.xticks.map(t => t[0]); x0 = Math.min(...xt) - 0.5; x1 = Math.max(...xt) + 0.5; }
  else if (spec.xlog) { x0 = Math.pow(10, Math.floor(Math.log10(x0) + 1e-9)); x1 = Math.pow(10, Math.ceil(Math.log10(x1) - 1e-9)); xt = logTicks(x0, x1); }
  else { xt = niceTicks(x0, x1, 6); x0 = Math.min(x0, xt[0]); x1 = Math.max(x1, xt[xt.length - 1]); }
  if (spec.ylog) { y0 = Math.pow(10, Math.floor(Math.log10(y0) + 1e-9)); y1 = Math.pow(10, Math.ceil(Math.log10(y1) - 1e-9)); yt = logTicks(y0, y1); }
  else {
    if (spec.yzero) y0 = Math.min(0, y0);
    const pad = (y1 - y0) * 0.06 || 1; yt = niceTicks(y0 - pad, y1 + pad, 5);
    const step = yt[1] - yt[0];
    while (yt[0] > y0) yt.unshift(+(yt[0] - step).toPrecision(12));
    while (yt[yt.length - 1] < y1) yt.push(+(yt[yt.length - 1] + step).toPrecision(12));
    y0 = yt[0]; y1 = yt[yt.length - 1];
  }
  const X = v => m.l + (tx(v) - tx(x0)) / (tx(x1) - tx(x0)) * (W - m.l - m.r);
  const Y = v => H - m.b - (ty(v) - ty(y0)) / (ty(y1) - ty(y0)) * (H - m.t - m.b);
  const wrap = h('div', 'plot', fig);
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': spec.title}, null);
  wrap.appendChild(svg);
  for (const v of yt) {
    el('line', {x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: 'var(--rule)', 'stroke-width': 1}, svg);
    el('text', {x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end'}, svg, fmt(v));
  }
  if (!spec.ylog && y0 < 0 && y1 > 0) el('line', {x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: 'var(--muted)', 'stroke-width': 1}, svg);
  xt.forEach((v, i) => {
    el('text', {x: X(v), y: H - m.b + 15, 'text-anchor': 'middle'}, svg, spec.xticks ? spec.xticks[i][1] : fmt(v));
  });
  el('text', {x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'axis-title'}, svg, spec.xlabel);
  el('text', {x: 10, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', class: 'axis-title', transform: `rotate(-90 10 ${(m.t + H - m.b) / 2})`}, svg, spec.ylabel);
  const marks = [];
  for (const s of spec.series) {
    if (s.mark === 'line') {
      el('path', {d: s.pts.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(''),
        fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}, svg);
    } else {
      for (const p of s.pts) el('circle', {cx: X(p[0]), cy: Y(p[1]), r: 4, fill: s.color, stroke: 'var(--bg)', 'stroke-width': 1.5}, svg);
    }
    for (const p of s.pts) marks.push({x: X(p[0]), y: Y(p[1]), s, p});
  }
  const ring = el('circle', {r: 7, fill: 'none', stroke: 'var(--fg)', 'stroke-width': 1.5, visibility: 'hidden'}, svg);
  const tip = h('div', 'tip', wrap); tip.hidden = true;
  svg.addEventListener('pointermove', ev => {
    const r = svg.getBoundingClientRect(), k = W / r.width;
    const px = (ev.clientX - r.left) * k, py = (ev.clientY - r.top) * k;
    let best = null, bd = 1e9;
    for (const mk of marks) { const d = (mk.x - px) ** 2 + (mk.y - py) ** 2; if (d < bd) { bd = d; best = mk; } }
    if (!best || bd > 60 * 60) { tip.hidden = true; ring.setAttribute('visibility', 'hidden'); return; }
    ring.setAttribute('cx', best.x); ring.setAttribute('cy', best.y); ring.setAttribute('visibility', 'visible');
    tip.textContent = '';
    h('b', '', tip, fmt(best.p[1]));
    tip.appendChild(document.createTextNode('  at ' + (spec.xticks ? spec.xticks.find(t => t[0] === Math.round(best.p[0]))[1] : fmt(best.p[0])) + '  ·  ' + best.s.name));
    tip.hidden = false;
    tip.style.left = Math.min(Math.max(best.x / k, 90), r.width - 90) + 'px'; tip.style.top = best.y / k + 'px';
  });
  svg.addEventListener('pointerleave', () => { tip.hidden = true; ring.setAttribute('visibility', 'hidden'); });
  const det = h('details', '', fig); h('summary', '', det, 'Data');
  const sc = h('div', 'scroll', det), tb = h('table', '', sc);
  const hr = h('tr', '', tb); h('th', '', hr, 'Series'); h('th', '', hr, spec.xlabel); h('th', '', hr, spec.ylabel);
  for (const s of spec.series) for (const p of s.pts) {
    const tr = h('tr', '', tb); h('td', '', tr, s.name);
    h('td', '', tr, spec.xticks ? spec.xticks.find(t => t[0] === Math.round(p[0]))[1] : fmt(p[0])); h('td', '', tr, fmt(p[1]));
  }
}

// ---- against Walton
const walton = document.getElementById('walton');
const PSI_IN = 'Incoming tangential / normal speed', PSI_OUT = 'Outgoing tangential / normal speed';
{
  const series = [];
  [0.25, 0.5, 1.0].forEach((mu, i) => {
    const rows = V.friction.filter(r => r.mu === mu);
    series.push({name: 'hard, μ = ' + mu, color: COLORS[i], mark: 'line', pts: rows.map(r => [r.psi_in, r.hard])});
    series.push({name: 'spring, μ = ' + mu, color: COLORS[i], mark: 'dots', pts: rows.map(r => [r.psi_in, r.soft])});
  });
  chart(walton, {title: 'Friction coefficient', note: 'Normal closing speed 1, k = 1e8, 50 kicks per contact time. The paper uses μ = 0.5.', xlabel: PSI_IN, ylabel: PSI_OUT, series});
}
{
  const series = [{name: 'hard (β0 = 0.35)', color: 'var(--ref)', mark: 'line', pts: V.kt_ratio_hard.map(r => [r.psi_in, r.hard])}];
  [[0.125, 'spring, kt/kn = 0.125'], [0.2, 'spring, kt/kn = 0.2'], [2 / 7, 'spring, kt/kn = 2/7']].forEach(([ratio, name], i) => {
    series.push({name, color: COLORS[i], mark: 'dots', pts: V.kt_ratio.filter(r => Math.abs(r.kt_ratio - ratio) < 1e-6).map(r => [r.psi_in, r.soft])});
  });
  chart(walton, {title: 'The tangential spring alone', note: 'The earlier law, without the Walton collision phase. μ = 0.5, closing speed 1, k = 1e8.', xlabel: PSI_IN, ylabel: PSI_OUT, series});
}
{
  const series = [];
  [0.1, 1.0, 4.0].forEach((vn, i) => {
    const rows = V.speed_tangential.filter(r => r.vn === vn);
    series.push({name: 'hard, speed ' + vn, color: COLORS[i], mark: 'line', pts: rows.map(r => [r.psi_in, r.hard])});
    series.push({name: 'spring, speed ' + vn, color: COLORS[i], mark: 'dots', pts: rows.map(r => [r.psi_in, r.soft])});
  });
  chart(walton, {title: 'Normal closing speed, at working settings', note: 'μ = 0.5, k = 1e6, 20 kicks per contact time. At high speed and grazing angle the line of centres turns during the contact.', xlabel: PSI_IN, ylabel: PSI_OUT, series});
}
chart(walton, {title: 'Normal restitution against impact speed', note: 'Head-on collisions at k = 1e6. The line is the paper\'s e(v) = 1 − 0.3 v^0.75, held at 0.7 above speed 1.',
  xlabel: 'Normal closing speed', ylabel: 'Restitution', xlog: true,
  series: [{name: 'paper law', color: 'var(--ref)', mark: 'line', pts: V.restitution.map(r => [r.speed, r.law])},
    {name: 'spring, 20 kicks per contact', color: COLORS[0], mark: 'dots', pts: V.restitution.map(r => [r.speed, r.soft_20])},
    {name: 'spring, 50 kicks per contact', color: COLORS[1], mark: 'dots', pts: V.restitution.map(r => [r.speed, r.soft_50])}]});

// ---- against the spring-dashpot laws
const laws = document.getElementById('laws');
{
  const speeds = [0.1, 1.0, 4.0];
  const one = V.contact.filter(r => r.speed === 1.0);
  chart(laws, {title: 'Contact time against stiffness', note: 'Law: π / ω, with ω the damped frequency of the pair. Measured as kicks received times the time step.',
    xlabel: 'Stiffness k', ylabel: 'Contact time', xlog: true, ylog: true,
    series: [{name: 'law (speed 1)', color: 'var(--ref)', mark: 'line', pts: one.map(r => [r.stiffness, r.contact_time_law])}]
      .concat(speeds.map((v, i) => ({name: 'measured, speed ' + v, color: COLORS[i], mark: 'dots', pts: V.contact.filter(r => r.speed === v).map(r => [r.stiffness, r.contact_time])})))});
  const series = [];
  speeds.forEach((v, i) => {
    const rows = V.contact.filter(r => r.speed === v);
    series.push({name: 'law, speed ' + v, color: COLORS[i], mark: 'line', pts: rows.map(r => [r.stiffness, r.max_overlap_law])});
    series.push({name: 'measured, speed ' + v, color: COLORS[i], mark: 'dots', pts: rows.map(r => [r.stiffness, r.max_overlap])});
  });
  chart(laws, {title: 'Peak overlap against stiffness', note: 'Law: the damped spring\'s first maximum, close to v √(m_eff / k). Overlap is in diameters.',
    xlabel: 'Stiffness k', ylabel: 'Peak overlap', xlog: true, ylog: true, series});
}
chart(laws, {title: 'Restitution error against time step', note: 'Largest relative error in restitution over head-on collisions at speeds 0.3, 1 and 3 (k = 1e6).',
  xlabel: 'Kicks per contact time', ylabel: 'Relative error', xlog: true, ylog: true,
  series: [{name: 'restitution error', color: COLORS[0], mark: 'dots', pts: V.steps.map(r => [r.steps, r.max_restitution_error])}]});
chart(laws, {title: 'Layer energy error against time step', note: 'Energy the books cannot account for after 30 cycles of a 100-sphere layer (tc, k = 1e6), as a fraction of throughput. Every value is a loss.',
  xlabel: 'Kicks per contact time', ylabel: 'Unaccounted energy (loss)', xlog: true, ylog: true,
  series: [{name: 'energy loss', color: COLORS[0], mark: 'dots', pts: V.layer_energy.map(r => [r.steps, Math.abs(r.residual)])}]});
chart(laws, {title: 'Sliding to rolling on the plate', note: 'A sphere set sliding at speed 1 on a static plate, μ = 0.5. Law: Coulomb friction until the surface speed matches, then rolling at 5/7.',
  xlabel: 'Time', ylabel: 'Speed', yzero: true,
  series: [{name: 'law, centre', color: COLORS[0], mark: 'line', pts: V.rolling.map(r => [r.t, r.v_law])},
    {name: 'measured, centre', color: COLORS[0], mark: 'dots', pts: V.rolling.filter((r, i) => i % 3 === 0).map(r => [r.t, r.v])},
    {name: 'law, rim (ω r)', color: COLORS[1], mark: 'line', pts: V.rolling.map(r => [r.t, r.surface_law])},
    {name: 'measured, rim (ω r)', color: COLORS[1], mark: 'dots', pts: V.rolling.filter((r, i) => i % 3 === 0).map(r => [r.t, r.surface])}]});

// ---- layer pilot
const METHODS = Object.keys(PILOT.setup.methods);
const SHORT = {'hard': 'hard', 'tc k=1e5': 'tc 1e5', 'tc k=1e6': 'tc 1e6', 'tc k=1e7': 'tc 1e7', 'tc k=1e6, springs below 0.1': 'tc <0.1', 'always k=1e6': 'always'};
const layer = document.getElementById('layer');
function strip(cond, key, title, note, ylabel) {
  const rows = PILOT.runs.filter(r => r.condition === cond && r.finished);
  if (!rows.length) return;
  const used = METHODS.filter(mth => rows.some(r => r.method === mth));
  chart(layer, {title, note, xlabel: 'Contact method', ylabel, xticks: used.map((mth, i) => [i, SHORT[mth] || mth]),
    series: [{name: ylabel, color: COLORS[0], mark: 'dots', pts: rows.map(r => [used.indexOf(r.method), r[key]])}]});
}
strip('runs hard', 'mean_height', 'Mean height where the hard engine runs', 'Γ = 3.5, f* = 0.5, 5 layers. One dot per seed.', 'Mean centre-of-mass height');
strip('runs hard', 'mean_kinetic_per_sphere', 'Kinetic energy where the hard engine runs', 'Same runs. Translational plus rotational, per sphere.', 'Mean kinetic energy per sphere');
strip('stalls hard', 'mean_height', 'Mean height where the hard engine stalls', 'Γ = 5, f* = 0.3, 10 layers. No hard-sphere baseline exists here.', 'Mean centre-of-mass height');
strip('stalls hard', 'mean_kinetic_per_sphere', 'Kinetic energy where the hard engine stalls', 'Same runs.', 'Mean kinetic energy per sphere');

function sweep(key, title, ylabel, cond, note) {
  const rows = PILOT.runs.concat(EXTRA).filter(r => r.condition === cond && r.finished);
  const pick = (kicks) => rows.filter(r => /^tc k=1e\d/.test(r.method) && !/below/.test(r.method) && (/50 kicks/.test(r.method) ? 50 : 20) === kicks)
    .map(r => [Number(r.method.match(/k=(1e\d)/)[1]), r[key]]);
  const series = [{name: '20 kicks per contact', color: COLORS[0], mark: 'dots', pts: pick(20)}];
  if (pick(50).length) series.push({name: '50 kicks per contact', color: COLORS[1], mark: 'dots', pts: pick(50)});
  chart(layer, {title, note, xlabel: 'Stiffness k', ylabel, xlog: true, series});
}
const WAVY = 'Γ = 5, f* = 0.3, 10 layers (a standing wave), tc trigger. One dot per seed.';
const FLAT = 'Γ = 5, f* = 0.6, 10 layers (a flat slab landing every second cycle), tc trigger. One dot per seed.';
sweep('mean_height', 'Mean height against stiffness, wavy condition', 'Mean centre-of-mass height', 'stalls hard', WAVY);
sweep('mean_kinetic_per_sphere', 'Kinetic energy against stiffness, wavy condition', 'Mean kinetic energy per sphere', 'stalls hard', WAVY);
sweep('mean_height', 'Mean height against stiffness, flat condition', 'Mean centre-of-mass height', 'flat', FLAT);
sweep('mean_kinetic_per_sphere', 'Kinetic energy against stiffness, flat condition', 'Mean kinetic energy per sphere', 'flat', FLAT);

function table(id, cond) {
  const host = document.getElementById(id), tb = h('table', '', host);
  const cols = [['Method', null], ['Seeds finished', null], ['Cycles reached', null], ['CPU s per run', 'cpu_seconds'],
    ['Hard collisions / sphere / cycle', 'hard_collisions_per_sphere_cycle'], ['Spring kicks / sphere / cycle', 'kick_forces_per_sphere_cycle'],
    ['Largest overlap', 'max_overlap'], ['Energy residual', 'energy_residual']];
  const hr = h('tr', '', tb); cols.forEach(c => h('th', '', hr, c[0]));
  for (const mth of METHODS) {
    const rows = PILOT.runs.filter(r => r.condition === cond && r.method === mth);
    if (!rows.length) continue;
    const done = rows.filter(r => r.finished), tr = h('tr', '', tb);
    h('td', '', tr, mth); h('td', '', tr, done.length + ' of ' + rows.length);
    h('td', '', tr, rows.map(r => fmt(r.cycles)).join(', '));
    const mean = k => { const v = (k === 'cpu_seconds' ? rows : done).map(r => r[k]); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
    for (const c of cols.slice(3)) {
      let v = mean(c[1]);
      if (c[1] === 'max_overlap' && done.length) v = Math.max(...done.map(r => r[c[1]]));
      if (c[1] === 'energy_residual' && done.length) v = done.map(r => r[c[1]]).reduce((a, b) => Math.abs(b) > Math.abs(a) ? b : a, 0);
      h('td', '', tr, v === null ? '–' : fmt(v));
    }
  }
}
table('table-runs', 'runs hard');
table('table-stalls', 'stalls hard');
</script>
"""


def main():
    out = Path(sys.argv[1])
    validation = json.loads((RESULTS / "validation.json").read_text())
    pilot = json.loads((RESULTS / "layer_pilot.json").read_text())
    extra = []
    for name in ("steps_check.json", "stiff_check.json", "stiff_check_1e9.json"):
        extra += [r for r in json.loads((RESULTS / name).read_text())["runs"] if r["finished"]]
    extra += [r for r in json.loads((RESULTS / "flat_compare.json").read_text())["runs"]
              if r["finished"]]
    summary_path = RESULTS / "report_summary.html"
    summary = summary_path.read_text() if summary_path.exists() else "<p>Summary pending.</p>"
    page = (PAGE.replace("__VALIDATION__", json.dumps(validation, separators=(",", ":")))
            .replace("__PILOT__", json.dumps(pilot, separators=(",", ":")))
            .replace("__EXTRA__", json.dumps(extra, separators=(",", ":")))
            .replace("__SUMMARY__", summary)
            .replace("__DATE__", "13 November 2026"))
    out.write_text(page)
    print("wrote", out, len(page), "bytes")


if __name__ == "__main__":
    main()
