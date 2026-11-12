"use strict";
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) e.append(c instanceof Node ? c : document.createTextNode(c ?? ""));
  return e;                                   // text only goes through textNodes: API data can never inject markup
};
const store = {
  get(k, d) { try { const v = localStorage.getItem("ea_" + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("ea_" + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const S = { tab: "picks", meta: null, fixtures: [], league: "all", hideRefused: true, timer: null };
const base = () => (store.get("api", "") || (location.protocol.startsWith("http") ? location.origin : "")).replace(/\/$/, "");
const pct = x => (x == null ? "–" : (100 * x).toFixed(0) + "%");
const num = (x, d = 3) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));

async function api(path) {
  const b = base(); if (!b) throw new Error("Set the API address in Settings");
  const tok = store.get("token", "");
  const r = await fetch(b + path, { headers: tok ? { Authorization: "Bearer " + tok } : {} });
  if (r.status === 401) throw new Error("API token missing or wrong (Settings)");
  if (!r.ok) throw new Error("API error " + r.status);
  return r.json();
}
const kelly = (p, o, f, cap) => { const b = o - 1; return b > 0 && p != null ? Math.min(Math.max(((p * o - 1) / b) * f, 0), cap) : 0; };
const eta = iso => {
  const m = (new Date(iso) - (Date.now() + (S.skew || 0))) / 60000;   // synthetic demo runs on its own clock if (m < 0) return "started";
  return m < 90 ? Math.round(m) + " min" : m < 2880 ? (m / 60).toFixed(1) + " h" : Math.round(m / 1440) + " d";
};
const bestEdge = f => { const e = [f.edge_home, f.edge_draw, f.edge_away]; let k = 0; e.forEach((v, i) => { if ((v ?? -9) > (e[k] ?? -9)) k = i; }); return { k, v: e[k] }; };

function bar(f) {
  const seg = (c, p, l) => h("span", { class: c, style: `width:${(p ?? 0) * 100}%` }, p > 0.12 ? `${l} ${pct(p)}` : "");
  return h("div", { class: "bar" }, seg("h", f.p_home, "H"), seg("d", f.p_draw, "D"), seg("a", f.p_away, "A"));
}
function card(f) {
  const be = bestEdge(f), minEdge = store.get("minEdge", 3) / 100;
  const chips = [h("span", { class: "chip" }, f.stage), h("span", { class: "chip " + (f.dqs >= 0.9 ? "" : "warn") }, "DQS " + num(f.dqs, 2))];
  if (f.predictable && be.v != null && be.v >= minEdge) chips.push(h("span", { class: "chip edge" }, `edge ${["H", "D", "A"][be.k]} +${(be.v * 100).toFixed(1)}pp`));
  if (!f.predictable) chips.push(h("span", { class: "chip bad" }, "Refused: " + (f.reason || "low quality")));
  return h("div", { class: "card" + (f.predictable ? "" : " refused"), onclick: () => f.predictable && openSheet(f) },
    h("div", { class: "row mut" }, h("span", {}, f.league), h("span", {}, "KO in " + eta(f.kickoff))),
    h("div", { class: "teams" }, `${f.home}  vs  ${f.away}`),
    f.predictable ? bar(f) : h("div", { class: "mut" }, "No prediction made"),
    h("div", { class: "chips" }, chips));
}
function openSheet(f) {
  const kf = store.get("kelly", 0.15), cap = 0.02, d = $("#sheet");
  const rows = [["Home", "home"], ["Draw", "draw"], ["Away", "away"]].map(([n, k]) => {
    const o = f["odds_" + k], p = f["p_" + k];
    return h("tr", {}, h("td", {}, n), h("td", { class: "n" }, pct(p)), h("td", { class: "n" }, pct(f["mkt_" + k])),
      h("td", { class: "n" }, f["edge_" + k] == null ? "–" : (f["edge_" + k] * 100).toFixed(1) + "pp"), h("td", { class: "n" }, num(o, 2)),
      h("td", { class: "n" }, o ? (kelly(p, o, kf, cap) * 100).toFixed(2) + "%" : "–"));
  });
  d.replaceChildren(h("div", { class: "row" }, h("strong", {}, `${f.home} vs ${f.away}`), h("button", { class: "act", onclick: () => d.close() }, "Close")),
    h("div", { class: "mut" }, `${f.league} · ${new Date(f.kickoff).toLocaleString()}`), bar(f),
    h("table", {}, h("tr", {}, h("th", {}, ""), h("th", { class: "n" }, "Model"), h("th", { class: "n" }, "Market"), h("th", { class: "n" }, "Edge"), h("th", { class: "n" }, "Odds"), h("th", { class: "n" }, "Sim stake")), rows),
    h("p", { class: "mut" }, `Expected goals ${num(f.xg_home, 2)} – ${num(f.xg_away, 2)} · model disagreement ±${(f.ci_width * 100).toFixed(1)}pp · simulated fractional Kelly (${kf}), capped at 2% of bankroll. Not advice.`),
    h("p", { class: "mut" }, `Audit: ${f.fixture_id} · ${f.model_version} · ${f.stage}`));
  d.showModal();
}
const $ = (s, el = document) => el.querySelector(s);
const empty = t => h("div", { class: "empty" }, t);

async function picks() {
  S.fixtures = await api("/v1/fixtures/upcoming");
  const leagues = [...new Set(S.fixtures.map(f => f.league))].sort();
  const sel = h("select", { onchange: e => { S.league = e.target.value; render(); } }, h("option", { value: "all" }, "All leagues"),
    leagues.map(l => h("option", { value: l, selected: l === S.league }, l)));
  const list = S.fixtures.filter(f => (S.league === "all" || f.league === S.league) && (!S.hideRefused || f.predictable));
  const refused = S.fixtures.filter(f => !f.predictable).length;
  return [sel, h("label", {}, h("input", { type: "checkbox", checked: S.hideRefused, style: "width:auto;margin-right:8px", onchange: e => { S.hideRefused = e.target.checked; render(); } }), `Hide refused (${refused})`),
    ...(list.length ? list.map(card) : [empty("No fixtures in the T-24h window yet. Predictions appear 24 hours before kickoff.")])];
}
function spark(series, dates) {
  const names = Object.keys(series), vals = names.flatMap(n => series[n]).filter(v => v != null);
  if (!vals.length) return empty("No scored matches yet. The Brainbox learns after matches finish.");
  const lo = Math.min(...vals), hi = Math.max(...vals), W = 320, H = 110, n = dates.length;
  const colors = ["#3b82f6", "#a78bfa", "#f97316", "#22c55e", "#eab308", "#ef4444", "#06b6d4", "#94a3b8"];
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  names.forEach((nm, i) => {
    const pts = series[nm].map((v, j) => v == null ? null : `${(j / Math.max(n - 1, 1)) * W},${H - ((v - lo) / (hi - lo || 1)) * (H - 8) - 4}`).filter(Boolean).join(" ");
    const pl = document.createElementNS(svg.namespaceURI, "polyline"); pl.setAttribute("points", pts); pl.setAttribute("fill", "none");
    pl.setAttribute("stroke", colors[i % colors.length]); pl.setAttribute("stroke-width", nm === "final" ? 3 : 1.5); svg.append(pl);
  });
  return h("div", {}, svg, h("div", { class: "chips" }, names.map((nm, i) => h("span", { class: "chip", style: `color:${colors[i % colors.length]}` }, nm))));
}
async function brain() {
  const b = await api("/v1/brainbox/status"), w = b.weights || {};
  const out = [h("h3", {}, "Ensemble weights"), ...Object.entries(w).map(([k, v]) => h("div", { class: "w" }, h("div", { class: "l" }, k), h("div", { class: "t" }, h("i", { style: `width:${v * 100}%` })), h("div", { class: "mut" }, pct(v)))),
    h("div", { class: "mut" }, `Market trust ${pct(b.market_weight)} · active config ${b.active_config} · scored window ${b.n_window}${b.ece != null ? " · ECE " + num(b.ece) : " · ECE needs 150+ matches"}`),
    h("h3", {}, "Log loss by week (lower is better)"), spark(b.logloss.series, b.logloss.dates)];
  const sup = (b.guardrail || {}).suppressed || [];
  out.push(h("h3", {}, "Guardrail"), h("div", { class: "card" }, sup.length ? "Suppressed: " + sup.join(", ") : "No league suppressed."));
  out.push(h("h3", {}, "Drift"), b.drift.length ? h("table", {}, h("tr", {}, h("th", {}, "Feature"), h("th", { class: "n" }, "PSI"), h("th", {}, "Status")), b.drift.map(d => h("tr", {}, h("td", {}, d.feature), h("td", { class: "n" }, num(d.psi)), h("td", {}, d.status)))) : empty("Waiting for data"));
  out.push(h("h3", {}, "Champion vs challenger"), h("table", {}, h("tr", {}, h("th", {}, "Model"), h("th", { class: "n" }, "n"), h("th", { class: "n" }, "Δ log loss"), h("th", {}, "Decision")),
    (b.cc || []).map(c => h("tr", {}, h("td", {}, c.model), h("td", { class: "n" }, c.matches_shadowed), h("td", { class: "n" }, num(c.d_logloss, 4)), h("td", {}, c.note || "-")))));
  if ((b.actions || []).length) out.push(h("h3", {}, "Recent Brainbox actions"), ...b.actions.map(a => h("div", { class: "mut" }, "• " + a)));
  return out;
}
async function data() {
  const s = await api("/v1/data/sources");
  return [h("h3", {}, "Data sources"), h("table", {}, h("tr", {}, h("th", {}, "Source"), h("th", { class: "n" }, "Age s"), h("th", { class: "n" }, "TTL s"), h("th", {}, "Status")),
    s.map(r => h("tr", {}, h("td", {}, r.source), h("td", { class: "n" }, r.freshness_s ?? "–"), h("td", { class: "n" }, r.ttl_s), h("td", {}, r.status))))];
}
async function backtest() {
  const [curve, sm] = await Promise.all([api("/v1/backtest/latest"), api("/v1/backtest/summary")]);
  if (!sm || !sm.n_test) return [empty("No backtest stored yet. Run: edgeagent backtest")];
  const rows = ["freq_baseline", "model", "final", "market"].filter(k => sm[k]).map(k => h("tr", {}, h("td", {}, k), h("td", { class: "n" }, num(sm[k].logloss)), h("td", { class: "n" }, sm[k].accuracy == null ? "–" : pct(sm[k].accuracy)), h("td", { class: "n" }, num(sm[k].ece))));
  const out = [h("h3", {}, `Walk-forward test (${sm.n_test} matches)`), h("table", {}, h("tr", {}, h("th", {}, ""), h("th", { class: "n" }, "Log loss"), h("th", { class: "n" }, "Accuracy"), h("th", { class: "n" }, "ECE")), rows),
    h("div", { class: "mut" }, `Uniform guess log loss: ${num(sm.uniform_logloss)}`)];
  const bt = sm.betting || {};
  out.push(h("h3", {}, "Simulated betting"), h("div", { class: "card" }, `${bt.n_bets || 0} bets · ROI/bet ${num(bt.roi_per_bet)} (95% CI ${num((bt.roi_ci95 || [])[0])} to ${num((bt.roi_ci95 || [])[1])}) · max drawdown ${pct(bt.max_drawdown)}`),
    h("div", { class: "mut" }, bt.note || ""));
  if (curve.length) { const v = curve.map(c => c.bankroll), lo = Math.min(...v), hi = Math.max(...v); const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("viewBox", "0 0 320 100");
    const pl = document.createElementNS(svg.namespaceURI, "polyline"); pl.setAttribute("fill", "none"); pl.setAttribute("stroke", "#22c55e"); pl.setAttribute("stroke-width", 2);
    pl.setAttribute("points", v.map((y, i) => `${(i / Math.max(v.length - 1, 1)) * 320},${96 - ((y - lo) / (hi - lo || 1)) * 92}`).join(" ")); svg.append(pl); out.push(h("h3", {}, "Simulated bankroll"), svg); }
  return out;
}
async function settings() {
  const field = (label, key, type = "text", ph = "") => [h("label", {}, label), h("input", { type, placeholder: ph, value: store.get(key, type === "number" ? "" : ""), id: "f_" + key })];
  const t = [h("h3", {}, "Connection"), ...field("API address (blank = this server)", "api", "text", "https://your-server:8000"), ...field("API token (if your server requires one)", "token", "password"),
    h("h3", {}, "Display"), ...field("Highlight edge at least (percentage points)", "minEdge", "number", "3"), ...field("Kelly fraction for simulated stake", "kelly", "number", "0.15"),
    h("button", { class: "act", onclick: () => { for (const k of ["api", "token"]) store.set(k, $("#f_" + k).value.trim());
      for (const k of ["minEdge", "kelly"]) { const v = parseFloat($("#f_" + k).value); if (!Number.isNaN(v)) store.set(k, v); } boot(); } }, "Save and reconnect"),
    h("p", { class: "mut" }, "This app never places bets. Staking figures are simulations. No model can guarantee outcomes."),
    h("p", { class: "mut" }, "Install: browser menu → Add to Home screen (Android Chrome) or Share → Add to Home Screen (iOS Safari).")];
  return t;
}
const TABS = { picks, brain, data, backtest, settings };
async function loadMeta() {
  try { S.meta = await api("/v1/meta"); S.skew = S.meta.synthetic ? new Date(S.meta.now.replace(" ", "T") + "Z") - Date.now() : 0; } catch { /* badge shows offline */ }
}
async function render() {
  const v = $("#view");
  try {
    await loadMeta(); v.replaceChildren(h("div", { class: "empty" }, "Loading…")); v.replaceChildren(...(await TABS[S.tab]())); setBadge(); }
  catch (e) { v.replaceChildren(h("div", { class: "card" }, h("strong", {}, "Can’t reach the engine. "), String(e.message || e)), ...(S.tab === "settings" ? await settings() : [])); setBadge(true); }
}
async function setBadge(off) {
  const b = $("#badge");
  if (off) { b.className = "badge off"; b.textContent = "offline"; return; }
  try { S.meta = await api("/v1/meta"); b.className = "badge " + (S.meta.synthetic ? "synthetic" : "live"); b.textContent = S.meta.synthetic ? "SYNTHETIC DATA" : "live"; } catch { b.className = "badge off"; b.textContent = "offline"; }
}
function boot() {
  clearInterval(S.timer); render();
  if (store.get("auto", true)) S.timer = setInterval(() => { if (!document.hidden && S.tab !== "settings") render(); }, 60000);
}
document.querySelectorAll("nav button").forEach(b => b.addEventListener("click", () => { S.tab = b.dataset.tab; document.querySelectorAll("nav button").forEach(x => x.classList.toggle("on", x === b)); render(); }));
if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost" || location.hostname === "127.0.0.1")) navigator.serviceWorker.register("sw.js").catch(() => {});
boot();
