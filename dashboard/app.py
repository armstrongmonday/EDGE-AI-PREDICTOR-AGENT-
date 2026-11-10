import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
import api_client as api
from utils import kelly_stake

st.set_page_config(page_title="EdgeAgent v2", page_icon="🧠", layout="wide")

@st.cache_data(ttl=30)
def load():
    return api.get_meta(), api.get_fixtures(), api.get_brainbox(), api.get_sources(), api.get_backtest()

meta, (fx, fx_demo), (bb, bb_demo), (src, _), (bt, bt_summary, bt_demo) = load()

st.title("🧠 EdgeAgent v2")
st.caption("Calibrated probabilities, not guarantees. Research and education only.")
if fx_demo:
    st.warning("Backend not reachable: showing SYNTHETIC demo data. Nothing here is a real prediction.")
elif meta.get("synthetic"):
    st.info("Connected to the backend's SYNTHETIC league simulator (no real matches). Set EDGEAGENT_SOURCE=live with API keys for real data.")

with st.sidebar:
    st.header("Filters")
    leagues = st.multiselect("League", sorted(fx.league.unique()), default=sorted(fx.league.unique()))
    min_dqs = st.slider("Min data-quality score", 0.5, 1.0, 0.8, 0.01)
    min_edge = st.slider("Min edge (pp)", 0.0, 10.0, 3.0, 0.5) / 100
    kf = st.slider("Kelly fraction (simulation)", 0.05, 0.5, 0.15, 0.05)
    if st.button("Refresh now"):
        st.cache_data.clear(); st.rerun()
    st.info("**Notes**\n- 1X2 accuracy of ~50-58% is realistic.\n- Aviator cannot be predicted.\n- Simulated staking only.")

pool = fx[fx.league.isin(leagues)]
refused = pool[~pool.predictable | (pool.dqs < min_dqs)]
view = pool[pool.predictable & (pool.dqs >= min_dqs)].copy()
tabs = st.tabs(["Upcoming", "Match detail", "Brainbox", "Data quality", "Backtest", "Aviator Lab"])

# ---- Upcoming
with tabs[0]:
    if len(refused):
        reasons = refused.get("reason", pd.Series(dtype=str)).fillna("below data-quality threshold").value_counts()
        st.caption(f"{len(refused)} fixture(s) refused: " + "; ".join(f"{k} ({v})" for k, v in reasons.items()))
    if view.empty:
        st.info("No fixtures pass the data-quality gate right now. The system refuses to predict on weak data.")
    else:
        out = view.assign(
            match=view.home + " vs " + view.away,
            ko_in_h=view.hours_to_ko.round(1),
            best_edge=view[["edge_home", "edge_draw", "edge_away"]].max(axis=1),
            pick=view[["p_home", "p_draw", "p_away"]].idxmax(axis=1).map({"p_home": "Home", "p_draw": "Draw", "p_away": "Away"}),
        )[["match", "league", "ko_in_h", "stage", "p_home", "p_draw", "p_away", "pick", "best_edge", "dqs"]]
        pc = lambda lbl: st.column_config.ProgressColumn(lbl, min_value=0, max_value=1, format="%.2f")
        st.dataframe(out, hide_index=True, width="stretch", column_config={
            "p_home": pc("P(Home)"), "p_draw": pc("P(Draw)"), "p_away": pc("P(Away)"),
            "best_edge": st.column_config.NumberColumn("Best edge", format="%.3f"),
            "dqs": st.column_config.NumberColumn("DQS", format="%.2f"), "ko_in_h": "KO in (h)"})
        vb = view[(view[["edge_home", "edge_draw", "edge_away"]].max(axis=1) >= min_edge)]
        st.subheader(f"Value flags (edge ≥ {min_edge*100:.1f}pp): {len(vb)}")
        st.caption("A flagged edge is a model-vs-market gap, not a sure win. Check CLV over a large sample.")

# ---- Match detail
with tabs[1]:
    if view.empty:
        st.info("Nothing to show.")
    else:
        lab = view.apply(lambda r: f"{r.home} vs {r.away} ({r.league})", axis=1).tolist()
        row = view.iloc[lab.index(st.selectbox("Fixture", lab))]
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            fig.add_bar(name="Model", x=["Home", "Draw", "Away"], y=[row.p_home, row.p_draw, row.p_away])
            fig.add_bar(name="Market (de-vigged)", x=["Home", "Draw", "Away"], y=[row.mkt_home, row.mkt_draw, row.mkt_away])
            fig.update_layout(barmode="group", yaxis_title="Probability", height=360, margin=dict(t=20))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.metric("Stage", row.stage); st.metric("Data quality", f"{row.dqs:.2f}")
            st.metric("CI half-width", f"±{row.ci_width*100:.1f}pp")
            st.metric("Expected goals", f"{row.xg_home:.2f} - {row.xg_away:.2f}")
        st.subheader("Simulated staking (fractional Kelly, 2% cap)")
        tbl = pd.DataFrame({"outcome": ["Home", "Draw", "Away"],
                            "model_p": [row.p_home, row.p_draw, row.p_away],
                            "odds": [row.odds_home, row.odds_draw, row.odds_away],
                            "edge": [row.edge_home, row.edge_draw, row.edge_away]})
        tbl["sim_stake_%"] = [100 * kelly_stake(p, o, kf) for p, o in zip(tbl.model_p, tbl.odds)]
        st.dataframe(tbl.round(3), hide_index=True, width="stretch")
        with st.expander("Audit record (reproducibility)"):
            st.json({"fixture_id": int(row.fixture_id), "model_version": row.model_version,
                     "stage": row.stage, "dqs": round(row.dqs, 3), "locked": row.hours_to_ko <= 0.5})

# ---- Brainbox
with tabs[2]:
    st.caption("Meta-learning layer: weights, drift, calibration, champion/challenger.")
    if bb_demo: st.caption("Synthetic Brainbox metrics.")
    ll = bb["logloss"]
    if ll.empty:
        st.info("No settled predictions scored yet. The Brainbox learns after matches finish.")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure([go.Scatter(x=ll.index, y=ll[m], name=m, mode="lines") for m in ll.columns])
        fig.update_layout(title="Rolling log loss by model (lower is better)", height=340, margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")
    with c2:
        w = bb["weights"]
        st.plotly_chart(go.Figure(go.Bar(x=w.index, y=w.values)).update_layout(
            title=f"Ensemble weights (market trust {bb.get('market_weight', 0):.0%})", height=340, margin=dict(t=40)), width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        cal = bb["calib"]; fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash"))
        fig.add_scatter(x=cal.pred, y=cal.obs, mode="lines+markers", name="Observed")
        fig.update_layout(title="Reliability curve" + (f" (ECE {bb['ece']:.3f})" if bb.get("ece") is not None else " (ECE needs 150+ matches)"), height=340, margin=dict(t=40),
                          xaxis_title="Predicted", yaxis_title="Observed")
        st.plotly_chart(fig, width="stretch")
    with c4:
        st.markdown("**Drift detector (PSI > 0.2 triggers recalibration)**")
        st.dataframe(bb["drift"].round(3), hide_index=True, width="stretch")
    st.markdown("**Champion / challenger gate** (promote only if p < 0.05 and calibration holds)")
    st.dataframe(bb["cc"], hide_index=True, width="stretch")
    sup = (bb.get("guardrail") or {}).get("suppressed", [])
    if bb.get("ece") is None:
        st.info("Guardrail: waiting for enough settled matches to measure calibration.")
    elif sup:
        st.error(f"Guardrail: predictions suppressed for {', '.join(sup)} (calibration above limit).")
    else:
        st.success(f"Guardrail: ECE {bb['ece']:.3f} within limit. Predictions enabled.")
    for a in (bb.get("actions") or [])[-5:]:
        st.caption(f"Brainbox action: {a}")

# ---- Data quality
with tabs[3]:
    s = src.copy(); s["stale"] = s.freshness_s > s.ttl_s * 3
    st.dataframe(s, hide_index=True, width="stretch")
    st.plotly_chart(go.Figure(go.Histogram(x=fx.dqs, nbinsx=15)).update_layout(
        title="DQS distribution (gate at 0.80)", height=300, margin=dict(t=40)), width="stretch")
    st.metric("Fixtures blocked by DQS gate", int((~fx.predictable).sum()))

# ---- Backtest
with tabs[4]:
    if bt_demo:
        st.caption("Demo curves are synthetic.")
    if bt.empty:
        st.info("No backtest stored yet. Run: python -m edgeagent.cli backtest")
    else:
        c1, c2 = st.columns(2)
        c1.plotly_chart(go.Figure(go.Scatter(x=bt.bet, y=bt.bankroll)).update_layout(
            title="Simulated bankroll", height=300, margin=dict(t=40)), width="stretch")
        c2.plotly_chart(go.Figure(go.Scatter(x=bt.bet, y=bt.clv_cum, line=dict(color="#22c55e"))).update_layout(
            title="Cumulative closing-line value", height=300, margin=dict(t=40)), width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Max drawdown", f"{bt.drawdown.min()*100:.1f}%")
        m2.metric("Final bankroll (start 100)", f"{bt.bankroll.iloc[-1]:.1f}")
        m3.metric("Cumulative CLV", f"{bt.clv_cum.iloc[-1]*100:.1f}%")
    if bt_summary:
        rows = {k: bt_summary[k] for k in ("model", "final", "market") if k in bt_summary}
        st.markdown("**Probability quality on the walk-forward test set (lower log loss / Brier is better)**")
        st.dataframe(pd.DataFrame(rows).T.assign(uniform_logloss=bt_summary["uniform_logloss"]).round(4), width="stretch")
        bet = bt_summary.get("betting", {})
        st.caption(f"{bet.get('n_bets', 0)} simulated bets | ROI/bet {bet.get('roi_per_bet', 0):+.3f} "
                   f"(95% CI {bet.get('roi_ci95', ['?', '?'])[0]:.3f} to {bet.get('roi_ci95', ['?', '?'])[1]:.3f}) | {bet.get('note', '')}")

# ---- Aviator Lab
with tabs[5]:
    st.subheader("Why Aviator can't be predicted")
    st.write("Each round is independent. Any fixed cash-out strategy converges to the house edge.")
    c1, c2, c3 = st.columns(3)
    edge_h = c1.slider("House edge", 0.01, 0.08, 0.03, 0.01)
    target = c2.slider("Cash-out multiplier", 1.1, 10.0, 2.0, 0.1)
    rounds = c3.select_slider("Rounds", [1_000, 10_000, 100_000, 1_000_000], 100_000)
    if st.button("Run simulation"):
        rng = np.random.default_rng()
        u = rng.random(rounds)
        crash = np.maximum(1.0, (1 - edge_h) / u)
        pnl = np.where(crash >= target, target - 1, -1.0)
        st.plotly_chart(go.Figure(go.Scatter(y=np.cumsum(pnl)[::max(1, rounds // 2000)])).update_layout(
            title="Cumulative profit (1 unit per round)", height=320, margin=dict(t=40)), width="stretch")
        st.metric("Avg return per round", f"{pnl.mean():+.4f}", help=f"Theory: {-edge_h:+.3f}")
        st.caption("Streaks and 'patterns' are noise. Predictor tools and signals sold online are scams.")
