"""SYNTHETIC data so the UI runs before the backend exists. Never real predictions."""
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from utils import devig, edge, stage_for

TEAMS = {
    "Premier League": ["Arsenal", "Chelsea", "Liverpool", "Man City", "Spurs", "Newcastle"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico", "Sevilla", "Valencia", "Villarreal"],
    "Serie A": ["Inter", "Milan", "Juventus", "Napoli", "Roma", "Lazio"],
}
MODELS = ["Elo", "Dixon-Coles", "LightGBM", "Logistic", "NeuralNet", "Ensemble"]
MIN_DQS = 0.8

def fixtures(n=12, seed=7):
    rng = np.random.default_rng(seed)
    now, rows, leagues = datetime.now(timezone.utc), [], list(TEAMS)
    for i in range(n):
        lg = leagues[i % 3]
        h, a = rng.choice(TEAMS[lg], 2, replace=False)
        p = rng.dirichlet([4, 2.5, 3])
        m = np.clip(p + rng.normal(0, 0.03, 3), 0.03, None); m /= m.sum()
        odds = 1 / (m * 1.05)
        mkt = devig(odds); hrs = float(rng.uniform(0.3, 70)); dqs = float(rng.uniform(0.62, 1.0))
        ok = dqs >= MIN_DQS
        e = edge(p, mkt)
        rows.append(dict(
            fixture_id=1000 + i, league=lg, home=h, away=a,
            kickoff=now + timedelta(hours=hrs), hours_to_ko=hrs, stage=stage_for(hrs),
            p_home=p[0] if ok else None, p_draw=p[1] if ok else None, p_away=p[2] if ok else None,
            odds_home=odds[0], odds_draw=odds[1], odds_away=odds[2],
            mkt_home=mkt[0], mkt_draw=mkt[1], mkt_away=mkt[2],
            edge_home=e[0], edge_draw=e[1], edge_away=e[2],
            ci_width=float(rng.uniform(0.03, 0.09)), dqs=dqs, predictable=ok,
            xg_home=float(rng.uniform(0.9, 2.2)), xg_away=float(rng.uniform(0.6, 1.8)),
            model_version="ens-2026.11.03"))
    return pd.DataFrame(rows).sort_values("hours_to_ko").reset_index(drop=True)

def brainbox(seed=11):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30, freq="W")
    base = dict(zip(MODELS, [1.04, 1.01, 0.995, 1.00, 1.015, 0.985]))
    ll = pd.DataFrame({m: base[m] + rng.normal(0, 0.012, 30).cumsum() * 0.2 for m in MODELS}, index=weeks)
    recent = ll.tail(8).mean(); w = np.exp(-12 * (recent - recent.min())); w /= w.sum()
    pred = np.linspace(0.05, 0.95, 10)
    obs = np.clip(pred + rng.normal(0, 0.025, 10), 0, 1)
    drift = pd.DataFrame({"feature": ["xg_diff_form", "rest_days", "market_move", "lineup_strength", "elo_gap"],
                          "psi": rng.uniform(0.02, 0.28, 5)})
    drift["status"] = np.where(drift.psi > 0.2, "DRIFT: recalibrate", "OK")
    cc = pd.DataFrame({"model": ["ens-2026.11.03 (champion)", "lgbm-challenger-A", "nn-challenger-B"],
                       "matches_shadowed": [0, 214, 131], "d_logloss": [0.0, -0.0031, 0.0012],
                       "p_value": [None, 0.21, 0.64], "promote": ["-", "Not yet (p>0.05)", "No"]})
    return dict(logloss=ll, weights=w, calib=pd.DataFrame({"pred": pred, "obs": obs}), drift=drift, cc=cc,
                ece=float(np.abs(pred - obs).mean()))

def sources():
    return pd.DataFrame({"source": ["API-Football", "Odds API", "Understat", "Weather API", "Pinnacle"],
                         "freshness_s": [42, 18, 3600, 900, 25], "ttl_s": [300, 60, 86400, 1800, 60],
                         "completeness": [0.99, 0.97, 0.93, 1.0, 0.95], "status": ["OK"] * 5})

def backtest(seed=3, n=900):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.004, 0.09, n)   # tiny edge, large variance: realistic shape
    eq = 100 * np.cumprod(1 + r * 0.02)
    dd = eq / np.maximum.accumulate(eq) - 1
    clv = np.cumsum(rng.normal(0.0008, 0.012, n))
    return pd.DataFrame({"bet": range(n), "bankroll": eq, "drawdown": dd, "clv_cum": clv})
