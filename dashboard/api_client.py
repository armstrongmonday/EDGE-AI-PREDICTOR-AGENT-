"""Talks to the FastAPI backend (src/edgeagent/api). Falls back to clearly-flagged synthetic demo data."""
import os
import numpy as np
import pandas as pd
import requests
import demo_data as demo

BASE = os.getenv("EDGEAGENT_API_URL", "http://localhost:8000")
FX_COLS = ["fixture_id", "league", "home", "away", "kickoff", "hours_to_ko", "stage", "p_home", "p_draw", "p_away",
           "odds_home", "odds_draw", "odds_away", "mkt_home", "mkt_draw", "mkt_away", "edge_home", "edge_draw", "edge_away",
           "ci_width", "dqs", "predictable", "reason", "xg_home", "xg_away", "model_version"]
NUMERIC = [c for c in FX_COLS if c not in ("fixture_id", "league", "home", "away", "kickoff", "stage", "predictable", "reason", "model_version")]

def _get(path, params=None):
    r = requests.get(f"{BASE}{path}", params=params, timeout=5)
    r.raise_for_status()
    return r.json()

def get_meta():
    try:
        return _get("/v1/meta")
    except Exception:
        return {"source": "demo", "synthetic": True, "reachable": False}

def get_fixtures():
    try:
        df = pd.DataFrame(_get("/v1/fixtures/upcoming")).reindex(columns=FX_COLS)
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
        for c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["predictable"] = df["predictable"].fillna(False).astype(bool)
        return df, False
    except Exception:
        return demo.fixtures(), True

def get_brainbox():
    try:
        js = _get("/v1/brainbox/status")
        ll = js["logloss"]
        out = dict(js)
        out["logloss"] = pd.DataFrame({k: pd.to_numeric(pd.Series(v), errors="coerce") for k, v in ll["series"].items()},
                                      index=pd.to_datetime(ll["dates"]))
        out["weights"] = pd.Series(js["weights"], dtype=float)
        out["calib"] = pd.DataFrame(js["calib"]).reindex(columns=["pred", "obs"])
        out["drift"] = pd.DataFrame(js["drift"]).reindex(columns=["feature", "psi", "ks_p", "status"])
        cc = pd.DataFrame(js["cc"]).reindex(columns=["model", "matches_shadowed", "d_logloss", "p_value", "note"])
        out["cc"] = cc.rename(columns={"note": "decision"})
        out["ece"] = js.get("ece")
        return out, False
    except Exception:
        d = demo.brainbox()
        return d, True

def get_sources():
    try:
        return pd.DataFrame(_get("/v1/data/sources")), False
    except Exception:
        return demo.sources(), True

def get_backtest():
    try:
        df = pd.DataFrame(_get("/v1/backtest/latest")).reindex(columns=["bet", "bankroll", "drawdown", "clv_cum"])
        try:
            summ = _get("/v1/backtest/summary")
        except Exception:
            summ = {}
        return df, summ, False
    except Exception:
        return demo.backtest(), {}, True
