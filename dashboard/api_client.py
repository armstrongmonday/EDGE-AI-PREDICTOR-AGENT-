"""Talks to the FastAPI backend (src/edgeagent/api). Falls back to synthetic demo data."""
import os, requests
import pandas as pd
import demo_data as demo

BASE = os.getenv("EDGEAGENT_API_URL", "http://localhost:8000")

def _get(path, params=None):
    r = requests.get(f"{BASE}{path}", params=params, timeout=3)
    r.raise_for_status()
    return r.json()

def get_fixtures():
    try:
        df = pd.DataFrame(_get("/v1/fixtures/upcoming"))
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
        return df, False
    except Exception:
        return demo.fixtures(), True

def get_brainbox():
    try:
        return _get("/v1/brainbox/status"), False
    except Exception:
        return demo.brainbox(), True

def get_sources():
    try:
        return pd.DataFrame(_get("/v1/data/sources")), False
    except Exception:
        return demo.sources(), True

def get_backtest():
    try:
        return pd.DataFrame(_get("/v1/backtest/latest")), False
    except Exception:
        return demo.backtest(), True
