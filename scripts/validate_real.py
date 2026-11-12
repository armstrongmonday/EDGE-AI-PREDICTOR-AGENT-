"""Real-data sanity check: walk-forward evaluation on real English Premier League results (footballcsv, 2010-11 to 2019-20).
These files have NO bookmaker odds, so this measures predictive SKILL versus naive baselines. It cannot measure
betting edge or whether the model beats the market; use football-data.co.uk odds (backfill) for that."""
from __future__ import annotations
import io, json, sys
import httpx
import pandas as pd
from edgeagent.backtest import walk_forward
from edgeagent.config import BB_DEFAULTS, CONFIGS, RISK_DEFAULTS
from edgeagent.ingestion.matching import normalize_team
from edgeagent.utils import make_fixture_id

BASE = "https://raw.githubusercontent.com/footballcsv/england/master/2010s"

def load(seasons=range(2010, 2020)) -> pd.DataFrame:
    frames = []
    for y in seasons:
        r = httpx.get(f"{BASE}/{y}-{str(y + 1)[2:]}/eng.1.csv", timeout=30); r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text)).dropna(subset=["FT"])
        score = d.FT.str.extract(r"(\d+)-(\d+)").astype(int)
        frames.append(pd.DataFrame({"date": pd.to_datetime(d.Date, format="%a %b %d %Y"), "home": d["Team 1"].map(normalize_team),
                                    "away": d["Team 2"].map(normalize_team), "hg": score[0], "ag": score[1]}))
    m = pd.concat(frames).sort_values("date").reset_index(drop=True)
    m["league"], m["status"] = "EPL", "FT"
    m["fixture_id"] = [make_fixture_id("EPL", r.date, r.home, r.away) for r in m.itertuples()]
    for c in ["odds_h", "odds_d", "odds_a", "close_h", "close_d", "close_a"]:
        m[c] = float("nan")
    return m

if __name__ == "__main__":
    m = load()
    print(f"{len(m)} real matches, {m.date.min():%Y-%m-%d} to {m.date.max():%Y-%m-%d}", file=sys.stderr)
    res = walk_forward.run(m, CONFIGS["A"], BB_DEFAULTS, RISK_DEFAULTS, retrain_days=120)
    print(json.dumps({k: v for k, v in res["summary"].items() if k != "betting"}, indent=2, default=float))
