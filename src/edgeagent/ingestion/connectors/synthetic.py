"""SYNTHETIC league simulator. Lets the whole pipeline run offline. It is NOT real sport data and says nothing
about real-world accuracy. The simulated market is unbiased around the true probabilities plus noise, so a
model cannot systematically beat it here, which is the realistic expectation as well."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ...models.poisson import probs_from_lams
from ...utils import stable_rand, utcnow

NAMES = ["Northgate", "Eastvale", "Westmoor", "Southfield", "Ashford", "Brookhaven", "Cedar Park", "Dunmore", "Elmwood",
         "Fairhaven", "Glenrock", "Highcliff", "Ironbridge", "Juniper", "Kingsmere", "Lakeside", "Marlow", "Netherby"]

def _norm(lp: np.ndarray) -> np.ndarray:
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)

class SyntheticSource:
    is_synthetic = True
    name = "synthetic"

    def __init__(self, seed: int = 42, leagues=("ALPHA", "BETA"), n_teams: int = 16, seasons: int = 3,
                 future_rounds: int = 40, now=None):
        self.rng = np.random.default_rng(seed)
        self._now = (now or utcnow()).replace(minute=0, second=0, microsecond=0)
        self.league_names = {c: f"{c.title()} League" for c in leagues}
        self._build(leagues, n_teams, seasons, future_rounds)

    def now(self):
        return self._now

    def advance(self, hours: float):
        self._now = self._now + timedelta(hours=hours)

    def _build(self, leagues, n, seasons, future_rounds):
        rows = []
        r_past = seasons * 2 * (n - 1)
        for code in leagues:
            teams = [f"{NAMES[i]} {code[0]}" for i in range(n)]
            att = self.rng.normal(0, 0.25, n); dfn = self.rng.normal(0, 0.25, n)
            att -= att.mean(); dfn -= dfn.mean()
            arr, base = list(range(n)), []
            for r in range(n - 1):
                pairs = [(arr[i], arr[n - 1 - i]) for i in range(n // 2)]
                base.append([(b, a) for a, b in pairs] if r % 2 else pairs)
                arr = [arr[0]] + [arr[-1]] + arr[1:-1]
            for rnd in range(r_past + future_rounds):
                r = rnd % (2 * (n - 1))
                pairs = base[r % (n - 1)]
                if r >= n - 1:
                    pairs = [(b, a) for a, b in pairs]
                for k, (h, a) in enumerate(pairs):
                    lh = np.exp(0.2 + 0.25 + att[h] - dfn[a]); la = np.exp(0.2 + att[a] - dfn[h])
                    if rnd < r_past:
                        date = self._now - timedelta(days=7 * (r_past - rnd)) - timedelta(hours=float(self.rng.uniform(1, 72)))
                    else:
                        date = self._now + timedelta(days=7 * (rnd - r_past)) + timedelta(hours=float(self.rng.uniform(0.3, 72)))
                    rows.append(dict(fixture_id=f"S-{code}-{rnd:03d}-{k}", league=code, date=date, home=teams[h], away=teams[a],
                                     lh=lh, la=la, hg=int(self.rng.poisson(lh)), ag=int(self.rng.poisson(la)),
                                     e=self.rng.normal(0, 1, 3)))
        df = pd.DataFrame(rows)
        self.true = probs_from_lams(df.lh.to_numpy(), df.la.to_numpy())
        self.all = df.drop(columns=["lh", "la"]).reset_index(drop=True)

    def _odds(self, idx, sd, margin):
        lp = np.log(self.true[idx]) + sd[:, None] * np.stack(self.all.e.to_numpy()[idx])
        return 1 / (_norm(lp) * (1 + margin))

    def _snap(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["fixture_id", "league", "date", "home", "away", "odds_h", "odds_d", "odds_a", "close_h", "close_d", "close_a"])
        idx = df.index.to_numpy()
        h = np.clip((df.date - self._now).dt.total_seconds().to_numpy() / 3600, 0, 72)
        O = self._odds(idx, 0.03 + 0.05 * h / 72, 0.05)
        C = self._odds(idx, np.full(len(idx), 0.03), 0.05)
        out = df[["fixture_id", "league", "date", "home", "away"]].copy()
        out[["odds_h", "odds_d", "odds_a"]] = O
        out[["close_h", "close_d", "close_a"]] = C
        return out

    async def history(self, leagues=None, seasons: int = 3) -> pd.DataFrame:
        d = self.all[self.all.date + timedelta(hours=2) <= self._now]
        out = self._snap(d)
        out["hg"], out["ag"], out["status"] = d.hg, d.ag, "FT"
        return out

    async def upcoming(self, leagues=None, days: int = 3) -> pd.DataFrame:
        d = self.all[(self.all.date > self._now) & (self.all.date <= self._now + timedelta(days=days))]
        out = self._snap(d)
        out[["close_h", "close_d", "close_a"]] = np.nan
        out["hg"], out["ag"], out["status"] = np.nan, np.nan, "NS"
        return out

    async def results(self, ids) -> pd.DataFrame:
        d = self.all[self.all.fixture_id.isin(ids) & (self.all.date + timedelta(hours=2) <= self._now)]
        out = self._snap(d)
        out["hg"], out["ag"], out["status"] = d.hg, d.ag, "FT"
        return out

    async def context(self, fixture_id: str, hours_to_ko: float) -> dict:
        u = stable_rand(fixture_id)
        stale = u > 0.93
        return {"odds_age_s": 4000.0 if stale else 15 + 400 * stable_rand(fixture_id + "a"),
                "odds_ttl_s": 900.0, "lineups": stable_rand(fixture_id + "l") < 0.9, "injuries": True,
                "dispersion": 0.01 + 0.02 * stable_rand(fixture_id + "d"), "kickoff_diff_s": 0.0}

    def status(self) -> list[dict]:
        return [{"source": "synthetic-generator", "freshness_s": 0, "ttl_s": 900, "completeness": 1.0, "status": "OK (synthetic)"}]
