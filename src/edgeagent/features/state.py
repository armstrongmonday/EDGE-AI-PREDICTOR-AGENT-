"""Leak-free incremental team state: Elo + exponentially weighted form. Features are computed BEFORE a match is applied."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

HOME_ADV = 65.0
FEATURES = ["elo_diff", "gd_diff", "pts_diff", "att_diff", "def_diff", "rest_h", "rest_a", "n_h", "n_a"]

class TeamState:
    def __init__(self, k: float = 20.0, alpha: float = 0.15):
        self.k, self.alpha = k, alpha
        self.elo = defaultdict(lambda: 1500.0)
        self.gd = defaultdict(float)
        self.pts = defaultdict(lambda: 1.3)
        self.gf = defaultdict(lambda: 1.35)
        self.ga = defaultdict(lambda: 1.35)
        self.last, self.n = {}, defaultdict(int)

    def _rest(self, t, date) -> float:
        return float(min((date - self.last[t]).days, 14)) if t in self.last else 7.0

    def features(self, home: str, away: str, date) -> dict:
        return {"elo_diff": self.elo[home] - self.elo[away], "gd_diff": self.gd[home] - self.gd[away],
                "pts_diff": self.pts[home] - self.pts[away], "att_diff": self.gf[home] - self.gf[away],
                "def_diff": self.ga[away] - self.ga[home], "rest_h": self._rest(home, date),
                "rest_a": self._rest(away, date), "n_h": float(self.n[home]), "n_a": float(self.n[away])}

    def update(self, home: str, away: str, hg: int, ag: int, date) -> None:
        d = self.elo[home] + HOME_ADV - self.elo[away]
        e = 1 / (1 + 10 ** (-d / 400))
        s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = self.k * (np.log(abs(hg - ag) + 1) + 1) * (s - e)
        self.elo[home] += delta
        self.elo[away] -= delta
        a = self.alpha
        for t, gf, ga in ((home, hg, ag), (away, ag, hg)):
            pts = 3.0 if gf > ga else 1.0 if gf == ga else 0.0
            self.gd[t] = a * (gf - ga) + (1 - a) * self.gd[t]
            self.pts[t] = a * pts + (1 - a) * self.pts[t]
            self.gf[t] = a * gf + (1 - a) * self.gf[t]
            self.ga[t] = a * ga + (1 - a) * self.ga[t]
            self.last[t] = date
            self.n[t] += 1

def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, TeamState]:
    """matches must be sorted by date. Returns pre-match features per row and the final state."""
    st, rows = TeamState(), []
    for r in matches.itertuples(index=False):
        rows.append(st.features(r.home, r.away, r.date))
        st.update(r.home, r.away, int(r.hg), int(r.ag), r.date)
    return pd.DataFrame(rows, index=matches.index, columns=FEATURES), st
