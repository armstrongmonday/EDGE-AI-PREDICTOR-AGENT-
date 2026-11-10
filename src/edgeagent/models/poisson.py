"""Independent-Poisson attack/defence model with time-decay weights (Dixon-Coles low-score correction not included)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .base import Model

K = 10
_I, _J = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")

def score_matrix(lh: np.ndarray, la: np.ndarray) -> np.ndarray:
    k = np.arange(K)
    J = poisson.pmf(k[None, :], lh[:, None])[:, :, None] * poisson.pmf(k[None, :], la[:, None])[:, None, :]
    return J / J.sum((1, 2), keepdims=True)

def probs_from_lams(lh, la) -> np.ndarray:
    J = score_matrix(np.asarray(lh, float), np.asarray(la, float))
    return np.column_stack([(J * (_I > _J)).sum((1, 2)), (J * (_I == _J)).sum((1, 2)), (J * (_I < _J)).sum((1, 2))])

class PoissonAD(Model):
    name = "poisson_ad"
    def __init__(self, xi: float = 0.0018, reg: float = 2.0):
        self.xi, self.reg = xi, reg
        self.att, self.dfn, self.mu, self.ha = {}, {}, 0.2, 0.25

    def fit(self, matches, X=None):
        teams = sorted(set(matches.home) | set(matches.away))
        ix = {t: i for i, t in enumerate(teams)}
        T = len(teams)
        h = matches.home.map(ix).to_numpy(); a = matches.away.map(ix).to_numpy()
        gh = matches.hg.to_numpy(float); ga = matches.ag.to_numpy(float)
        days = (matches.date.max() - matches.date).dt.total_seconds().to_numpy() / 86400
        w = np.exp(-self.xi * days)

        def f(x):
            att, dfn, mu, ha = x[:T], x[T:2 * T], x[-2], x[-1]
            lh = np.exp(mu + ha + att[h] - dfn[a]); la = np.exp(mu + att[a] - dfn[h])
            nll = (w * (lh - gh * np.log(lh))).sum() + (w * (la - ga * np.log(la))).sum()
            nll += self.reg * ((att ** 2).sum() + (dfn ** 2).sum())
            rh, ra = w * (lh - gh), w * (la - ga)
            g = np.zeros_like(x)
            np.add.at(g, h, rh); np.add.at(g, a, ra)
            np.add.at(g, T + a, -rh); np.add.at(g, T + h, -ra)
            g[:T] += 2 * self.reg * att; g[T:2 * T] += 2 * self.reg * dfn
            g[-2], g[-1] = (rh + ra).sum(), rh.sum()
            return nll, g

        x0 = np.zeros(2 * T + 2); x0[-2] = 0.2; x0[-1] = 0.25
        r = minimize(f, x0, jac=True, method="L-BFGS-B")
        self.att = {t: r.x[i] for t, i in ix.items()}; self.dfn = {t: r.x[T + i] for t, i in ix.items()}
        self.mu, self.ha = float(r.x[-2]), float(r.x[-1])
        return self

    def lambdas(self, matches: pd.DataFrame):
        ah = matches.home.map(lambda t: self.att.get(t, 0.0)).to_numpy(); aa = matches.away.map(lambda t: self.att.get(t, 0.0)).to_numpy()
        dh = matches.home.map(lambda t: self.dfn.get(t, 0.0)).to_numpy(); da = matches.away.map(lambda t: self.dfn.get(t, 0.0)).to_numpy()
        return np.exp(self.mu + self.ha + ah - da), np.exp(self.mu + aa - dh)

    def predict_many(self, matches, X=None):
        return probs_from_lams(*self.lambdas(matches))

    def goals(self, home: str, away: str) -> tuple[float, float]:
        lh, la = self.lambdas(pd.DataFrame({"home": [home], "away": [away]}))
        return float(lh[0]), float(la[0])
