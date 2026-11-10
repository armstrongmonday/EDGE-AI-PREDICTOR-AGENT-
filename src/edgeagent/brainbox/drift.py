from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(ref, cur, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    b = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(((b - a) * np.log(b / a)).sum())

class PageHinkley:
    """Detects an upward shift in a loss stream (e.g. log loss getting worse)."""
    def __init__(self, delta: float = 0.01, threshold: float = 5.0):
        self.delta, self.threshold, self.n, self.mean, self.cum, self.min = delta, threshold, 0, 0.0, 0.0, 0.0
    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min = min(self.min, self.cum)
        return (self.cum - self.min) > self.threshold

def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, features, psi_threshold=0.2, ks_alpha=0.01) -> list[dict]:
    out = []
    for f in features:
        if len(cur) < 100 or len(ref) < 100:
            out.append({"feature": f, "psi": 0.0, "ks_p": 1.0, "status": "insufficient data"}); continue
        p, ks = psi(ref[f], cur[f]), float(ks_2samp(ref[f], cur[f]).pvalue)
        out.append({"feature": f, "psi": round(p, 4), "ks_p": round(ks, 4),
                    "status": "DRIFT: recalibrate" if p > psi_threshold and ks < ks_alpha else "OK"})
    return out
