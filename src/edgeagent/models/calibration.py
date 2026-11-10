from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    """Per-class isotonic calibration, renormalised. Stays identity until enough out-of-sample data exists."""
    def __init__(self, min_samples: int = 300):
        self.min_samples, self.iso = min_samples, None
    def fit(self, P, y):
        P, y = np.asarray(P), np.asarray(y)
        if len(y) < self.min_samples:
            self.iso = None
            return self
        self.iso = [IsotonicRegression(y_min=1e-3, y_max=1 - 1e-3, out_of_bounds="clip").fit(P[:, k], (y == k).astype(float)) for k in range(3)]
        return self
    @property
    def fitted(self) -> bool:
        return self.iso is not None
    def transform(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        if self.iso is None:
            return P
        Q = np.column_stack([self.iso[k].predict(P[:, k]) for k in range(3)])
        return Q / Q.sum(1, keepdims=True)

def reliability(P, y, bins: int = 10):
    """All classes pooled: returns (mean predicted, observed frequency, count) per bin."""
    p = np.asarray(P).ravel(); t = (np.asarray(y)[:, None] == np.arange(3)[None, :]).ravel().astype(float)
    ix = np.minimum((p * bins).astype(int), bins - 1)
    pred, obs, cnt = [], [], []
    for b in range(bins):
        m = ix == b
        if m.any():
            pred.append(p[m].mean()); obs.append(t[m].mean()); cnt.append(int(m.sum()))
    return np.array(pred), np.array(obs), np.array(cnt)

def ece(P, y, bins: int = 10) -> float:
    pred, obs, cnt = reliability(P, y, bins)
    return float((np.abs(pred - obs) * cnt).sum() / cnt.sum()) if len(cnt) else 0.0
