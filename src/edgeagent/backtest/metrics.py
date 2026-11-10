from __future__ import annotations

import numpy as np


def logloss(P, y) -> float:
    P, y = np.asarray(P), np.asarray(y)
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())

def brier(P, y) -> float:
    P, y = np.asarray(P), np.asarray(y)
    return float(((P - np.eye(3)[y]) ** 2).sum(1).mean())

def bootstrap_ci(x, n=1000, seed=0, q=(2.5, 97.5)):
    x = np.asarray(x, float)
    if len(x) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return tuple(map(float, np.percentile(m, q)))
