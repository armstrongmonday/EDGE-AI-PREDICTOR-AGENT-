from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def proportional(odds) -> np.ndarray:
    imp = 1 / np.asarray(odds, float)
    return imp / imp.sum(-1, keepdims=True)

def power(odds) -> np.ndarray:
    odds = np.asarray(odds, float)
    if odds.ndim > 1:
        return np.array([power(r) for r in odds])
    imp = 1 / odds
    k = brentq(lambda k: (imp ** k).sum() - 1, 0.5, 5.0)
    return imp ** k

def devig(odds, method: str = "proportional") -> np.ndarray:
    return power(odds) if method == "power" else proportional(odds)
