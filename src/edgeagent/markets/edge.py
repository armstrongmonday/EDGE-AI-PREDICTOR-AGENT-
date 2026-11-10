from __future__ import annotations

import numpy as np


def edge(p, market_p) -> np.ndarray:
    return np.asarray(p, float) - np.asarray(market_p, float)

def clv(taken_odds: float, close_prob: float) -> float:
    """Closing-line value: taken odds vs the de-vigged closing fair price. Positive = beat the close."""
    return float(taken_odds * close_prob - 1)
