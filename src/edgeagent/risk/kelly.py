from __future__ import annotations

import numpy as np


def kelly_stake(p: float, odds: float, fraction: float = 0.15, cap: float = 0.02) -> float:
    """Simulated stake as share of bankroll: fractional Kelly with a per-bet cap."""
    b = odds - 1
    if b <= 0 or p is None or not np.isfinite(p):
        return 0.0
    return float(np.clip(((p * odds - 1) / b) * fraction, 0.0, cap))

def cap_daily(stakes: list[float], max_daily: float = 0.06) -> list[float]:
    tot = sum(stakes)
    return stakes if tot <= max_daily or tot == 0 else [s * max_daily / tot for s in stakes]
