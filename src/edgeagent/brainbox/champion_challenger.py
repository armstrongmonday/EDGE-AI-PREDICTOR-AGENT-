from __future__ import annotations

import numpy as np
from scipy.stats import ttest_1samp


def compare(ll_champ, ll_chall, min_matches: int, alpha: float, chall_ece_ok: bool = True) -> dict:
    """d_logloss = challenger - champion (negative = challenger better). One-sided paired t-test."""
    d = np.asarray(ll_chall, float) - np.asarray(ll_champ, float)
    n = len(d)
    if n < 5 or d.std() == 0:
        return {"matches_shadowed": n, "d_logloss": float(d.mean()) if n else 0.0, "p_value": None, "promote": False, "note": "insufficient data"}
    p = float(ttest_1samp(d, 0.0, alternative="less").pvalue)
    ok = n >= min_matches and p < alpha and d.mean() < 0 and chall_ece_ok
    note = "promote" if ok else ("need more matches" if n < min_matches else "not significant")
    return {"matches_shadowed": n, "d_logloss": float(d.mean()), "p_value": p, "promote": bool(ok), "note": note}
