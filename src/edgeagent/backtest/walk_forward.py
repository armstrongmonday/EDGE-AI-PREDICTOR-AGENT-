"""Walk-forward backtest: models are refit on past data only; market trust is learned from earlier blocks only."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..features.state import build_features
from ..markets.devig import devig
from ..models.base import outcome
from ..models.bundle import fit_bundle, market_probs, predict_many_bundle
from ..models.calibration import ece
from ..risk.kelly import kelly_stake
from .metrics import bootstrap_ci, brier, logloss


def run(matches: pd.DataFrame, cfg: dict, bb: dict, risk: dict, start_frac: float = 0.5, retrain_days: int = 60) -> dict:
    M = matches.sort_values("date").reset_index(drop=True)
    X, _ = build_features(M)
    n, y, mk = len(M), outcome(M), market_probs(M)
    start = int(n * start_frac)
    model_p, blocks, idx = np.full((n, 3), np.nan), [], start
    while idx < n:
        end = idx
        while end < n and M.date[end] < M.date[idx] + timedelta(days=retrain_days):
            end += 1
        b = fit_bundle(M.iloc[:idx], X.iloc[:idx], cfg, bb)
        model_p[idx:end] = predict_many_bundle(b, M.iloc[idx:end], X.iloc[idx:end])[2]
        blocks.append((idx, end)); idx = end

    final, grid = model_p.copy(), np.linspace(0, 1, 21)
    for s, e in blocks:
        prev = np.arange(start, s)
        prev = prev[~np.isnan(mk[prev]).any(1)] if len(prev) else prev
        w = 0.5
        if len(prev) >= 150:
            w = float(grid[int(np.argmin([logloss((1 - g) * model_p[prev] + g * mk[prev], y[prev]) for g in grid]))])
        has = ~np.isnan(mk[s:e]).any(1)
        seg = final[s:e]
        seg[has] = (1 - w) * model_p[s:e][has] + w * mk[s:e][has]

    T = np.arange(start, n)
    hm = T[~np.isnan(mk[T]).any(1)]
    freq = np.bincount(y[:start], minlength=3) / start
    acc = lambda P: float((P.argmax(1) == y[T]).mean())
    summary = {"n_test": len(T), "uniform_logloss": float(np.log(3)),
               "freq_baseline": {"logloss": logloss(np.tile(freq, (len(T), 1)), y[T]), "accuracy": float((freq.argmax() == y[T]).mean())},
               "model": {"logloss": logloss(model_p[T], y[T]), "brier": brier(model_p[T], y[T]), "ece": ece(model_p[T], y[T]), "accuracy": acc(model_p[T])},
               "final": {"logloss": logloss(final[T], y[T]), "brier": brier(final[T], y[T]), "ece": ece(final[T], y[T]), "accuracy": acc(final[T])}}
    if len(hm):
        summary["market"] = {"logloss": logloss(mk[hm], y[hm]), "brier": brier(mk[hm], y[hm]), "ece": ece(mk[hm], y[hm])}

    bank, peak, rows, profits, clvs = 100.0, 100.0, [], [], []
    O = M[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    C = M[["close_h", "close_d", "close_a"]].to_numpy(float)
    for i in hm:
        e = final[i] - mk[i]
        k = int(np.argmax(e))
        if e[k] < risk["min_edge"]:
            continue
        stake = bank * kelly_stake(final[i, k], O[i, k], risk["kelly_fraction"], risk["max_stake_pct"])
        if stake <= 0:
            continue
        pnl = stake * (O[i, k] - 1) if y[i] == k else -stake
        bank += pnl; peak = max(peak, bank)
        profits.append(pnl / stake)
        if not np.isnan(C[i]).any():
            clvs.append(float(O[i, k] * devig(C[i])[k] - 1))
        rows.append({"bet": len(rows), "bankroll": bank, "drawdown": bank / peak - 1, "clv_cum": float(np.sum(clvs))})
    summary["betting"] = {"n_bets": len(rows), "final_bankroll": bank,
                          "max_drawdown": min([r["drawdown"] for r in rows], default=0.0),
                          "roi_per_bet": float(np.mean(profits)) if profits else 0.0, "roi_ci95": bootstrap_ci(profits),
                          "mean_clv": float(np.mean(clvs)) if clvs else 0.0, "clv_ci95": bootstrap_ci(clvs),
                          "note": "simulated flat-odds fills at stored prices; no limits, slippage or account restrictions"}
    return {"summary": summary, "curve": rows}
