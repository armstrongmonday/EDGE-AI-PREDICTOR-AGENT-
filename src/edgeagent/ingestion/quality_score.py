"""Data Quality Score (DQS 0-1): history depth, odds freshness, lineup/injury context, source agreement."""
from __future__ import annotations


def compute_dqs(n_hist: float, has_odds: bool, odds_age_s: float, odds_ttl_s: float, hours_to_ko: float,
                lineups: bool, injuries: bool, dispersion: float = 0.0, kickoff_diff_s: float = 0.0) -> dict:
    hist = min(n_hist / 20.0, 1.0)
    if has_odds:
        odds = 1.0 if odds_age_s <= odds_ttl_s else max(0.0, 1 - (odds_age_s - odds_ttl_s) / (5 * odds_ttl_s))
    else:
        odds = 0.5
    ctx = ((1.0 if hours_to_ko > 1.5 else float(lineups)) + float(injuries)) / 2
    agree = 1 - 0.5 * min(dispersion / 0.05, 1) - 0.5 * min(kickoff_diff_s / 1800, 1)
    score = 0.3 * hist + 0.3 * odds + 0.2 * ctx + 0.2 * agree
    return {"score": round(score, 4), "history": hist, "odds": odds, "context": ctx, "agreement": agree}
