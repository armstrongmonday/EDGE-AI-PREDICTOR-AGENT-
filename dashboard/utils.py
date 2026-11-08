"""Pure helpers mirroring README v2: de-vig, edge, fractional Kelly, staged snapshots."""
import numpy as np

STAGES = [(24, "T-24h baseline"), (6, "T-6h injuries/market"), (1, "T-1h lineups"), (0.5, "T-30m locked")]

def devig(odds):
    """Decimal odds -> margin-free probabilities (proportional method)."""
    imp = 1 / np.asarray(odds, dtype=float)
    return imp / imp.sum()

def edge(model_p, market_p):
    return np.asarray(model_p) - np.asarray(market_p)

def kelly_stake(p, odds, fraction=0.15, cap=0.02):
    """Simulated stake as share of bankroll. Fractional Kelly with exposure cap."""
    b = odds - 1
    f = (p * odds - 1) / b if b > 0 else 0.0
    return float(np.clip(f * fraction, 0.0, cap))

def stage_for(hours_to_ko):
    if hours_to_ko > 24:
        return "Waiting (>24h)"
    label = STAGES[0][1]
    for h, name in STAGES:
        if hours_to_ko <= h:
            label = name
    return label

def log_loss(probs, outcome_idx):
    return float(-np.log(np.clip(probs[outcome_idx], 1e-12, 1)))
