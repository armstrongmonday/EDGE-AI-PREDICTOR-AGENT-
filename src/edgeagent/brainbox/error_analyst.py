"""Rule-based post-match error analysis. Large misses are classified; only systematic patterns raise actions."""
from __future__ import annotations

from collections import Counter


def classify(ll: float, p_out: float, mkt_out: float | None, dqs: float, surprise_ll: float = 1.8) -> str | None:
    if ll < surprise_ll:
        return None
    if mkt_out is not None and mkt_out < 0.2:
        return "variance (market agreed it was unlikely)"
    if dqs < 0.9:
        return "data quality (stale or incomplete inputs)"
    if mkt_out is not None and mkt_out > 0.3:
        return "model weakness (market gave it a real chance)"
    return "unexplained"

def summarize(causes: list[str], min_misses: int = 20) -> dict:
    c = Counter(causes)
    n = sum(c.values())
    action = None
    if n >= min_misses and c.get("model weakness (market gave it a real chance)", 0) / n > 0.5:
        action = "Systematic model weakness: review features / retrain challenger"
    elif n >= min_misses and c.get("data quality (stale or incomplete inputs)", 0) / n > 0.5:
        action = "Systematic data-quality misses: check connectors and TTLs"
    return {"n_misses": n, "causes": dict(c), "action": action}
