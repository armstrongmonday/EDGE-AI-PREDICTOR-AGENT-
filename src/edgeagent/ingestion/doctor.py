"""`edgeagent doctor`: verifies keys, connectivity, league ids/sport keys and team-name matching with YOUR accounts.
Uses only free endpoints (/status, /leagues, /v4/sports) plus one fixtures call and one odds call per league."""
from __future__ import annotations
from . import matching  # noqa: F401  (ensures alias table imports cleanly)

async def run(src) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    add = lambda name, ok, detail="": out.append((name, bool(ok), detail))
    cfg = src.cfg
    try:
        acc = await src.apif.account()
        req = acc.get("requests", {})
        add("API-Football key", bool(acc), f"plan={acc.get('subscription', {}).get('plan')} used {req.get('current')}/{req.get('limit_day')} today")
    except Exception as e:
        add("API-Football key", False, f"{type(e).__name__}: {e}")
    try:
        sports = {s["key"]: s for s in await src.odds.sports()}
        add("Odds API key", True, f"{len(sports)} sports visible")
    except Exception as e:
        sports = {}
        add("Odds API key", False, f"{type(e).__name__}: {e}")
    for code in src._codes(None):
        lg = cfg.leagues[code]
        try:
            name = await src.apif.league_name(lg["af"])
            add(f"{code}: API-Football league id {lg['af']}", name is not None, f"-> {name!r} (check this is the league you expect)")
        except Exception as e:
            add(f"{code}: API-Football league id {lg['af']}", False, str(e))
        if sports:
            s = sports.get(lg["odds"])
            add(f"{code}: odds sport key {lg['odds']}", bool(s and s.get("active")), "active" if s and s.get("active") else "missing or out of season")
    try:
        up = await src.upcoming(days=3)
        if len(up):
            matched = int(up.odds_h.notna().sum())
            add("fixtures + odds matching (next 3 days)", matched / len(up) >= 0.8, f"{matched}/{len(up)} fixtures matched to bookmaker odds")
        else:
            add("fixtures + odds matching (next 3 days)", True, "no fixtures in window (off-season?)")
        for u in src.unmatched[:5]:
            add("unmatched", False, u)
    except Exception as e:
        add("fixtures + odds matching", False, f"{type(e).__name__}: {e}")
    try:
        h = await src.history(seasons=1)
        add("history (football-data.co.uk)", len(h) > 100, f"{len(h)} matches with odds columns: {int(h.odds_h.notna().sum()) if len(h) else 0}")
    except Exception as e:
        add("history (football-data.co.uk)", False, f"{type(e).__name__}: {e}")
    return out
