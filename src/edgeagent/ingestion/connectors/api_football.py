"""API-Football v3 (api-sports.io). Response layout follows the provider docs as I understand them; verify against a live key."""
from __future__ import annotations

import pandas as pd

from ...utils import make_fixture_id
from ..matching import normalize_team
from .http import HttpConnector

_DONE, _OPEN = {"FT", "AET", "PEN"}, {"NS", "TBD"}

class ApiFootball(HttpConnector):
    def __init__(self, key: str, cache=None, transport=None):
        super().__init__("https://v3.football.api-sports.io", {"x-apisports-key": key}, cache, transport=transport)

    async def fixtures(self, code: str, league_id: int, season: int, date_from: str, date_to: str, ttl: float = 900) -> pd.DataFrame:
        js = await self.get_json("/fixtures", {"league": league_id, "season": season, "from": date_from, "to": date_to}, ttl)
        rows = []
        for it in js.get("response", []):
            short = it["fixture"]["status"]["short"]
            status = "FT" if short in _DONE else "NS" if short in _OPEN else None
            if status is None:
                continue
            date = pd.to_datetime(it["fixture"]["date"], utc=True).tz_localize(None)
            h, a = normalize_team(it["teams"]["home"]["name"]), normalize_team(it["teams"]["away"]["name"])
            g = it.get("goals") or {}
            rows.append({"fixture_id": make_fixture_id(code, date, h, a), "league": code, "date": date, "home": h, "away": a,
                         "hg": g.get("home"), "ag": g.get("away"), "status": status, "af_id": it["fixture"]["id"]})
        return pd.DataFrame(rows, columns=["fixture_id", "league", "date", "home", "away", "hg", "ag", "status", "af_id"])

    async def lineups_available(self, af_id: int, ttl: float = 300) -> bool:
        js = await self.get_json("/fixtures/lineups", {"fixture": af_id}, ttl)
        return len(js.get("response", [])) >= 2

    async def injuries_known(self, af_id: int, ttl: float = 900) -> bool:
        await self.get_json("/injuries", {"fixture": af_id}, ttl)
        return True
