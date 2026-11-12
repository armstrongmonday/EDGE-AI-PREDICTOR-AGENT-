"""The Odds API v4 (h2h, decimal). Consensus = median price per outcome across bookmakers; dispersion = std of de-vigged home prob."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...markets.devig import devig
from ..matching import normalize_team
from .http import HttpConnector


class OddsApi(HttpConnector):
    def __init__(self, key: str, cache=None, transport=None):
        super().__init__("https://api.the-odds-api.com", {}, cache, transport=transport)
        self.key = key

    async def sports(self, ttl: float = 3600) -> list[dict]:
        return await self.get_json("/v4/sports", {}, ttl, secret_params={"apiKey": self.key})  # free endpoint, no credits

    async def snapshot(self, sport_key: str, ttl: float = 900, regions: str = "eu,uk") -> list[dict]:
        js = await self.get_json(f"/v4/sports/{sport_key}/odds", {"regions": regions, "markets": "h2h", "oddsFormat": "decimal"},
                                 ttl, secret_params={"apiKey": self.key})
        out = []
        for ev in js:
            per_book = []
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk["key"] != "h2h":
                        continue
                    px = {o["name"]: o["price"] for o in mk["outcomes"]}
                    if ev["home_team"] in px and ev["away_team"] in px and "Draw" in px:
                        per_book.append([px[ev["home_team"]], px["Draw"], px[ev["away_team"]]])
            if not per_book:
                continue
            B = np.array(per_book, float)
            out.append({"home": normalize_team(ev["home_team"]), "away": normalize_team(ev["away_team"]),
                        "date": pd.to_datetime(ev["commence_time"], utc=True).tz_localize(None), "odds": np.median(B, 0).tolist(),
                        "dispersion": float(np.std(devig(B)[:, 0])) if len(B) > 1 else 0.0, "n_books": len(B)})
        return out
