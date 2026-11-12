"""Live source: football-data.co.uk (history + odds), API-Football (fixtures/lineups), The Odds API (current odds)."""
from __future__ import annotations

import asyncio
from datetime import timedelta

import numpy as np
import pandas as pd

from ...utils import utcnow
from ..cache import Cache
from ..matching import match_event
from . import football_data
from .api_football import ApiFootball
from .odds_api import OddsApi


class LiveSource:
    is_synthetic = False
    name = "live"

    def __init__(self, cfg, transport=None):
        if not (cfg.api_football_key and cfg.odds_api_key):
            raise RuntimeError("live source needs API_FOOTBALL_KEY and ODDS_API_KEY")
        cache = Cache(cfg.redis_url)
        self.cfg, self.ttl = cfg, cfg.sources["ttl"]
        self.apif, self.odds = ApiFootball(cfg.api_football_key, cache, transport), OddsApi(cfg.odds_api_key, cache, transport)
        self.meta: dict[str, dict] = {}
        self.unmatched: list[str] = []
        self.league_names = {k: v["name"] for k, v in cfg.leagues.items()}

    def now(self):
        return utcnow()

    def _codes(self, leagues):
        wanted = leagues or [c.strip() for c in self.cfg.enabled_leagues.split(",") if c.strip()]
        return [c for c in wanted if c in self.cfg.leagues]

    async def history(self, leagues=None, seasons: int = 3) -> pd.DataFrame:
        frames = []
        for code in self._codes(leagues):
            for s in football_data.season_codes(seasons, self.now()):
                try:
                    frames.append(await football_data.fetch(self.cfg.leagues[code]["fd"], code, s))
                except Exception as e:  # missing season file etc.
                    self.unmatched.append(f"history {code} {s}: {type(e).__name__}")
        return pd.concat(frames).drop_duplicates("fixture_id") if frames else pd.DataFrame()

    async def upcoming(self, leagues=None, days: int = 3) -> pd.DataFrame:
        now, frames = self.now(), []
        season = now.year if now.month >= 7 else now.year - 1
        for code in self._codes(leagues):
            lg = self.cfg.leagues[code]
            fx = await self.apif.fixtures(code, lg["af"], season, f"{now:%Y-%m-%d}", f"{now + timedelta(days=days):%Y-%m-%d}", self.ttl["fixtures"])
            fx = fx[(fx.status == "NS") & (fx.date > now)].copy()
            ev = await self.odds.snapshot(lg["odds"], self.ttl["odds"], self.cfg.odds_regions)
            fx[["odds_h", "odds_d", "odds_a"]] = np.nan
            for i, r in fx.iterrows():
                m = match_event(r.home, r.away, r.date, ev)
                if m is None:
                    self.unmatched.append(f"odds not matched: {r.home} v {r.away}")
                    continue
                fx.loc[i, ["odds_h", "odds_d", "odds_a"]] = m["odds"]
                self.meta[r.fixture_id] = {"dispersion": m["dispersion"], "odds_ts": now, "af_id": int(r.af_id), "kickoff_diff_s": abs((m["date"] - r.date).total_seconds())}
            for _, r in fx.iterrows():
                self.meta.setdefault(r.fixture_id, {"af_id": int(r.af_id)})
            frames.append(fx.drop(columns=["af_id"]))
        return pd.concat(frames) if frames else pd.DataFrame()

    async def results(self, ids) -> pd.DataFrame:
        now, frames = self.now(), []
        for code in {i.split("-")[0] for i in ids}:
            if code not in self.cfg.leagues:
                continue
            season = now.year if now.month >= 7 else now.year - 1
            fx = await self.apif.fixtures(code, self.cfg.leagues[code]["af"], season, f"{now - timedelta(days=4):%Y-%m-%d}", f"{now:%Y-%m-%d}", 600)
            frames.append(fx[(fx.status == "FT") & fx.fixture_id.isin(ids)].drop(columns=["af_id"]))
        return pd.concat(frames) if frames else pd.DataFrame()

    async def context(self, fixture_id: str, hours_to_ko: float) -> dict:
        m = self.meta.get(fixture_id, {})
        lineups = False
        if hours_to_ko <= 1.5 and "af_id" in m:
            try:
                lineups = await self.apif.lineups_available(m["af_id"], self.ttl["lineups"])
            except Exception:
                lineups = False
        age = (self.now() - m["odds_ts"]).total_seconds() if "odds_ts" in m else 1e9
        return {"odds_age_s": age, "odds_ttl_s": float(self.ttl["odds"]), "lineups": lineups, "injuries": "af_id" in m,
                "dispersion": m.get("dispersion", 0.0), "kickoff_diff_s": m.get("kickoff_diff_s", 0.0)}

    def status(self) -> list[dict]:
        loop_t = asyncio.get_event_loop().time()
        row = lambda name, c, ttl: {"source": name, "freshness_s": int(loop_t - c.last_ok) if c.last_ok else None, "ttl_s": ttl,
                                    "completeness": 1.0 if c.last_ok else 0.0, "status": "OK" if c.last_ok else "no successful call yet"}
        return [row("API-Football", self.apif, self.ttl["fixtures"]), row("The Odds API", self.odds, self.ttl["odds"]),
                {"source": "football-data.co.uk", "freshness_s": None, "ttl_s": self.ttl["history"], "completeness": 1.0, "status": "history"}]
