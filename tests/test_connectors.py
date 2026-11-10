"""Parsing tests use MOCKED responses written from the providers' documented layouts. They prove our parsing and
resilience logic, not that the providers still return this exact shape: verify once with real keys."""
import asyncio

import httpx

from edgeagent.ingestion.connectors import football_data
from edgeagent.ingestion.connectors.api_football import ApiFootball
from edgeagent.ingestion.connectors.odds_api import OddsApi
from edgeagent.ingestion.matching import match_event, normalize_team

AF = {"response": [{"fixture": {"id": 11, "date": "2026-11-14T15:00:00+00:00", "status": {"short": "NS"}},
                    "teams": {"home": {"name": "Manchester United"}, "away": {"name": "Wolves"}}, "goals": {"home": None, "away": None}},
                   {"fixture": {"id": 12, "date": "2026-11-14T15:00:00+00:00", "status": {"short": "PST"}},
                    "teams": {"home": {"name": "A"}, "away": {"name": "B"}}, "goals": {}}]}
ODDS = [{"home_team": "Manchester United", "away_team": "Wolverhampton Wanderers", "commence_time": "2026-11-14T15:00:00Z",
         "bookmakers": [{"key": k, "markets": [{"key": "h2h", "outcomes": [{"name": "Manchester United", "price": h},
                         {"name": "Draw", "price": 3.6}, {"name": "Wolverhampton Wanderers", "price": a}]}]} for k, h, a in [("x", 1.8, 4.6), ("y", 1.85, 4.4)]]}]

def test_api_football_parse_and_retry():
    calls = {"n": 0}
    def handler(req):
        calls["n"] += 1
        return httpx.Response(500) if calls["n"] == 1 else httpx.Response(200, json=AF)
    c = ApiFootball("k", transport=httpx.MockTransport(handler)); c.retries = 3
    import edgeagent.ingestion.connectors.http as h
    async def go():
        orig = h.asyncio.sleep
        h.asyncio.sleep = lambda s: orig(0)
        try:
            return await c.fixtures("EPL", 39, 2026, "2026-11-09", "2026-11-16")
        finally:
            h.asyncio.sleep = orig
    df = asyncio.run(go())
    assert calls["n"] == 2 and len(df) == 1 and df.status[0] == "NS" and df.home[0] == "Manchester United"

def test_odds_consensus_and_api_key_not_cached():
    seen = {}
    def handler(req):
        seen["q"] = str(req.url); return httpx.Response(200, json=ODDS)
    c = OddsApi("SECRET", transport=httpx.MockTransport(handler))
    ev = asyncio.run(c.snapshot("soccer_epl"))
    assert "SECRET" in seen["q"] and len(ev) == 1 and ev[0]["n_books"] == 2
    assert abs(ev[0]["odds"][0] - 1.825) < 1e-9
    assert all("SECRET" not in k for k in c.cache._mem)

def test_matching_across_providers():
    assert normalize_team("Man United") == normalize_team("Manchester United")
    cands = [{"home": "Manchester United", "away": "Wolverhampton Wanderers", "date": __import__("pandas").Timestamp("2026-11-14 15:00")}]
    assert match_event("Manchester United", "Wolves", __import__("pandas").Timestamp("2026-11-14 15:00"), cands) is not None
    assert match_event("Manchester United", "Chelsea", __import__("pandas").Timestamp("2026-11-14 15:00"), cands) is None

def test_football_data_csv_parse():
    csv = "Date,HomeTeam,AwayTeam,FTHG,FTAG,PSH,PSD,PSA,PSCH,PSCD,PSCA\n09/08/2025,Man United,Wolves,2,1,1.8,3.6,4.6,1.75,3.7,4.8\n"
    d = football_data.parse(csv, "EPL")
    assert d.fixture_id[0] == "EPL-20250809-manchester-united-wolverhampton-wanderers" and d.odds_h[0] == 1.8 and d.close_a[0] == 4.8
