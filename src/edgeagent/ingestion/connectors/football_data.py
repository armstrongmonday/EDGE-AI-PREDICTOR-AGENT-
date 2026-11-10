"""football-data.co.uk historical CSVs: results plus Pinnacle/Bet365 odds. Column names follow the site's published notes."""
from __future__ import annotations

import io

import httpx
import pandas as pd

from ...utils import make_fixture_id
from ..matching import normalize_team


def season_codes(n: int, now) -> list[str]:
    y = now.year if now.month >= 7 else now.year - 1
    return [f"{(y - i) % 100:02d}{(y - i + 1) % 100:02d}" for i in range(n)]

def parse(csv_text: str, code: str) -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(csv_text)).dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    date = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
    first = lambda *cols: next((d[c] for c in cols if c in d.columns), pd.Series(float("nan"), index=d.index))
    out = pd.DataFrame({"date": date, "home": d.HomeTeam.map(normalize_team), "away": d.AwayTeam.map(normalize_team),
                        "hg": d.FTHG.astype(int), "ag": d.FTAG.astype(int), "league": code, "status": "FT",
                        "odds_h": first("PSH", "B365H", "AvgH"), "odds_d": first("PSD", "B365D", "AvgD"), "odds_a": first("PSA", "B365A", "AvgA"),
                        "close_h": first("PSCH", "B365CH", "AvgCH"), "close_d": first("PSCD", "B365CD", "AvgCD"),
                        "close_a": first("PSCA", "B365CA", "AvgCA")}).dropna(subset=["date"])
    out["fixture_id"] = [make_fixture_id(code, r.date, r.home, r.away) for r in out.itertuples()]
    return out

async def fetch(div: str, code: str, season: str, client: httpx.AsyncClient | None = None) -> pd.DataFrame:
    async with (client or httpx.AsyncClient(timeout=30)) as c:
        r = await c.get(f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv")
        r.raise_for_status()
    return parse(r.text, code)
