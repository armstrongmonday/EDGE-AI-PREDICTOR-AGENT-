from __future__ import annotations

import re
import unicodedata
import zlib
from datetime import UTC, datetime

STAGE_HOURS = [24, 6, 1, 0.5]
STAGE_LABELS = {0: "Waiting (>24h)", 1: "T-24h baseline", 2: "T-6h injuries/market", 3: "T-1h lineups", 4: "T-30m locked"}

def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)

def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def stable_rand(key: str) -> float:
    return (zlib.crc32(key.encode()) % 10_000) / 10_000

def stage_for_hours(h: float) -> int:
    s = 0
    for i, t in enumerate(STAGE_HOURS, 1):
        if h <= t:
            s = i
    return s

def make_fixture_id(code: str, date, home: str, away: str) -> str:
    return f"{code}-{date:%Y%m%d}-{slug(home)}-{slug(away)}"
