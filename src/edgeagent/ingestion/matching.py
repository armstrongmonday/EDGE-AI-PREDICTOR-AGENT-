"""Team-name normalisation and fuzzy event matching across providers. Extend ALIASES for other leagues;
unmatched names are surfaced rather than silently guessed."""
from __future__ import annotations

import re
import unicodedata
from datetime import timedelta
from difflib import SequenceMatcher

ALIASES = {"man united": "manchester united", "man utd": "manchester united", "man city": "manchester city",
           "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur", "newcastle": "newcastle united",
           "wolves": "wolverhampton wanderers", "west ham": "west ham united", "nottm forest": "nottingham forest",
           "nott m forest": "nottingham forest", "leeds": "leeds united", "brighton": "brighton and hove albion",
           "leicester": "leicester city", "norwich": "norwich city", "ipswich": "ipswich town", "luton": "luton town",
           "sheffield utd": "sheffield united", "west brom": "west bromwich albion", "atletico madrid": "atletico",
           "ath madrid": "atletico", "ath bilbao": "athletic club", "athletic bilbao": "athletic club",
           "betis": "real betis", "sociedad": "real sociedad", "inter milan": "inter", "internazionale": "inter",
           "ac milan": "milan", "bayern munich": "bayern", "m gladbach": "monchengladbach",
           "borussia monchengladbach": "monchengladbach", "paris saint germain": "paris sg", "psg": "paris sg"}

def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(re.sub(r"\b(fc|afc|cf|sc|ssc|calcio)\b", " ", s).split())

def normalize_team(name: str) -> str:
    c = _clean(name)
    return ALIASES.get(c, c).title()

def similar(a: str, b: str) -> float:
    a, b = normalize_team(a).lower(), normalize_team(b).lower()
    if a == b:
        return 1.0
    return max(SequenceMatcher(None, a, b).ratio(), 0.9 if (a in b or b in a) else 0.0)

def match_event(home: str, away: str, date, candidates, tol_hours: float = 4, cutoff: float = 0.78):
    """candidates: iterable of dicts with home, away, date. Returns the best candidate or None."""
    best, score = None, 0.0
    for c in candidates:
        if abs((c["date"] - date) / timedelta(hours=1)) > tol_hours:
            continue
        s = min(similar(home, c["home"]), similar(away, c["away"]))
        if s > score:
            best, score = c, s
    return best if score >= cutoff else None
