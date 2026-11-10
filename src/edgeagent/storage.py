"""DuckDB store. One shared connection guarded by a lock (DuckDB connections are not thread-safe)."""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

MATCH_COLS = ["fixture_id", "league", "date", "home", "away", "hg", "ag", "status",
              "odds_h", "odds_d", "odds_a", "close_h", "close_d", "close_a"]
PRED_COLS = ["fixture_id", "stage", "ts", "model_version", "predictable", "locked", "dqs", "reason",
             "p_h", "p_d", "p_a", "model_h", "model_d", "model_a", "mkt_h", "mkt_d", "mkt_a",
             "odds_h", "odds_d", "odds_a", "edge_h", "edge_d", "edge_a", "ci_width", "xg_h", "xg_a",
             "chall_h", "chall_d", "chall_a", "details"]

DDL = [
    """CREATE TABLE IF NOT EXISTS matches(fixture_id VARCHAR PRIMARY KEY, league VARCHAR, date TIMESTAMP,
       home VARCHAR, away VARCHAR, hg INTEGER, ag INTEGER, status VARCHAR, odds_h DOUBLE, odds_d DOUBLE,
       odds_a DOUBLE, close_h DOUBLE, close_d DOUBLE, close_a DOUBLE)""",
    """CREATE TABLE IF NOT EXISTS predictions(fixture_id VARCHAR, stage INTEGER, ts TIMESTAMP, model_version VARCHAR,
       predictable BOOLEAN, locked BOOLEAN, dqs DOUBLE, reason VARCHAR, p_h DOUBLE, p_d DOUBLE, p_a DOUBLE,
       model_h DOUBLE, model_d DOUBLE, model_a DOUBLE, mkt_h DOUBLE, mkt_d DOUBLE, mkt_a DOUBLE,
       odds_h DOUBLE, odds_d DOUBLE, odds_a DOUBLE, edge_h DOUBLE, edge_d DOUBLE, edge_a DOUBLE,
       ci_width DOUBLE, xg_h DOUBLE, xg_a DOUBLE, chall_h DOUBLE, chall_d DOUBLE, chall_a DOUBLE,
       details VARCHAR, PRIMARY KEY (fixture_id, stage))""",
    """CREATE TABLE IF NOT EXISTS scores(fixture_id VARCHAR, model VARCHAR, date TIMESTAMP, logloss DOUBLE,
       brier DOUBLE, correct INTEGER, PRIMARY KEY (fixture_id, model))""",
    "CREATE TABLE IF NOT EXISTS kv(key VARCHAR PRIMARY KEY, value VARCHAR)",
]

def _py(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v

_SETTLED = """SELECT * FROM (
    SELECT p.*, m.hg AS hg, m.ag AS ag, m.date AS mdate, m.league AS mleague,
           row_number() OVER (PARTITION BY p.fixture_id ORDER BY p.stage DESC) AS rn
    FROM predictions p JOIN matches m ON p.fixture_id = m.fixture_id
    WHERE m.status = 'FT' AND p.predictable) WHERE rn = 1"""

class Store:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.con = duckdb.connect(path)
        for ddl in DDL:
            self.con.execute(ddl)

    def _x(self, sql, params=None):
        with self._lock:
            self.con.execute(sql, params or [])

    def _df(self, sql, params=None) -> pd.DataFrame:
        with self._lock:
            return self.con.execute(sql, params or []).df()

    # ---- matches
    def upsert_matches(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        d = df.reindex(columns=MATCH_COLS).drop_duplicates("fixture_id", keep="last").copy()
        d["hg"], d["ag"] = d["hg"].astype("Int64"), d["ag"].astype("Int64")
        co = lambda c: f"{c}=COALESCE(excluded.{c}, matches.{c})"
        sets = ", ".join(co(c) for c in ["hg", "ag", "odds_h", "odds_d", "odds_a", "close_h", "close_d", "close_a"])
        with self._lock:
            self.con.register("_m", d)
            self.con.execute(f"""INSERT INTO matches SELECT * FROM _m ON CONFLICT (fixture_id) DO UPDATE SET
                league=excluded.league, date=excluded.date,
                status=CASE WHEN matches.status='FT' THEN 'FT' ELSE excluded.status END, {sets}""")
            self.con.unregister("_m")
        return len(d)

    def matches(self, finished: bool | None = None, after=None) -> pd.DataFrame:
        q, p = "SELECT * FROM matches WHERE 1=1", []
        if finished is True:
            q += " AND status='FT' AND hg IS NOT NULL"
        elif finished is False:
            q += " AND status<>'FT'"
        if after is not None:
            q += " AND date > ?"; p.append(after)
        return self._df(q + " ORDER BY date, fixture_id", p)

    def set_close_from_latest(self, fixture_id: str):
        self._x("""UPDATE matches SET close_h=COALESCE(close_h, odds_h), close_d=COALESCE(close_d, odds_d),
                   close_a=COALESCE(close_a, odds_a) WHERE fixture_id=?""", [fixture_id])

    # ---- predictions
    def locked(self, fixture_id: str) -> bool:
        return bool(self._df("SELECT count(*) AS n FROM predictions WHERE fixture_id=? AND locked", [fixture_id]).n[0])

    def has_prediction(self, fixture_id: str, stage: int) -> bool:
        return bool(self._df("SELECT count(*) AS n FROM predictions WHERE fixture_id=? AND stage>=?",
                             [fixture_id, stage]).n[0])

    def save_prediction(self, rec: dict) -> bool:
        """Locked predictions are immutable: nothing is written once a fixture has a locked row."""
        if self.locked(rec["fixture_id"]):
            return False
        vals = [_py(rec.get(c)) for c in PRED_COLS]
        self._x(f"INSERT OR REPLACE INTO predictions ({','.join(PRED_COLS)}) VALUES ({','.join('?' * len(PRED_COLS))})", vals)
        return True

    def predictions_latest(self, after) -> pd.DataFrame:
        return self._df("""SELECT p.*, m.league, m.home, m.away, m.date FROM predictions p
            JOIN matches m ON p.fixture_id=m.fixture_id WHERE m.date > ?
            QUALIFY row_number() OVER (PARTITION BY p.fixture_id ORDER BY p.stage DESC) = 1
            ORDER BY m.date""", [after])

    def get_prediction(self, fixture_id: str) -> pd.DataFrame:
        return self._df("""SELECT p.*, m.league, m.home, m.away, m.date FROM predictions p
            JOIN matches m ON p.fixture_id=m.fixture_id WHERE p.fixture_id=? ORDER BY stage DESC LIMIT 1""", [fixture_id])

    def unsettled_ids(self, before) -> list[str]:
        d = self._df("""SELECT DISTINCT m.fixture_id FROM matches m JOIN predictions p ON p.fixture_id=m.fixture_id
                        WHERE m.status<>'FT' AND m.date < ?""", [before])
        return d.fixture_id.tolist()

    def settled_predictions(self, limit: int | None = None, unscored_only=False) -> pd.DataFrame:
        q = _SETTLED + (" AND fixture_id NOT IN (SELECT fixture_id FROM scores WHERE model='final')" if unscored_only else "")
        q += " ORDER BY mdate DESC" + (f" LIMIT {int(limit)}" if limit else "")
        return self._df(q).iloc[::-1].reset_index(drop=True)

    # ---- scores
    def save_scores(self, rows: list[tuple]):
        if rows:
            with self._lock:
                self.con.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?)", [tuple(_py(v) for v in r) for r in rows])

    def scores(self) -> pd.DataFrame:
        return self._df("SELECT * FROM scores")

    def weekly_logloss(self) -> pd.DataFrame:
        return self._df("""SELECT date_trunc('week', date) AS wk, model, avg(logloss) AS ll, count(*) AS n
                           FROM scores GROUP BY 1, 2 ORDER BY 1""")

    # ---- kv
    def kv_get(self, key, default=None):
        d = self._df("SELECT value FROM kv WHERE key=?", [key])
        return json.loads(d.value[0]) if len(d) else default

    def kv_set(self, key, value):
        self._x("INSERT OR REPLACE INTO kv VALUES (?,?)", [key, json.dumps(value, default=_py)])
