"""Orchestrator: ingest -> features -> models -> market -> risk-aware record -> settle -> Brainbox learning."""
from __future__ import annotations

import json
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from .brainbox.brainbox import Brainbox
from .config import Settings
from .features.state import build_features
from .ingestion.quality_score import compute_dqs
from .markets.devig import devig
from .models.bundle import fit_bundle, predict_bundle
from .storage import Store
from .utils import STAGE_LABELS, stage_for_hours

log = logging.getLogger("edgeagent")
RETRAIN_EVERY = 200

def make_source(cfg: Settings, store: Store | None = None):
    if cfg.source == "live":
        from .ingestion.connectors.live import LiveSource
        return LiveSource(cfg)
    from .ingestion.connectors.synthetic import SyntheticSource
    anchor = store.kv_get("sim_anchor") if store else None
    src = SyntheticSource(now=pd.Timestamp(anchor).to_pydatetime() if anchor else None)
    if store:  # persist the simulated clock so restarts continue the same synthetic world
        store.kv_set("sim_anchor", str(src._now))
        now = store.kv_get("sim_now")
        if now:
            src._now = pd.Timestamp(now).to_pydatetime()
    return src

class Pipeline:
    def __init__(self, cfg: Settings | None = None, store: Store | None = None, source=None):
        self.cfg = cfg or Settings()
        self.store = store or Store(self.cfg.db_path)
        self.source = source or make_source(self.cfg, self.store)
        self.bb = Brainbox(self.cfg.brainbox, self.store)
        self.bundles, self.state, self.ref, self.seen, self.since_train = {}, None, None, set(), 0
        self.active = self.store.kv_get("active_config", "A")
        self.market_weight = self.store.kv_get("market_weight", 0.5)
        self.suppressed = set(self.store.kv_get("suppressed", []))
        self.version = self.store.kv_get("version", "")

    @property
    def other(self) -> str:
        return "B" if self.active == "A" else "A"

    @property
    def active_bundle(self):
        return self.bundles[self.active]

    def advance(self, hours: float):
        """Synthetic source only: move the simulated clock and persist it."""
        self.source.advance(hours)
        self.store.kv_set("sim_now", str(self.source.now()))

    # ---------- data + training
    async def backfill(self, leagues=None, seasons: int = 3) -> int:
        return self.store.upsert_matches(await self.source.history(leagues, seasons))

    def train(self) -> None:
        M = self.store.matches(finished=True)
        X, self.state = build_features(M)
        self.seen, self.ref = set(M.fixture_id), X.tail(2000).reset_index(drop=True)
        self.bundles = {k: fit_bundle(M, X, c, self.cfg.brainbox) for k, c in self.cfg.model_configs.items()}
        self.version = f"ens-{self.active}-{M.date.max():%Y%m%d}"
        w = self.store.kv_get("weights")
        b = self.active_bundle
        if w and list(w) == b.ensemble.names:
            b.ensemble.set_weights([w[k] for k in b.ensemble.names])
        self.store.kv_set("train_report", {k: v.report for k, v in self.bundles.items()})
        self.store.kv_set("version", self.version)
        self.since_train = 0

    async def ensure_ready(self):
        if len(self.store.matches(finished=True)) < 100:
            await self.backfill()
        if not self.bundles:
            self.train()

    def promote(self):
        self.active = self.other
        self.store.kv_set("active_config", self.active)
        self.store.kv_set("promoted_at", str(self.source.now()))
        self.version = f"ens-{self.active}-{self.version.rsplit('-', 1)[-1]}"

    # ---------- prediction
    def predict_one(self, r, stage: int, hours: float, ctx: dict, now) -> dict:
        b, ch = self.active_bundle, self.bundles[self.other]
        feats = self.state.features(r.home, r.away, r.date)
        members, raw, model_p = predict_bundle(b, r.home, r.away, feats)
        _, _, chall_p = predict_bundle(ch, r.home, r.away, feats)
        odds = np.array([r.odds_h, r.odds_d, r.odds_a], float)
        has_odds = not np.isnan(odds).any()
        mkt = devig(odds) if has_odds else None
        w = self.market_weight if has_odds else 0.0
        p = (1 - w) * model_p + w * mkt if has_odds else model_p
        dq = compute_dqs(min(feats["n_h"], feats["n_a"]), has_odds, ctx["odds_age_s"], ctx["odds_ttl_s"], hours,
                         ctx["lineups"], ctx["injuries"], ctx["dispersion"], ctx["kickoff_diff_s"])
        gate = self.cfg.brainbox["min_dqs"]
        reason = None
        if dq["score"] < gate:
            reason = f"data quality {dq['score']:.2f} < {gate}"
        elif r.league in self.suppressed:
            reason = "guardrail: league calibration degraded"
        ok = reason is None
        lh, la = b.models["poisson_ad"].goals(r.home, r.away)
        ci = max(0.01, float(1.96 * np.std(np.stack(list(members.values())), axis=0).max()))
        e = (p - mkt) if has_odds else None
        g = lambda arr, i: float(arr[i]) if arr is not None and ok else None
        return {"fixture_id": r.fixture_id, "stage": stage, "ts": now, "model_version": self.version, "predictable": ok,
                "locked": stage == 4, "dqs": dq["score"], "reason": reason,
                **{f"p_{c}": g(p, i) for i, c in enumerate("hda")}, **{f"model_{c}": g(model_p, i) for i, c in enumerate("hda")},
                **{f"mkt_{c}": float(mkt[i]) if has_odds else None for i, c in enumerate("hda")},
                **{f"odds_{c}": float(odds[i]) if has_odds else None for i, c in enumerate("hda")},
                **{f"edge_{c}": g(e, i) for i, c in enumerate("hda")}, "ci_width": ci, "xg_h": lh, "xg_a": la,
                **{f"chall_{c}": g(chall_p, i) for i, c in enumerate("hda")},
                "details": json.dumps({"members": {k: v.tolist() for k, v in members.items()}, "raw": raw.tolist(),
                                       "feats": feats, "config": self.active, "dq": dq})}

    async def run_once(self, days: int = 3) -> list[dict]:
        if not self.bundles:
            await self.ensure_ready()
        now = self.source.now()
        self.store.upsert_matches(await self.source.upcoming(days=days))
        fx = self.store.matches(finished=False, after=now)
        fx = fx[fx.date <= now + timedelta(days=days)]
        out = []
        for r in fx.itertuples():
            hours = (r.date - now).total_seconds() / 3600
            stage = stage_for_hours(hours)
            if stage == 0 or self.store.has_prediction(r.fixture_id, stage):
                continue
            rec = self.predict_one(r, stage, hours, await self.source.context(r.fixture_id, hours), now)
            if self.store.save_prediction(rec):
                out.append(rec)
        return out

    # ---------- settle + learn
    async def settle(self) -> dict:
        now = self.source.now()
        ids = self.store.unsettled_ids(now)
        if ids:
            self.store.upsert_matches(await self.source.results(ids))
            for fid in ids:
                self.store.set_close_from_latest(fid)
        M = self.store.matches(finished=True)
        new = M[~M.fixture_id.isin(self.seen)]
        for r in new.itertuples():
            self.state.update(r.home, r.away, int(r.hg), int(r.ag), r.date)
            self.seen.add(r.fixture_id)
        self.since_train += len(new)
        scored = self.bb.score()
        actions = self.bb.update(self).get("actions", []) if scored else []
        if self.since_train >= RETRAIN_EVERY:
            self.train()
            actions.append("retrained all bundles")
        return {"new_results": len(new), "scored": scored, "actions": actions}

    # ---------- API views
    def fixtures_view(self) -> list[dict]:
        now = self.source.now()
        df = self.store.predictions_latest(now)
        names = {**{k: v["name"] for k, v in self.cfg.leagues.items()}, **getattr(self.source, "league_names", {})}
        rows = []
        for r in df.to_dict("records"):
            hours = (r["date"] - now).total_seconds() / 3600
            rec = {k: (None if pd.isna(v) else v) for k, v in r.items() if k != "details"}
            rec.update(league=names.get(r["league"], r["league"]), kickoff=pd.Timestamp(r["date"]).isoformat() + "Z",
                       hours_to_ko=hours, stage=STAGE_LABELS[int(r["stage"])], predictable=bool(r["predictable"]),
                       p_home=rec["p_h"], p_draw=rec["p_d"], p_away=rec["p_a"],
                       odds_home=rec["odds_h"], odds_draw=rec["odds_d"], odds_away=rec["odds_a"],
                       mkt_home=rec["mkt_h"], mkt_draw=rec["mkt_d"], mkt_away=rec["mkt_a"],
                       edge_home=rec["edge_h"], edge_draw=rec["edge_d"], edge_away=rec["edge_a"],
                       xg_home=rec["xg_h"], xg_away=rec["xg_a"])
            rec.pop("date", None)
            rows.append(rec)
        return rows
