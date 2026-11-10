"""The Brainbox: scores every settled prediction, then adapts weights, market trust, calibration, drift
monitoring, the champion/challenger gate and per-league guardrails. It improves reliability; it cannot
create information the data does not contain."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..backtest.metrics import logloss
from ..features.state import FEATURES
from .champion_challenger import compare
from .drift import PageHinkley, drift_report
from .error_analyst import classify, summarize

DRIFT_FEATURES = [f for f in FEATURES if f not in ("n_h", "n_a")]  # counters trend by construction
from ..models.base import outcome
from ..models.calibration import ece, reliability
from ..models.ensemble import WeightedEnsemble


def _arr(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]].to_numpy(float)

def _ll_rows(P, y) -> np.ndarray:
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))

class Brainbox:
    def __init__(self, cfg: dict, store):
        self.cfg, self.store = cfg, store

    # ---------- scoring
    def score(self) -> int:
        W = self.store.settled_predictions(unscored_only=True)
        if W.empty:
            return 0
        y = outcome(W.rename(columns={}))
        rows, causes = [], []
        det = [json.loads(d) for d in W.details]
        sets = {"ensemble": _arr(W, "model"), "final": _arr(W, "p"), "market": _arr(W, "mkt"), "challenger": _arr(W, "chall")}
        for n in det[0]["members"]:
            sets[n] = np.array([d["members"][n] for d in det])
        for name, P in sets.items():
            ok = ~np.isnan(P).any(1)
            if not ok.any():
                continue
            ll = _ll_rows(np.where(ok[:, None], P, 1 / 3), y)
            br = ((np.where(ok[:, None], P, 1 / 3) - np.eye(3)[y]) ** 2).sum(1)
            cor = (np.where(ok[:, None], P, 1 / 3).argmax(1) == y).astype(int)
            for i in np.flatnonzero(ok):
                rows.append((W.fixture_id[i], name, W.mdate[i], float(ll[i]), float(br[i]), int(cor[i])))
        P, M = sets["final"], sets["market"]
        for i in range(len(W)):
            ll = float(-np.log(max(P[i, y[i]], 1e-12)))
            c = classify(ll, P[i, y[i]], None if np.isnan(M[i]).any() else M[i, y[i]], float(W.dqs[i]))
            if c:
                causes.append(c)
        self.store.save_scores(rows)
        if causes:
            prev = self.store.kv_get("error_causes", [])
            self.store.kv_set("error_causes", (prev + causes)[-500:])
        return len(W)

    # ---------- adaptation
    def update(self, pipe) -> dict:
        c = self.cfg
        W = self.store.settled_predictions(limit=c["rolling_window_matches"])
        if W.empty:
            return {}
        y, det = outcome(W), [json.loads(d) for d in W.details]
        names, actions = list(det[0]["members"]), []
        b = pipe.bundles[pipe.active]
        n = len(W)

        if n >= 30 and list(b.ensemble.names) == names:
            mean_ll = {k: logloss(np.array([d["members"][k] for d in det]), y) for k in names}
            b.ensemble.set_weights(WeightedEnsemble.softmax_weights(mean_ll, names, c["weight_temperature"]))
            self.store.kv_set("weights", b.ensemble.weights())

        F, Mk, Pm = _arr(W, "p"), _arr(W, "mkt"), _arr(W, "model")
        has_mkt = ~np.isnan(Mk).any(1)
        if has_mkt.sum() >= 100:
            grid = np.linspace(0, 1, 21)
            lls = [logloss((1 - w) * Pm[has_mkt] + w * Mk[has_mkt], y[has_mkt]) for w in grid]
            new_w = float(grid[int(np.argmin(lls))])
            if abs(new_w - pipe.market_weight) > 1e-9:
                actions.append(f"market weight {pipe.market_weight:.2f} -> {new_w:.2f}")
            pipe.market_weight = new_w
            self.store.kv_set("market_weight", new_w)

        last = self.store.kv_get("recal_n", 0)
        if n >= c["calibration_min_samples"] and n - last >= c["recalibrate_every_matches"] or (not b.calibrator.fitted and n >= c["calibration_min_samples"]):
            b.calibrator.fit(np.array([d["raw"] for d in det]), y)
            self.store.kv_set("recal_n", n)
            actions.append(f"recalibrated on last {n} matches")

        cur = pd.DataFrame([d["feats"] for d in det]).tail(200)
        drift = drift_report(pipe.ref, cur, DRIFT_FEATURES, c["psi_threshold"], c["ks_alpha"]) if pipe.ref is not None else []
        ph = PageHinkley()
        ph_alarm = any(ph.update(v) for v in _ll_rows(F, y))
        now_drift = sorted(d["feature"] for d in drift if d["status"].startswith("DRIFT"))
        if now_drift and now_drift != (self.store.kv_get("brainbox_status", {}) or {}).get("drifting", []):
            actions.append(f"feature drift detected ({', '.join(now_drift)}): recalibration/retrain recommended")

        promoted_at = self.store.kv_get("promoted_at")
        V = W[pd.to_datetime(W.ts) > pd.Timestamp(promoted_at)] if promoted_at else W
        okc = ~np.isnan(_arr(V, "chall")).any(1) if len(V) else np.array([], bool)
        cc_row = {"model": f"config {pipe.other} (challenger)", "matches_shadowed": 0, "d_logloss": 0.0, "p_value": None, "promote": False, "note": "no data"}
        if len(V) and okc.any():
            yv = outcome(V)
            r = compare(_ll_rows(_arr(V, "model")[okc], yv[okc]), _ll_rows(_arr(V, "chall")[okc], yv[okc]),
                        c["min_matches_to_promote"], c["promotion_p_value"], True)
            cc_row.update(r)
            if r["promote"]:
                pipe.promote()
                actions.append(f"promoted config {pipe.active} to champion (p={r['p_value']:.3f})")

        by_league, suppressed = {}, set()
        for lg, idx in W.groupby("mleague").groups.items():
            ix = np.array(list(idx))
            if len(ix) >= c["min_ece_samples"] // 3:
                e = ece(F[ix], y[ix]); by_league[lg] = {"ece": round(e, 4), "n": len(ix)}
                if len(ix) >= c["min_ece_samples"] and e > c["max_ece"]:
                    suppressed.add(lg)
        pipe.suppressed = suppressed
        self.store.kv_set("suppressed", sorted(suppressed))
        glob = ece(F, y) if n >= c["min_ece_samples"] else None

        status = {"n_window": n, "ece": glob, "drift": drift, "ph_alarm": bool(ph_alarm), "drifting": now_drift,
                  "cc": [{"model": f"config {pipe.active} (champion)", "matches_shadowed": 0, "d_logloss": 0.0, "p_value": None, "promote": False, "note": "-"}, cc_row],
                  "guardrail": {"max_ece": c["max_ece"], "ece_by_league": by_league, "suppressed": sorted(suppressed)},
                  "errors": summarize(self.store.kv_get("error_causes", [])), "actions": actions}
        log = self.store.kv_get("actions_log", []) + [a for a in actions]
        self.store.kv_set("actions_log", log[-50:])
        self.store.kv_set("brainbox_status", status)
        return status

    # ---------- API view
    def status(self, pipe=None) -> dict:
        st = self.store.kv_get("brainbox_status", {}) or {}
        wk = self.store.weekly_logloss()
        series = {m: g.ll.round(5).tolist() for m, g in wk.groupby("model")} if len(wk) else {}
        dates = sorted(wk.wk.dt.strftime("%Y-%m-%d").unique().tolist()) if len(wk) else []
        full = {m: [None] * len(dates) for m in series}
        for m, g in wk.groupby("model"):
            for d, v in zip(g.wk.dt.strftime("%Y-%m-%d"), g.ll):
                full[m][dates.index(d)] = round(float(v), 5)
        W = self.store.settled_predictions(limit=self.cfg["rolling_window_matches"])
        calib = {"pred": [], "obs": [], "n": []}
        if len(W):
            p, o, cnt = reliability(_arr(W, "p"), outcome(W))
            calib = {"pred": p.round(4).tolist(), "obs": o.round(4).tolist(), "n": cnt.tolist()}
        return {"logloss": {"dates": dates, "series": full},
                "weights": pipe.active_bundle.ensemble.weights() if pipe is not None and pipe.bundles else self.store.kv_get("weights", {}), "market_weight": self.store.kv_get("market_weight", 0.5),
                "calib": calib, "ece": st.get("ece"), "n_window": st.get("n_window", 0), "drift": st.get("drift", []),
                "cc": st.get("cc", []), "guardrail": st.get("guardrail", {"max_ece": self.cfg["max_ece"], "suppressed": []}),
                "errors": st.get("errors", {}), "actions": self.store.kv_get("actions_log", [])[-10:],
                "ph_alarm": st.get("ph_alarm", False), "train_report": self.store.kv_get("train_report", {}),
                "active_config": self.store.kv_get("active_config", "A")}
