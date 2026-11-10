"""A Bundle = fitted members + ensemble weights + calibrator. Two bundles (A/B) run as champion/challenger."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.metrics import brier, logloss
from ..markets.devig import devig
from .base import outcome
from .calibration import Calibrator, ece
from .elo_model import EloModel
from .ensemble import WeightedEnsemble
from .ml import gbm, logistic
from .poisson import PoissonAD


def make_models(cfg: dict):
    return [EloModel(), PoissonAD(xi=cfg["xi"]), logistic(cfg["lr_C"]), gbm(cfg["gbm_depth"])]

def market_probs(m: pd.DataFrame) -> np.ndarray:
    O = m[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    out = np.full(O.shape, np.nan)
    ok = ~np.isnan(O).any(1)
    if ok.any():
        out[ok] = devig(O[ok])
    return out

@dataclass
class Bundle:
    cfg: dict
    models: dict
    ensemble: WeightedEnsemble
    calibrator: Calibrator
    report: dict
    trained_until: pd.Timestamp

def fit_bundle(matches: pd.DataFrame, X: pd.DataFrame, cfg: dict, bb: dict) -> Bundle:
    n = len(matches)
    if n < 100:
        raise ValueError(f"need at least 100 finished matches, have {n}")
    i1, i2 = int(n * 0.7), int(n * 0.85)
    M, y = matches.reset_index(drop=True), outcome(matches)
    X = X.reset_index(drop=True)
    first = {m.name: m.fit(M.iloc[:i1], X.iloc[:i1]) for m in make_models(cfg)}
    names = list(first)
    cal = {k: v.predict_many(M.iloc[i1:i2], X.iloc[i1:i2]) for k, v in first.items()}
    mean_ll = {k: logloss(P, y[i1:i2]) for k, P in cal.items()}
    ens = WeightedEnsemble(names)
    ens.set_weights(WeightedEnsemble.softmax_weights(mean_ll, names, bb["weight_temperature"]))
    raw = ens.combine(cal)
    calibrator = Calibrator(bb["calibration_min_samples"]).fit(raw, y[i1:i2])
    # honest hold-out report from models fitted on [:i1]
    test = {k: v.predict_many(M.iloc[i2:], X.iloc[i2:]) for k, v in first.items()}
    ens_t = calibrator.transform(ens.combine(test))
    yt = y[i2:]
    mk = market_probs(M.iloc[i2:]); ok = ~np.isnan(mk).any(1)
    report = {"n_test": len(yt), "uniform_logloss": float(np.log(3)),
              "members": {k: {"logloss": logloss(P, yt), "brier": brier(P, yt)} for k, P in test.items()},
              "ensemble": {"logloss": logloss(ens_t, yt), "brier": brier(ens_t, yt), "ece": ece(ens_t, yt)}}
    if ok.sum() > 20:
        report["market"] = {"logloss": logloss(mk[ok], yt[ok]), "brier": brier(mk[ok], yt[ok])}
        report["ensemble_same_rows"] = {"logloss": logloss(ens_t[ok], yt[ok])}
    final = {m.name: m.fit(M.iloc[:i2], X.iloc[:i2]) for m in make_models(cfg)}
    return Bundle(cfg, final, ens, calibrator, report, M.date.iloc[i2 - 1])

def predict_many_bundle(b: Bundle, M: pd.DataFrame, X: pd.DataFrame):
    members = {k: m.predict_many(M, X) for k, m in b.models.items()}
    raw = b.ensemble.combine(members)
    return members, raw, b.calibrator.transform(raw)

def predict_bundle(b: Bundle, home: str, away: str, feats: dict):
    M, X = pd.DataFrame({"home": [home], "away": [away]}), pd.DataFrame([feats])
    members, raw, cal = predict_many_bundle(b, M, X)
    return {k: v[0] for k, v in members.items()}, raw[0], cal[0]
