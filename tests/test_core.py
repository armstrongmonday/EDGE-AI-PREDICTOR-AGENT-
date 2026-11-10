import numpy as np
import pandas as pd

from edgeagent.brainbox.champion_challenger import compare
from edgeagent.brainbox.drift import psi
from edgeagent.features.state import build_features
from edgeagent.ingestion.quality_score import compute_dqs
from edgeagent.markets.devig import devig
from edgeagent.markets.edge import clv
from edgeagent.models.calibration import Calibrator, ece
from edgeagent.models.poisson import PoissonAD, probs_from_lams
from edgeagent.risk.kelly import cap_daily, kelly_stake
from edgeagent.utils import stage_for_hours


def test_devig_sums_to_one():
    o = np.array([[2.0, 3.5, 4.0], [1.5, 4.2, 7.0]])
    for m in ("proportional", "power"):
        assert np.allclose(devig(o, m).sum(1), 1)
    assert np.allclose(devig([2.0, 2.0, 2.0]), 1 / 3)

def test_kelly_cap_and_no_edge():
    assert kelly_stake(0.9, 3.0, 0.15, 0.02) == 0.02
    assert kelly_stake(0.3, 3.0) == 0.0
    assert sum(cap_daily([0.02] * 5, 0.06)) <= 0.0601

def test_clv_sign():
    assert clv(2.2, 0.5) > 0 and clv(1.8, 0.5) < 0

def test_stage_boundaries():
    assert [stage_for_hours(h) for h in (30, 20, 5, 0.9, 0.4)] == [0, 1, 2, 3, 4]

def test_features_are_leak_free(pipe):
    M = pipe.store.matches(finished=True).iloc[:300].copy()
    X1, _ = build_features(M)
    M2 = M.copy(); M2.loc[M2.index[-1], ["hg", "ag"]] = [9, 0]   # change the LAST result
    X2, _ = build_features(M2)
    assert X1.iloc[-1].equals(X2.iloc[-1])                        # its own features unchanged

def test_poisson_recovers_strength_order(pipe):
    M = pipe.store.matches(finished=True)
    m = PoissonAD().fit(M)
    sub = pd.DataFrame({"home": M.home.unique()[:2], "away": M.away.unique()[1:3]})
    P = m.predict_many(sub)
    assert np.allclose(P.sum(1), 1) and (P > 0).all()
    assert probs_from_lams([1.5], [1.0])[0, 0] > probs_from_lams([1.0], [1.5])[0, 0]

def test_calibrator_identity_until_enough_data_and_ece():
    rng = np.random.default_rng(0)
    P = rng.dirichlet([2, 2, 2], 400); y = np.array([rng.choice(3, p=p) for p in P])
    assert np.allclose(Calibrator(500).fit(P, y).transform(P), P)
    c = Calibrator(300).fit(P, y)
    assert np.allclose(c.transform(P).sum(1), 1) and ece(c.transform(P), y) < 0.1

def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 2000), rng.normal(0, 1, 500)) < 0.1
    assert psi(rng.normal(0, 1, 2000), rng.normal(1.5, 1, 500)) > 0.2

def test_challenger_needs_significance_and_volume():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 0.3, 300)
    assert not compare(a, a + rng.normal(0, 0.05, 300), 200, 0.05)["promote"]          # noise only
    assert compare(a, a - 0.1 + rng.normal(0, 0.05, 300), 200, 0.05)["promote"]         # real gain
    assert not compare(a[:50], a[:50] - 0.1, 200, 0.05)["promote"]                      # too few matches

def test_dqs_gate_on_stale_odds():
    ok = compute_dqs(30, True, 60, 900, 10, False, True, 0.01)
    bad = compute_dqs(30, True, 4000, 900, 10, False, True, 0.01)
    assert ok["score"] >= 0.8 > bad["score"]
