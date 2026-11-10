import asyncio
import json

import numpy as np

from edgeagent.backtest import walk_forward


def test_trained_models_beat_uniform_baseline(pipe):
    r = pipe.bundles["A"].report
    assert r["ensemble"]["logloss"] < r["uniform_logloss"]

def test_live_loop_predicts_locks_and_learns(pipe):
    async def go():
        for _ in range(8 * 14):                      # 14 simulated days, 3h steps
            await pipe.run_once(); pipe.source.advance(3); await pipe.settle()
    asyncio.run(go())
    s = pipe.store
    preds = s._df("SELECT * FROM predictions")
    assert len(preds) > 20 and preds.locked.any()
    assert s.scores().model.nunique() >= 6                       # members + ensemble + final + market + challenger
    # no fixture has predictions after its lock
    for fid, g in preds.groupby("fixture_id"):
        if g.locked.any():
            assert g[g.locked].stage.max() == g.stage.max() == 4
    # probabilities valid whenever a prediction is allowed
    ok = preds[preds.predictable]
    assert np.allclose(ok[["p_h", "p_d", "p_a"]].sum(1), 1, atol=1e-6)

def test_locked_prediction_is_immutable(pipe):
    row = pipe.store._df("SELECT fixture_id FROM predictions WHERE locked LIMIT 1").fixture_id[0]
    assert pipe.store.save_prediction({"fixture_id": row, "stage": 2, "predictable": True, "locked": False}) is False

def test_stale_odds_are_refused(pipe):
    p = pipe.store._df("SELECT predictable, reason, dqs FROM predictions WHERE NOT predictable")
    if len(p):
        assert p.reason.str.contains("data quality|guardrail").all()
        assert (p.dqs.isna() | (p.dqs < 0.8) | p.reason.str.contains("guardrail")).all()

def test_brainbox_status_shape(pipe):
    st = pipe.bb.status(pipe)
    for k in ["logloss", "weights", "market_weight", "calib", "drift", "cc", "guardrail", "errors", "train_report"]:
        assert k in st
    assert abs(sum(st["weights"].values()) - 1) < 1e-6
    json.dumps(st, default=str)

def test_walk_forward_has_no_free_lunch_on_efficient_synthetic_market(pipe):
    res = walk_forward.run(pipe.store.matches(finished=True), pipe.cfg.model_configs["A"], pipe.cfg.brainbox, pipe.cfg.risk, retrain_days=90)
    s = res["summary"]
    assert s["final"]["logloss"] < s["uniform_logloss"]
    assert s["final"]["logloss"] <= s["model"]["logloss"] + 0.01          # market blend should not hurt
    assert s["betting"]["mean_clv"] <= 0.02                                # no real edge exists in this simulation
