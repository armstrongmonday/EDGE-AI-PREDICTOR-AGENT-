# 🧠 EdgeAgent v2: Live-Data Agentic Sports Prediction with a Self-Improving "Brainbox"

> Fetches live fixtures, lineups and odds, builds pre-match probability forecasts from verified statistics, and uses a **Brainbox** learning layer that audits every prediction after the match and continuously improves the models.

![status](https://img.shields.io/badge/status-alpha-orange) ![python](https://img.shields.io/badge/python-3.11+-blue) ![license](https://img.shields.io/badge/license-MIT-green)

---

## ⚠️ Honest Disclaimer

- **No accuracy of 90-95% is claimed or achievable.** Betting markets are efficient. A strong model aims for **well-calibrated probabilities** and a small, measurable edge (e.g. positive closing-line value), not certainty.
- Football 1X2 outcomes are inherently noisy; realistic pick accuracy is roughly 50-58% on 3-way markets, depending on league and odds band.
- Predictions are probabilities, not guarantees. Backtests do not guarantee future results.
- **Aviator / crash games cannot be predicted** (provably-fair RNG). Only an educational simulator is included.
- Research and education only. Check local gambling laws. Gamble responsibly.

---

## 📌 Contents

1. [What's New in v2](#whats-new-in-v2)
2. [System Architecture](#system-architecture)
3. [Live Data Pipeline](#live-data-pipeline)
4. [Pre-Match Prediction Workflow](#pre-match-prediction-workflow)
5. [The Brainbox](#the-brainbox-self-improvement-layer)
6. [Models and Features](#models-and-features)
7. [Efficiency Design](#efficiency-design)
8. [Repository Structure](#repository-structure)
9. [Installation](#installation)
10. [Quick Start](#quick-start)
11. [Configuration](#configuration)
12. [Evaluation and Backtesting](#evaluation-and-backtesting)
13. [Aviator Lab](#aviator-lab-educational)
14. [Roadmap](#roadmap)
15. [Contributing and License](#contributing--license)

---

## What's New in v2

| Area | v1 | v2 |
|---|---|---|
| Data | Batch/historical | **Live fixtures, lineups, injuries, odds (async, cached)** |
| Timing | Ad hoc | **Scheduled pre-match snapshots (T-24h, T-6h, T-1h, T-30m)** |
| Learning | Static retrain | **Brainbox: online learning, drift detection, error analysis** |
| Speed | Pandas loops | **DuckDB + Parquet, async I/O, Redis cache, ONNX inference** |
| Output | Probabilities | **Probabilities + confidence + data-quality score + explanation** |

---

## System Architecture

```
                    ┌───────────────────────────┐
                    │     Orchestrator Agent    │
                    │ (scheduler + task graph)  │
                    └─────────────┬─────────────┘
   ┌───────────┬─────────────┬────┴───────┬──────────────┬─────────────┐
   ▼           ▼             ▼            ▼              ▼             ▼
 Live Data   Feature      Model        Market         Risk         Report
 Agent       Agent        Agent        Agent          Agent        Agent
 (APIs,      (incremental (ensemble +  (odds, de-vig, (stake       (LLM
 webhooks)   features)    calibration) edge, CLV)     simulation)  explanations)
   │            │             │            │
   └────────────┴──────┬──────┴────────────┘
                       ▼
              ┌─────────────────┐   feedback loop
              │    BRAINBOX     │◄───────────────┐
              │ (meta-learning) │                │
              └────────┬────────┘        Post-match results
                       ▼
        model weights · feature set · calibration · alerts
```

---

## Live Data Pipeline

**Goal:** always predict from the freshest verified data, and refuse to predict when data is stale or incomplete.

### Sources (check each provider's ToS and licensing)

| Data | Providers |
|---|---|
| Fixtures, results, standings | API-Football, football-data.org, SportMonks |
| Lineups, injuries, suspensions | API-Football, official club/league feeds |
| Advanced stats (xG, shots, pressing) | Understat, FBref, StatsBomb Open Data |
| Odds (multi-bookmaker) | The Odds API, Betfair Exchange, Pinnacle (sharp benchmark) |
| Context | Weather APIs, travel distance, fixture congestion |

### Ingestion rules

- **Async fetchers** (`httpx` + `asyncio`) with rate-limit aware retries and exponential backoff
- **ETag / If-Modified-Since** to skip unchanged payloads
- **Redis cache** with per-endpoint TTLs (odds: 30-60 s, lineups: 5 min, history: 24 h)
- **Schema validation** (Pydantic) and **cross-source reconciliation** (e.g. kickoff time, team IDs)
- **Data Quality Score (DQS 0-1)** per fixture: freshness, completeness, source agreement
- **Hard gate:** fixtures with DQS below the threshold are flagged "No prediction"

---

## Pre-Match Prediction Workflow

Predictions are produced in **staged snapshots** and **locked before kickoff** so evaluation is honest.

| Stage | Time to KO | Action |
|---|---|---|
| 1 | T-24h | Baseline forecast from ratings, form, xG, rest, venue |
| 2 | T-6h | Update with injuries, weather, market movement |
| 3 | T-1h | Confirmed lineups and key-player impact adjustment |
| 4 | T-30m | **Final locked forecast** + edge vs closing-line proxy |

Each prediction record stores: model version, feature snapshot hash, probabilities (H/D/A, O/U, BTTS), confidence interval, DQS, and market odds at the time of prediction. This makes every forecast **auditable and reproducible**.

---

## The Brainbox (Self-Improvement Layer)

The Brainbox is a meta-learning layer that sits above the models and answers: *which models, features and settings should we trust right now?*

### Components

1. **Performance Tracker**
   Records log loss, Brier score, calibration error and CLV per model, league, and odds band over rolling windows.

2. **Dynamic Ensemble Weighting**
   Model weights are updated from recent out-of-sample log loss (e.g. exponentially weighted / Hedge algorithm), so better-performing models gain influence automatically.

3. **Drift Detector**
   Monitors feature and error distributions (PSI, KS test, ADWIN). On drift (new season, managerial change, rule change), it triggers recalibration or retraining.

4. **Post-Match Error Analyst (LLM + stats)**
   For large misses, it compares expected vs actual (xG, red cards, late lineup changes, injuries) and classifies the cause as *variance*, *missing data*, or *model weakness*. Only systematic patterns generate action items.

5. **Auto-Calibration**
   Re-fits isotonic / Platt scaling on a rolling window so "65%" predictions really win about 65%.

6. **Feature Selector and Hyperparameter Tuner**
   Scheduled Optuna runs with strict walk-forward validation; promotes new features only if they improve out-of-sample metrics.

7. **Champion / Challenger Gate**
   New models run in shadow mode. A challenger replaces the champion only after beating it over N matches on log loss **and** calibration, with statistical significance.

8. **Safety Guardrails**
   Auto-suppresses predictions in leagues or markets where recent calibration or CLV degrades; raises alerts instead of silently continuing.

### Learning loop

```
Predict (locked) → Match ends → Ingest result → Score prediction
      → Brainbox analysis → Update weights / calibration / features
      → Shadow-test challengers → Promote if significantly better
```

> The Brainbox improves **reliability and calibration**. It can't create information the data doesn't contain, and it guards against overfitting by promoting changes only on out-of-sample evidence.

---

## Models and Features

**Models:** Elo/Glicko baseline · Dixon-Coles / bivariate Poisson · LightGBM/XGBoost · regularised logistic regression · small neural net · stacked ensemble

**Key features**
- Team strength: Elo, attack/defence ratings, xG for/against (home/away split)
- Form: exponentially weighted recent results and xG difference
- Schedule: rest days, congestion, travel distance
- Squad: confirmed lineup strength, missing-player impact (minutes-weighted)
- Market: de-vigged consensus probability, line movement, sharp-vs-soft gap
- Context: weather, stakes (relegation/title), derby flag

**Outputs:** 1X2, over/under, BTTS, expected goals, scoreline distribution

---

## Efficiency Design

- **DuckDB + Parquet** columnar storage for fast analytical queries
- **Incremental feature computation** (only new matches recomputed)
- **Async ingestion** and parallel agent execution
- **Redis caching** with TTLs to cut API calls and cost
- **ONNX / LightGBM native inference** for millisecond predictions
- **Batch scoring** per matchday; **event-driven re-scoring** only when inputs change materially
- **Model registry (MLflow)** with versioned artifacts for instant rollback
- **Docker + CI** for reproducible deploys

---

## Repository Structure

```
edgeagent/
├── README.md
├── LICENSE
├── pyproject.toml
├── Makefile
├── docker-compose.yml
├── .env.example
├── configs/
│   ├── agents.yaml
│   ├── sources.yaml          # API endpoints, TTLs, rate limits
│   ├── models.yaml
│   ├── brainbox.yaml         # windows, thresholds, promotion rules
│   └── risk.yaml
├── src/edgeagent/
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── live_data_agent.py
│   │   ├── feature_agent.py
│   │   ├── model_agent.py
│   │   ├── market_agent.py
│   │   ├── risk_agent.py
│   │   └── report_agent.py
│   ├── ingestion/
│   │   ├── connectors/       # api_football.py, odds_api.py, understat.py
│   │   ├── cache.py
│   │   ├── validators.py
│   │   └── quality_score.py
│   ├── features/
│   │   ├── ratings.py
│   │   ├── form.py
│   │   ├── xg.py
│   │   └── lineup_impact.py
│   ├── models/
│   │   ├── dixon_coles.py
│   │   ├── gbm.py
│   │   ├── ensemble.py
│   │   └── calibration.py
│   ├── brainbox/
│   │   ├── tracker.py
│   │   ├── weighting.py
│   │   ├── drift.py
│   │   ├── error_analyst.py
│   │   ├── auto_calibrate.py
│   │   ├── tuner.py
│   │   └── champion_challenger.py
│   ├── markets/
│   │   ├── devig.py
│   │   ├── edge.py
│   │   └── clv.py
│   ├── scheduler/
│   │   └── prematch_jobs.py
│   ├── backtest/
│   │   ├── walk_forward.py
│   │   └── metrics.py
│   └── api/main.py
├── aviator_lab/
├── dashboard/app.py
├── notebooks/
├── tests/
└── docs/
    ├── methodology.md
    ├── data_sources.md
    ├── brainbox.md
    └── limitations.md
```

---

## Installation

```bash
git clone https://github.com/<your-org>/edgeagent.git
cd edgeagent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d redis      # cache
```

## Implementation Status (read before trusting anything)

| Component | Status |
|---|---|
| Features, models, ensemble, calibration, Brainbox, pipeline, DuckDB store, FastAPI, CLI, dashboard | Built; **22 automated tests pass** (`make test`) |
| End-to-end loop (stage predictions, lock, settle, learn, promote) | Verified on a **synthetic league simulator** (`EDGEAGENT_SOURCE=synthetic`) |
| Live connectors (football-data.co.uk, API-Football, The Odds API) | Written from provider docs and unit-tested against **mocked** responses only. **Not yet run against real keys**; verify field names and team-name matching on first use |
| Real-world accuracy | **Unmeasured.** Run `backfill` + `backtest` on real history to find out |

What the synthetic runs show (not a claim about real sport): the ensemble beats the uniform baseline, the learned market blend lands near the market's log loss, and simulated betting shows **no edge**, because the simulated market is efficient by construction. Expect the same humility on real data.

**Deviations from the sections above (deliberate simplifications)**
- Goal model is independent Poisson with time decay (no Dixon-Coles low-score correction yet); gradient boosting uses scikit-learn's `HistGradientBoostingClassifier`, not LightGBM.
- `ci_width` is a **model-disagreement band** (1.96 x std across members), not a statistical confidence interval.
- Drift uses PSI + KS tests and a Page-Hinkley alarm on log loss (not ADWIN). The error analyst is rule-based; there is no LLM call yet.
- The final probability blends the calibrated model ensemble with the de-vigged market; the Brainbox learns the blend weight. Edge = final minus market, so it shrinks toward zero when the models add no information.
- Isotonic calibration stays an identity map until 300 out-of-sample matches exist. Team-name aliases cover the big leagues; extend `ingestion/matching.py` for others and watch `LiveSource.unmatched`.

## Quick Start

```bash
pip install -e ".[dev]" -r dashboard/requirements.txt
cp .env.example .env

# 1. Offline demo (synthetic world): train, predict by stage, lock, settle, learn
python -m edgeagent.cli simulate --hours 600 --step 3
python -m edgeagent.cli backtest          # walk-forward report, stored for the dashboard

# 2. Serve API + dashboard
python -m edgeagent.cli serve             # http://localhost:8000/docs
cd dashboard && EDGEAGENT_API_URL=http://localhost:8000 streamlit run app.py

# Or everything in containers (synthetic by default)
docker compose up --build                 # API :8000, dashboard :8501
```

**Live data**
```bash
export EDGEAGENT_SOURCE=live API_FOOTBALL_KEY=... ODDS_API_KEY=...
python -m edgeagent.cli backfill          # 3 seasons of results + odds (football-data.co.uk)
python -m edgeagent.cli train             # prints the honest hold-out report
python -m edgeagent.cli run               # loop: fetch, predict at T-24h/6h/1h/30m, lock, settle, learn
```
Mind the quotas: the free Odds API tier is roughly 500 credits/month and API-Football's free tier roughly 100 requests/day. TTLs and the loop interval live in `configs/sources.yaml` (defaults: 15-minute odds cache, 10-minute loop).

DuckDB allows one writing process per file: either run the pipeline inside the API (`EDGEAGENT_AUTORUN=1`) or run CLI commands while the API is stopped.

## API

| Route | Purpose |
|---|---|
| `GET /v1/fixtures/upcoming`, `/v1/fixtures/{id}` | Latest staged prediction, market, edge, DQS, refusal reason |
| `GET /v1/brainbox/status` | Log loss by model, weights, market trust, calibration, drift, challenger test, guardrails |
| `GET /v1/data/sources` | Source freshness and completeness |
| `GET /v1/backtest/latest`, `/v1/backtest/summary` | Curves and metrics from the last backtest |
| `GET /v1/meta` | Source type (synthetic/live), active config, version |
| `POST /v1/pipeline/run` | Manual tick; requires `X-Admin-Key` = `EDGEAGENT_ADMIN_KEY` |

## Configuration

`configs/brainbox.yaml`, `configs/risk.yaml` and `configs/sources.yaml` hold thresholds (promotion p-value, PSI limit, ECE guardrail, Kelly fraction, TTLs). Keys and switches go in `.env` (see `.env.example`).

## Evaluation and Backtesting

Accuracy alone is misleading. We track:

| Metric | Purpose |
|---|---|
| Log loss / Brier | Probability quality |
| Calibration (ECE, reliability curve) | Are 60% picks winning ~60%? |
| **Closing Line Value (CLV)** | Best indicator of a genuine edge |
| Simulated ROI with CI | Profitability with uncertainty |
| Max drawdown | Risk assessment |
| Hit rate by odds band | Avoids favourite-bias illusions |

```bash
python -m edgeagent.cli backtest --league epl \
  --start 2021-08-01 --end 2025-05-31 --walk-forward --retrain-every 30d
```

Rules: strict time-ordered splits, no future leakage, odds taken at realistic timestamps, margin included, bootstrap confidence intervals.

## Aviator Lab (Educational)

`/aviator_lab` is a **simulator that demonstrates why Aviator is unpredictable**: each round is independent, and every cash-out strategy converges to the house edge. Any "Aviator predictor" is a scam.

## Roadmap

- [ ] In-play live probability model
- [ ] Player-level injury impact model
- [ ] Bayesian uncertainty-aware edge filtering
- [ ] Multi-sport expansion (basketball, tennis)
- [ ] Public calibration/CLV leaderboard
- [ ] Reinforcement-learning meta-controller for Brainbox

## Contributing / License

Fork, branch, add tests, run `make lint test`, open a PR with backtest impact noted. Licensed under MIT.

---

*For research into probabilistic forecasting. No guarantees of profit.*
