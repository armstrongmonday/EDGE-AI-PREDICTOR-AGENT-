from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

def _yaml(name: str) -> dict:
    p = ROOT / "configs" / name
    return yaml.safe_load(p.read_text()) or {} if p.exists() else {}

LEAGUES = {
    "EPL": {"name": "Premier League", "fd": "E0", "af": 39, "odds": "soccer_epl"},
    "LALIGA": {"name": "La Liga", "fd": "SP1", "af": 140, "odds": "soccer_spain_la_liga"},
    "SERIEA": {"name": "Serie A", "fd": "I1", "af": 135, "odds": "soccer_italy_serie_a"},
    "BUNDES": {"name": "Bundesliga", "fd": "D1", "af": 78, "odds": "soccer_germany_bundesliga"},
    "LIGUE1": {"name": "Ligue 1", "fd": "F1", "af": 61, "odds": "soccer_france_ligue_one"},
}

RISK_DEFAULTS = {"kelly_fraction": 0.15, "min_edge": 0.03, "max_stake_pct": 0.02, "max_daily_exposure_pct": 0.06}
BB_DEFAULTS = {"rolling_window_matches": 300, "min_matches_to_promote": 200, "promotion_p_value": 0.05,
               "psi_threshold": 0.2, "ks_alpha": 0.01, "weight_temperature": 12.0, "calibration_min_samples": 300,
               "recalibrate_every_matches": 50, "max_ece": 0.05, "min_ece_samples": 150, "min_dqs": 0.8}
SRC_DEFAULTS = {"ttl": {"fixtures": 900, "odds": 900, "lineups": 300, "history": 86400}, "interval_s": 600}
CONFIGS = {"A": {"xi": 0.0018, "gbm_depth": 3, "lr_C": 1.0}, "B": {"xi": 0.0035, "gbm_depth": 4, "lr_C": 0.5}}

@dataclass
class Settings:
    api_football_key: str = os.getenv("API_FOOTBALL_KEY", "")
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    db_path: str = os.getenv("EDGEAGENT_DB", str(ROOT / "data" / "edgeagent.duckdb"))
    source: str = os.getenv("EDGEAGENT_SOURCE", "synthetic")  # synthetic | live
    admin_key: str = os.getenv("EDGEAGENT_ADMIN_KEY", "")
    autorun: bool = os.getenv("EDGEAGENT_AUTORUN", "0") == "1"
    sim_step_hours: float = float(os.getenv("EDGEAGENT_SIM_STEP_HOURS", "0"))
    risk: dict = field(default_factory=lambda: {**RISK_DEFAULTS, **_yaml("risk.yaml")})
    brainbox: dict = field(default_factory=lambda: {**BB_DEFAULTS, **_yaml("brainbox.yaml")})
    sources: dict = field(default_factory=lambda: {**SRC_DEFAULTS, **_yaml("sources.yaml")})
    leagues: dict = field(default_factory=lambda: LEAGUES)
    model_configs: dict = field(default_factory=lambda: CONFIGS)
