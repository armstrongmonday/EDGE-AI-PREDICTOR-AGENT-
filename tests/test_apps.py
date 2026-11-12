import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from edgeagent.api.main import WEB, create_app
from edgeagent.config import Settings
from edgeagent.desktop import load_env
from edgeagent.ingestion import doctor
from edgeagent.ingestion.connectors import football_data
from edgeagent.ingestion.connectors.live import LiveSource


def test_api_token_guard(pipe):
    pipe.cfg.api_token = "s3cret"
    try:
        c = TestClient(create_app(pipe))
        assert c.get("/v1/meta").status_code == 401
        assert c.get("/v1/meta", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/v1/meta", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200            # liveness stays open
        assert c.get("/app/").status_code == 200              # static shell holds no data
    finally:
        pipe.cfg.api_token = ""


def test_pwa_assets_are_installable():
    m = json.loads((WEB / "manifest.webmanifest").read_text())
    assert m["display"] == "standalone" and {i["sizes"] for i in m["icons"]} >= {"192x192", "512x512"}
    for f in ["index.html", "app.js", "sw.js", "style.css", "icons/icon-192.png", "icons/icon-512.png"]:
        assert (WEB / f).exists()
    html = (WEB / "index.html").read_text()
    assert 'rel="manifest"' in html and "viewport-fit=cover" in html
    assert "innerHTML" not in (WEB / "app.js").read_text()     # API text must never be injected as markup


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_web_js_parses():
    for f in ("app.js", "sw.js"):
        assert subprocess.run(["node", "--check", str(WEB / f)], capture_output=True).returncode == 0


def test_desktop_env_loader(tmp_path, monkeypatch):
    cfg = tmp_path / "config.env"
    cfg.write_text("# c\nEDGEAGENT_SOURCE=live\nAPI_FOOTBALL_KEY=abc\nODDS_API_KEY=\n")
    monkeypatch.delenv("EDGEAGENT_SOURCE", raising=False)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setenv("API_FOOTBALL_KEY", "from-real-env")
    import os
    load_env(cfg)
    assert os.environ["EDGEAGENT_SOURCE"] == "live"
    assert os.environ["API_FOOTBALL_KEY"] == "from-real-env"   # real env wins
    assert "ODDS_API_KEY" not in os.environ                    # blank values ignored
    monkeypatch.delenv("EDGEAGENT_SOURCE", raising=False)


def _mock(fail_key=False):
    ko = (pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    def handler(req):
        p = req.url.path
        if fail_key:
            return httpx.Response(403, json={"message": "bad key"})
        if p == "/status":
            return httpx.Response(200, json={"response": {"subscription": {"plan": "Free"}, "requests": {"current": 3, "limit_day": 100}}})
        if p == "/leagues":
            return httpx.Response(200, json={"response": [{"league": {"name": "Premier League"}}]})
        if p == "/fixtures":
            return httpx.Response(200, json={"response": [{"fixture": {"id": 7, "date": ko, "status": {"short": "NS"}},
                                                         "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Chelsea"}}, "goals": {}}]})
        if p == "/v4/sports":
            return httpx.Response(200, json=[{"key": "soccer_epl", "active": True}])
        if p.endswith("/odds"):
            ev = {"home_team": "Arsenal", "away_team": "Chelsea", "commence_time": ko.replace("+00:00", "Z"),
                  "bookmakers": [{"key": "b", "markets": [{"key": "h2h", "outcomes": [{"name": "Arsenal", "price": 2.0},
                                  {"name": "Draw", "price": 3.5}, {"name": "Chelsea", "price": 3.8}]}]}]}
            return httpx.Response(200, json=[ev])
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_doctor_all_green_with_mocked_providers(monkeypatch):
    async def fake_fetch(div, code, season, client=None):
        return pd.DataFrame({"fixture_id": [str(i) for i in range(150)], "odds_h": [2.0] * 150})
    monkeypatch.setattr(football_data, "fetch", fake_fetch)
    cfg = Settings(api_football_key="k", odds_api_key="k", enabled_leagues="EPL")
    res = asyncio.run(doctor.run(LiveSource(cfg, transport=_mock())))
    assert all(ok for _, ok, _ in res), res
    assert any("1/1 fixtures matched" in d for _, _, d in res)


def test_doctor_reports_bad_keys_instead_of_crashing(monkeypatch):
    async def fake_fetch(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(football_data, "fetch", fake_fetch)
    cfg = Settings(api_football_key="bad", odds_api_key="bad", enabled_leagues="EPL")
    res = asyncio.run(doctor.run(LiveSource(cfg, transport=_mock(fail_key=True))))
    assert not all(ok for _, ok, _ in res) and any(n == "API-Football key" and not ok for n, ok, _ in res)


def test_live_source_requires_both_keys():
    with pytest.raises(RuntimeError):
        LiveSource(Settings(api_football_key="", odds_api_key="x"))


def test_packaging_files_exist():
    root = Path(__file__).resolve().parents[1]
    for f in ["packaging/build_desktop.py", ".github/workflows/desktop-windows.yml", ".github/workflows/android-apk.yml", "mobile/capacitor.config.json"]:
        assert (root / f).exists()
