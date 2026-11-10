from fastapi.testclient import TestClient

from edgeagent.api.main import create_app


def test_routes(pipe):
    c = TestClient(create_app(pipe))
    assert c.get("/health").json() == {"ok": True}
    assert c.get("/v1/meta").json()["synthetic"] is True
    fx = c.get("/v1/fixtures/upcoming"); assert fx.status_code == 200
    rows = fx.json()
    for r in rows[:3]:
        assert {"fixture_id", "league", "home", "away", "kickoff", "hours_to_ko", "stage", "dqs", "predictable", "p_home"} <= set(r)
    if rows:
        assert c.get(f"/v1/fixtures/{rows[0]['fixture_id']}").status_code == 200
    assert c.get("/v1/fixtures/nope").status_code == 404
    assert c.get("/v1/brainbox/status").status_code == 200
    assert isinstance(c.get("/v1/data/sources").json(), list)
    assert c.get("/v1/backtest/latest").status_code == 200

def test_admin_route_requires_key(pipe):
    c = TestClient(create_app(pipe))
    assert c.post("/v1/pipeline/run").status_code == 401
