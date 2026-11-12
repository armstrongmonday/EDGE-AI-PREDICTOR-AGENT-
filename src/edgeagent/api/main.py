from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..pipeline import Pipeline
from ..scheduler.prematch_jobs import run_forever

log = logging.getLogger("edgeagent")
WEB = Path(__file__).resolve().parents[1] / "web"

def create_app(pipe: Pipeline | None = None) -> FastAPI:
    pipe = pipe or Pipeline()

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(run_forever(pipe, pipe.cfg.sources["interval_s"] if not pipe.cfg.sim_step_hours else 10,
                                               pipe.cfg.sim_step_hours)) if pipe.cfg.autorun else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="EdgeAgent API", version="0.2.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    def admin(x_admin_key: str = Header(default="")):
        if not pipe.cfg.admin_key or x_admin_key != pipe.cfg.admin_key:
            raise HTTPException(401, "admin key required")

    def guard(authorization: str = Header(default="")):
        """Optional bearer token (EDGEAGENT_API_TOKEN). Strongly recommended if the API is reachable from the internet."""
        tok = pipe.cfg.api_token
        if tok and not hmac.compare_digest(authorization.encode(), f"Bearer {tok}".encode()):
            raise HTTPException(401, "invalid or missing API token")

    G = [Depends(guard)]

    @app.get("/")
    def root():
        return RedirectResponse("/app/")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/v1/meta", dependencies=G)
    def meta():
        return {"source": pipe.source.name, "synthetic": bool(getattr(pipe.source, "is_synthetic", False)),
                "active_config": pipe.active, "version": pipe.version, "now": str(pipe.source.now())}

    @app.get("/v1/fixtures/upcoming", dependencies=G)
    def fixtures():
        return pipe.fixtures_view()

    @app.get("/v1/fixtures/{fixture_id}", dependencies=G)
    def fixture(fixture_id: str):
        rows = [r for r in pipe.fixtures_view() if r["fixture_id"] == fixture_id]
        if not rows:
            raise HTTPException(404, "no prediction for this fixture")
        return rows[0]

    @app.get("/v1/brainbox/status", dependencies=G)
    def brainbox():
        return pipe.bb.status(pipe)

    @app.get("/v1/data/sources", dependencies=G)
    def sources():
        return pipe.source.status()

    @app.get("/v1/backtest/latest", dependencies=G)
    def backtest():
        return (pipe.store.kv_get("backtest_latest") or {}).get("curve", [])

    @app.get("/v1/backtest/summary", dependencies=G)
    def backtest_summary():
        return (pipe.store.kv_get("backtest_latest") or {}).get("summary", {})

    @app.post("/v1/pipeline/run", dependencies=[Depends(admin)])
    async def run_now():
        made = await pipe.run_once()
        return {"predictions": len(made), **await pipe.settle()}

    if WEB.exists():
        app.mount("/app", StaticFiles(directory=WEB, html=True), name="web")
    return app

def app_factory() -> FastAPI:
    return create_app()
