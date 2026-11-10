from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..pipeline import Pipeline
from ..scheduler.prematch_jobs import run_forever

log = logging.getLogger("edgeagent")

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

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/v1/meta")
    def meta():
        return {"source": pipe.source.name, "synthetic": bool(getattr(pipe.source, "is_synthetic", False)),
                "active_config": pipe.active, "version": pipe.version, "now": str(pipe.source.now())}

    @app.get("/v1/fixtures/upcoming")
    def fixtures():
        return pipe.fixtures_view()

    @app.get("/v1/fixtures/{fixture_id}")
    def fixture(fixture_id: str):
        rows = [r for r in pipe.fixtures_view() if r["fixture_id"] == fixture_id]
        if not rows:
            raise HTTPException(404, "no prediction for this fixture")
        return rows[0]

    @app.get("/v1/brainbox/status")
    def brainbox():
        return pipe.bb.status(pipe)

    @app.get("/v1/data/sources")
    def sources():
        return pipe.source.status()

    @app.get("/v1/backtest/latest")
    def backtest():
        return (pipe.store.kv_get("backtest_latest") or {}).get("curve", [])

    @app.get("/v1/backtest/summary")
    def backtest_summary():
        return (pipe.store.kv_get("backtest_latest") or {}).get("summary", {})

    @app.post("/v1/pipeline/run", dependencies=[Depends(admin)])
    async def run_now():
        made = await pipe.run_once()
        return {"predictions": len(made), **await pipe.settle()}

    return app

def app_factory() -> FastAPI:
    return create_app()
