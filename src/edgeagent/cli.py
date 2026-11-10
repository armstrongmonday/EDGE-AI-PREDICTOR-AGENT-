from __future__ import annotations

import argparse
import asyncio
import json
import logging

from .pipeline import Pipeline


def main(argv=None):
    ap = argparse.ArgumentParser("edgeagent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backfill"); sub.add_parser("train"); sub.add_parser("status")
    r = sub.add_parser("run"); r.add_argument("--once", action="store_true"); r.add_argument("--interval", type=float, default=0)
    s = sub.add_parser("simulate"); s.add_argument("--hours", type=float, default=240); s.add_argument("--step", type=float, default=3)
    b = sub.add_parser("backtest"); b.add_argument("--retrain-days", type=int, default=60)
    v = sub.add_parser("serve"); v.add_argument("--port", type=int, default=8000); v.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("edgeagent.api.main:app_factory", factory=True, host=a.host, port=a.port)
        return
    pipe = Pipeline()

    async def go():
        if a.cmd == "backfill":
            print("matches stored:", await pipe.backfill())
        elif a.cmd == "train":
            await pipe.ensure_ready(); pipe.train(); print(json.dumps(pipe.bundles[pipe.active].report, indent=2, default=float))
        elif a.cmd == "status":
            print(json.dumps(pipe.bb.status(pipe), indent=2, default=str))
        elif a.cmd == "run":
            await pipe.ensure_ready()
            from .scheduler.prematch_jobs import run_forever
            if a.once:
                print(len(await pipe.run_once()), "predictions;", await pipe.settle())
            else:
                await run_forever(pipe, a.interval or pipe.cfg.sources["interval_s"], pipe.cfg.sim_step_hours)
        elif a.cmd == "simulate":
            if not getattr(pipe.source, "is_synthetic", False):
                raise SystemExit("simulate only works with the synthetic source (EDGEAGENT_SOURCE=synthetic)")
            await pipe.ensure_ready()
            for _ in range(int(a.hours / a.step)):
                await pipe.run_once(); pipe.advance(a.step); await pipe.settle()
            extra = 0                # keep going until something is in the T-24h window so the dashboard has content
            while not pipe.fixtures_view() and extra < 168:
                await pipe.run_once(); pipe.advance(a.step); await pipe.settle(); extra += a.step
            await pipe.run_once()
            print(json.dumps(pipe.bb.status(pipe)["actions"], indent=2))
        elif a.cmd == "backtest":
            from .backtest import walk_forward
            await pipe.ensure_ready()
            res = walk_forward.run(pipe.store.matches(finished=True), pipe.cfg.model_configs[pipe.active], pipe.cfg.brainbox,
                                   pipe.cfg.risk, retrain_days=a.retrain_days)
            pipe.store.kv_set("backtest_latest", res)
            print(json.dumps(res["summary"], indent=2, default=float))
    asyncio.run(go())

if __name__ == "__main__":
    main()
