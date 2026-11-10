from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("edgeagent")

async def run_forever(pipe, interval_s: float = 600, sim_step_hours: float = 0.0):
    """Predict at T-24h/6h/1h/30m stages, lock at T-30m, settle finished matches, let the Brainbox learn."""
    await asyncio.to_thread(_ready, pipe)
    while True:
        try:
            made = await pipe.run_once()
            if getattr(pipe.source, "is_synthetic", False) and sim_step_hours:
                pipe.advance(sim_step_hours)
            res = await pipe.settle()
            log.info("predictions=%d settled=%d scored=%d actions=%s", len(made), res["new_results"], res["scored"], res["actions"])
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("pipeline tick failed")
        await asyncio.sleep(interval_s)

def _ready(pipe):
    asyncio.run(pipe.ensure_ready())
