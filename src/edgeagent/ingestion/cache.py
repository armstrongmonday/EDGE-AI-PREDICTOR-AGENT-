from __future__ import annotations

import json
import logging
import time

log = logging.getLogger("edgeagent")

class Cache:
    """TTL cache: Redis when REDIS_URL is set and reachable, otherwise in-process memory."""
    def __init__(self, redis_url: str = ""):
        self._mem: dict[str, tuple[float, str]] = {}
        self._r = None
        if redis_url:
            try:
                import redis.asyncio as aioredis
                self._r = aioredis.from_url(redis_url, decode_responses=True)
            except Exception:
                log.warning("redis unavailable, using memory cache")

    async def get(self, key: str):
        if self._r is not None:
            try:
                v = await self._r.get(key)
                return json.loads(v) if v else None
            except Exception:
                self._r = None
        hit = self._mem.get(key)
        return json.loads(hit[1]) if hit and hit[0] > time.time() else None

    async def set(self, key: str, value, ttl: float):
        s = json.dumps(value)
        if self._r is not None:
            try:
                await self._r.set(key, s, ex=int(ttl)); return
            except Exception:
                self._r = None
        self._mem[key] = (time.time() + ttl, s)
