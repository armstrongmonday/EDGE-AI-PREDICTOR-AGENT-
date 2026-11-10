from __future__ import annotations

import asyncio
import logging

import httpx

from ..cache import Cache

log = logging.getLogger("edgeagent")

class ConnectorError(RuntimeError):
    pass

class HttpConnector:
    """Async GET with TTL caching, retry/backoff and Retry-After handling. API keys never enter cache keys or logs."""
    def __init__(self, base: str, headers: dict | None = None, cache: Cache | None = None, timeout: float = 15,
                 retries: int = 4, transport: httpx.AsyncBaseTransport | None = None):
        self.base, self.headers, self.cache, self.retries = base, headers or {}, cache or Cache(), retries
        self.client = httpx.AsyncClient(timeout=timeout, transport=transport)
        self.last_ok: float | None = None
        self.calls = 0

    async def get_json(self, path: str, params: dict | None = None, ttl: float = 60, secret_params: dict | None = None):
        key = f"{self.base}{path}?{sorted((params or {}).items())}"
        hit = await self.cache.get(key)
        if hit is not None:
            return hit
        q = {**(params or {}), **(secret_params or {})}
        err = None
        for attempt in range(self.retries):
            try:
                self.calls += 1
                r = await self.client.get(self.base + path, params=q, headers=self.headers)
                if r.status_code == 429 or r.status_code >= 500:
                    wait = float(r.headers.get("retry-after", 2 ** attempt))
                    err = f"HTTP {r.status_code}"
                    await asyncio.sleep(min(wait, 30)); continue
                r.raise_for_status()
                data = r.json()
                await self.cache.set(key, data, ttl)
                self.last_ok = asyncio.get_event_loop().time()
                return data
            except httpx.TransportError as e:
                err = type(e).__name__
                await asyncio.sleep(2 ** attempt)
        raise ConnectorError(f"{self.base}{path} failed after {self.retries} tries ({err})")
