"""Redis-backed cache backend for PropertyOracle.

Requires the ``redis`` package (optional dependency).

Usage:
    from aurelius.cache.redis_cache import RedisCache

    cache = RedisCache(url="redis://localhost:6379/0")
    oracle = PropertyOracle(l2_cache=cache)
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from aurelius.cache import CacheBackend


class RedisCache(CacheBackend):
    """Redis-backed cache for PropertyOracle evaluation results.

    Only constructed when explicitly injected.  Never imported at module
    level, keeping Redis completely optional.

    Keys are prefixed with ``aurelius:cache:`` to namespace within a shared
    Redis instance.

    Args:
        url: Redis connection URL.  Defaults to ``REDIS_URL`` env var, then
            ``redis://localhost:6379/0``.
        prefix: Key prefix for namespacing.
        ttl: Time-to-live in seconds for cache entries (default 86400 = 1 day).
    """

    def __init__(
        self,
        url: str | None = None,
        prefix: str = "aurelius:cache:",
        ttl: int = 86400,
    ) -> None:
        import redis as _redis_module

        redis_url = url or os.environ.get("REDIS_URL") or "redis://localhost:6379/0"
        self._client = _redis_module.from_url(redis_url, decode_responses=True)
        self._prefix = prefix
        self._ttl = ttl

    def _key(self, k: str) -> str:
        return self._prefix + k

    def get(self, key: str) -> Any | None:
        val = self._client.get(self._key(key))
        if val is None:
            return None
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val

    def __setitem__(self, key: str, value: Any) -> None:
        self._client.setex(self._key(key), self._ttl, json.dumps(value, default=str))

    def __getitem__(self, key: str) -> Any:
        val = self._client.get(self._key(key))
        if val is None:
            raise KeyError(key)
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val

    def __iter__(self) -> Iterator[str]:
        cursor = 0
        pattern = self._prefix + "*"
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=100)
            for key in keys:
                yield str(key)[len(self._prefix) :]
            if cursor == 0:
                break

    def clear(self) -> None:
        cursor = 0
        pattern = self._prefix + "*"
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break
