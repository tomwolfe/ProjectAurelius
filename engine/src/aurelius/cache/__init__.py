"""Abstract cache backend for PropertyOracle — decouples storage from core engine.

CacheBackend is the abstract interface. DictCache is the default in-memory
backend. DiskCacheBackend wraps diskcache for persistent local storage.
RedisCacheBackend is available for containerized deployments.

Usage:
    cache = DictCache()               # default, zero dependencies
    cache = DiskCacheBackend()         # requires ``diskcache``
    cache = RedisCacheBackend()        # requires ``redis``
"""

from __future__ import annotations

import abc
import os
from collections.abc import Iterator
from typing import Any


class CacheBackend(abc.ABC):
    """Abstract cache storage interface for PropertyOracle."""

    @abc.abstractmethod
    def get(self, key: str) -> Any | None:
        ...

    @abc.abstractmethod
    def __setitem__(self, key: str, value: Any) -> None:
        ...

    @abc.abstractmethod
    def __getitem__(self, key: str) -> Any:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[str]:
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        ...


class DictCache(CacheBackend):
    """In-memory dictionary cache — default backend, zero extra dependencies."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._data.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def clear(self) -> None:
        self._data.clear()


class DiskCacheBackend(CacheBackend):
    """diskcache-based persistent cache — requires the ``diskcache`` package."""

    def __init__(
        self,
        directory: str | None = None,
        size_limit: int = 1_000_000_000,
    ) -> None:
        import diskcache

        self._cache = diskcache.Cache(
            directory=directory
            or os.path.join(os.path.expanduser("~"), ".aurelius", "oracle_cache"),
            size_limit=size_limit,
        )

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._cache)

    def clear(self) -> None:
        self._cache.clear()


class RedisCacheBackend(CacheBackend):
    """Optional Redis-backed cache — requires the ``redis`` package.

    Only constructed when explicitly injected.  Never imported at module
    level, keeping Redis completely optional.
    """

    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "aurelius:") -> None:
        import redis as _redis_module

        self._client = _redis_module.from_url(url)
        self._prefix = prefix

    def _key(self, k: str) -> str:
        return self._prefix + k

    def get(self, key: str) -> Any | None:
        import json

        val = self._client.get(self._key(key))
        if val is None:
            return None
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val

    def __setitem__(self, key: str, value: Any) -> None:
        import json

        self._client.set(self._key(key), json.dumps(value, default=str))

    def __getitem__(self, key: str) -> Any:
        val = self._client.get(self._key(key))
        if val is None:
            raise KeyError(key)
        import json

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
                yield key.decode("utf-8", errors="replace")[len(self._prefix) :]
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
