"""LRU (Least Recently Used) cache implementation.

Replaces ad-hoc ``dict`` caching in ``MutationEngine`` and
``PropertyOracle`` with a bounded, thread-safe LRU cache.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class LRUCache(Generic[T]):
    """Bounded LRU cache with thread-safe operations.

    Evicts the least recently accessed item when the cache exceeds
    *maxsize*.  Useful for caching fingerprint computations and
    oracle evaluations where repeated calls on the same SMILES string
    should hit the cache, but memory must be bounded.

    Args:
        maxsize: Maximum number of entries before eviction (default 4096).
    """

    __slots__ = ("_cache", "_lock", "_maxsize")

    def __init__(self, maxsize: int = 4096) -> None:
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.RLock()
        self._maxsize = maxsize

    def get(self, key: str) -> T | None:
        """Return the value for *key*, or ``None`` if missing.

        Moves the accessed item to the end (most recently used).
        """
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: T) -> None:
        """Insert or update the value for *key*.

        If the cache is full, evicts the least recently used item.
        """
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def __getitem__(self, key: str) -> T:
        """Dict-compatible access: raises KeyError if missing."""
        with self._lock:
            if key not in self._cache:
                raise KeyError(key)
            self._cache.move_to_end(key)
            return self._cache[key]

    def __setitem__(self, key: str, value: T) -> None:
        """Dict-compatible assignment (same as put)."""
        self.put(key, value)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def values(self) -> list[T]:
        with self._lock:
            return list(self._cache.values())

    def items(self) -> list[tuple[str, T]]:
        with self._lock:
            return list(self._cache.items())
