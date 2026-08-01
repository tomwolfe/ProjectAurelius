"""FastAPI server for Project Aurelius — decoupled from the core engine.

Provides REST endpoints for the Aurelius screening pipeline:
    POST /screen   — Screen a single molecule by SMILES
    POST /batch    — Screen multiple molecules by SMILES
    GET  /health   — Health check / status

All operational infrastructure (FastAPI, Uvicorn, optional Redis caching)
lives here, outside the ``aurelius`` package tree. The core engine knows
nothing about HTTP, web frameworks, or deployment.

Usage:
    uvicorn server.api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from aurelius.cache import CacheBackend, DictCache, DiskCacheBackend, RedisCacheBackend
from aurelius.pipeline import AureliusPipeline

logger = logging.getLogger(__name__)

_pipeline: AureliusPipeline | None = None

# ---------------------------------------------------------------------------
# API Key authentication
# ---------------------------------------------------------------------------

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_AURELIUS_API_KEY: str | None = os.environ.get("AURELIUS_API_KEY")


async def verify_api_key(api_key: str | None = Depends(API_KEY_HEADER)) -> None:
    if _AURELIUS_API_KEY is None:
        return
    if api_key != _AURELIUS_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# In-memory sliding-window rate limiter
# ---------------------------------------------------------------------------

_rate_store: dict[str, list[float]] = defaultdict(list)


class RateLimiter:
    def __init__(self, limit: int, window: float = 60.0) -> None:
        self.limit = limit
        self.window = window

    async def __call__(self, request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - self.window
        timestamps = _rate_store[ip]
        while timestamps and timestamps[0] < window_start:
            timestamps.pop(0)
        if len(timestamps) >= self.limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded ({self.limit} req / "
                    f"{self.window:.0f}s). Try again later."
                ),
            )
        timestamps.append(now)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ScreenRequest(BaseModel):
    smiles: str = Field(..., description="Canonical SMILES string of the molecule")


class BatchRequest(BaseModel):
    smiles: list[str] = Field(..., description="List of canonical SMILES strings", min_length=1)


class HealthResponse(BaseModel):
    status: str
    version: str = "11.0.0"
    pipeline_initialized: bool


# ---------------------------------------------------------------------------
# Cache backend factory — Redis if configured, otherwise DictCache
# ---------------------------------------------------------------------------


def _make_cache_backend() -> CacheBackend:
    """Create a cache backend based on the environment.

    If ``REDIS_URL`` is set, returns a ``RedisCacheBackend``.
    Otherwise returns a ``DictCache`` (in-memory). For persistent
    disk-based caching across restarts, set ``AURELIUS_CACHE_DIR``
    to a writable path to use ``DiskCacheBackend`` instead.
    """
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        logger.info("Using Redis cache backend: %s", redis_url.split("@")[-1] if "@" in redis_url else redis_url)
        return RedisCacheBackend(url=redis_url)
    cache_dir = os.environ.get("AURELIUS_CACHE_DIR")
    if cache_dir:
        logger.info("Using disk cache backend: %s", cache_dir)
        return DiskCacheBackend(directory=cache_dir)
    logger.info("Using in-memory DictCache (restart loses cache)")
    return DictCache()


# ---------------------------------------------------------------------------
# Lifespan — initialise pipeline once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _pipeline
    logging.basicConfig(level=logging.INFO)
    logger.info("Initialising Aurelius pipeline...")
    try:
        l2_cache = _make_cache_backend()
        pipeline = AureliusPipeline()
        pipeline.initialize()
        _pipeline = pipeline
        logger.info("Aurelius pipeline initialised successfully (cache: %s).", type(l2_cache).__name__)
    except Exception as exc:
        logger.error("Failed to initialise pipeline: %s", exc)
        _pipeline = None
    yield
    logger.info("Shutting down Aurelius API server.")
    _pipeline = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Project Aurelius API",
    version="11.0.0",
    description="REST API for the Aurelius molecule discovery pipeline",
    lifespan=lifespan,
)


def _get_pipeline() -> AureliusPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return _pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        pipeline_initialized=_pipeline is not None,
    )


@app.post(
    "/screen",
    dependencies=[Depends(verify_api_key), Depends(RateLimiter(30))],
)
async def screen(request: ScreenRequest) -> dict[str, Any]:
    pipeline = _get_pipeline()
    try:
        result = pipeline.screen_smiles(request.smiles)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/batch",
    dependencies=[Depends(verify_api_key), Depends(RateLimiter(10))],
)
async def batch(request: BatchRequest) -> list[dict[str, Any]]:
    """Screen multiple SMILES strings, one at a time.

    Each molecule is screened via ``screen_smiles`` (single-point
    ``MoleculeContext`` parsing inside the pipeline). Per-item failures are
    captured as ``{"is_viable": False, "error": ...}`` entries rather than
    aborting the whole batch, so a single bad SMILES does not discard valid
    results for the rest of the batch.
    """
    pipeline = _get_pipeline()
    output: list[dict[str, Any]] = []
    for smi in request.smiles:
        try:
            result = pipeline.screen_smiles(smi)
        except (ValueError, TypeError) as exc:
            output.append({
                "molecule_smiles": smi,
                "error": str(exc),
                "is_viable": False,
            })
        except Exception as exc:
            output.append({
                "molecule_smiles": smi,
                "error": str(exc),
                "is_viable": False,
            })
        else:
            output.append(result)
    return output
