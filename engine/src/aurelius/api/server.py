"""FastAPI server for Project Aurelius.

Provides REST endpoints for the Aurelius screening pipeline:
    POST /screen   — Screen a single molecule by SMILES
    POST /batch    — Screen multiple molecules by SMILES
    GET  /health   — Health check / status

Usage:
    uvicorn aurelius.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from aurelius.pipeline import AureliusPipeline

logger = logging.getLogger(__name__)

_pipeline: AureliusPipeline | None = None


# ---------------------------------------------------------------------------
# API Key authentication
# ---------------------------------------------------------------------------

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_AURELIUS_API_KEY: str | None = os.environ.get("AURELIUS_API_KEY")


async def verify_api_key(api_key: str | None = Depends(API_KEY_HEADER)) -> None:
    """Dependency that checks ``X-API-Key`` against ``AURELIUS_API_KEY``.

    When ``AURELIUS_API_KEY`` is unset (default) authentication is disabled
    and all requests are allowed.
    """
    if _AURELIUS_API_KEY is None:
        return
    if api_key != _AURELIUS_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# In-memory sliding-window rate limiter
# ---------------------------------------------------------------------------

_rate_store: dict[str, list[float]] = defaultdict(list)


class RateLimiter:
    """FastAPI-compatible rate-limit dependency.

    Parameters
    ----------
    limit : int
        Maximum number of requests allowed in the window.
    window : float
        Time window in seconds (default 60).
    """

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
    version: str = "10.2.0"
    pipeline_initialized: bool


# ---------------------------------------------------------------------------
# Lifespan — initialise pipeline once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _pipeline
    logging.basicConfig(level=logging.INFO)
    logger.info("Initialising Aurelius pipeline...")
    try:
        pipeline = AureliusPipeline()
        pipeline.initialize()
        _pipeline = pipeline
        logger.info("Aurelius pipeline initialised successfully.")
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
    version="10.2.0",
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
    """Return server status and pipeline initialisation state."""
    return HealthResponse(
        status="ok",
        pipeline_initialized=_pipeline is not None,
    )


@app.post(
    "/screen",
    dependencies=[Depends(verify_api_key), Depends(RateLimiter(30))],
)
async def screen(request: ScreenRequest) -> dict[str, Any]:
    """Screen a single molecule through the full Aurelius pipeline.

    Returns tier1, tier2, and score results.
    """
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
    """Screen multiple molecules through the full Aurelius pipeline.

    Returns a list of results in the same order as the input SMILES.
    """
    pipeline = _get_pipeline()
    results: list[dict[str, Any]] = []
    for smi in request.smiles:
        try:
            result = pipeline.screen_smiles(smi)
            results.append(result)
        except Exception as exc:
            results.append({
                "molecule_smiles": smi,
                "error": str(exc),
                "is_viable": False,
            })
    return results
