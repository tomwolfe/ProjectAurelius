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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aurelius.pipeline import AureliusPipeline

logger = logging.getLogger(__name__)

_pipeline: AureliusPipeline | None = None


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


@app.post("/screen")
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


@app.post("/batch")
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
