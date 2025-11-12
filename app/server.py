from __future__ import annotations

import logging
from typing import Any, Dict

import os
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import load_config
from .pipeline import run_pipeline


log = logging.getLogger("server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


"""/run uses query parameters; no request body model needed."""


app = FastAPI(title="Product Category API")

# Configure CORS for frontend access
_origins_env = os.getenv("FRONTEND_ORIGINS", "*")
_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    max_age=600,
)

# Optional API key for simple auth from frontend
_api_key_required = os.getenv("API_KEY")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}

# Additional simple health endpoints (some networks proxy-filter /healthz)
@app.get("/")
def root_health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/readyz")
def readyz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run(
    # Core parameters now accepted as query params
    month: str = Query(..., description="Month to query, format YYYYMM (legacy MM-YYYY also accepted)"),
    limit: int | None = Query(None, ge=1, description="Optional limit of rows"),
    dry_run: bool = Query(False, alias="dry-run", description="Do not upload to GCS"),
    org_id_raw: str | None = Query(
        None,
        alias="org_ID",
        description="Comma-separated org_ID list, e.g. 10001,10002 (defaults to ALL)",
    ),
    # Optional performance knobs (also via query)
    progress_every: int | None = Query(
        None, ge=1, description="How often to log progress counts"
    ),
    batch_size: int | None = Query(
        None, ge=1, description="Number of items per model call"
    ),
    concurrency: int | None = Query(
        None, ge=1, description="Number of concurrent model calls"
    ),
    deduplicate: bool = Query(True, description="Deduplicate repeated descriptions"),
    # Simple API key headers
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Dict[str, Any]:
    # Simple API key check (via X-API-Key or Authorization: Bearer <key>) if configured
    if _api_key_required:
        provided = None
        if x_api_key:
            provided = x_api_key
        elif authorization and authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        if provided != _api_key_required:
            raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        cfg = load_config()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # Resolve performance knobs: Query > .env > defaults
        eff_batch_size = batch_size if batch_size is not None else (cfg.classify_batch_size or 8)
        eff_concurrency = concurrency if concurrency is not None else (cfg.classify_concurrency or 4)
        eff_progress_every = (
            progress_every if progress_every is not None else (cfg.classify_progress_every or 1)
        )

        # Parse org_IDs if provided
        org_ids = None
        if org_id_raw:
            org_ids = [s for s in (x.strip() for x in org_id_raw.split(",")) if s]

        log.info(
            "API run | month=%s | limit=%s | dry_run=%s | batch_size=%s | concurrency=%s | progress_every=%s | dedupe=%s",
            month,
            limit,
            dry_run,
            eff_batch_size,
            eff_concurrency,
            eff_progress_every,
            deduplicate,
        )
        result = run_pipeline(
            cfg,
            month=month,
            limit=limit,
            dry_run=dry_run,
            progress_every=max(1, eff_progress_every),
            batch_size=max(1, eff_batch_size),
            concurrency=max(1, eff_concurrency),
            deduplicate=deduplicate,
            org_ids=org_ids,
        )
        return result
    except ValueError as e:
        # Validation errors (e.g., month format)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail="Internal server error")

