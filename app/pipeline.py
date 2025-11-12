from __future__ import annotations

import csv
import json
import re
from datetime import datetime
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

from google.cloud import bigquery

from .config import Config
from .bq import query_invoices_by_month, iter_invoices_by_month
from .classifier import GeminiClassifier
from .storage import upload_to_gcs
from .cache import load_cache_from_gcs, save_cache_to_gcs


# Accept YYYYMM as primary; allow legacy MM-YYYY for backward compatibility
RE_YYYYMM = re.compile(r"^(19|20)\d\d(0[1-9]|1[0-2])$")
RE_MMYYYY = re.compile(r"^(0[1-9]|1[0-2])-(19|20)\d\d$")


def _to_yyyymm(month: str) -> str:
    """Normalize supported month formats to YYYYMM.

    Supports:
    - YYYYMM (returns as-is)
    - MM-YYYY (legacy; converts to YYYYMM)
    """
    m = month.strip()
    if RE_YYYYMM.match(m):
        return m
    if RE_MMYYYY.match(m):
        mm, yyyy = m.split("-")
        return f"{yyyy}{mm}"
    raise ValueError("month must be YYYYMM (or legacy MM-YYYY)")

# Removed unused heuristic post-corrections to keep model-only outputs


def run_pipeline(
    cfg: Config,
    month: str,
    limit: int | None = None,
    dry_run: bool = False,
    progress_every: int = 1,
    batch_size: int = 8,
    concurrency: int = 4,
    deduplicate: bool = True,
    org_ids: list[str] | None = None,
) -> Dict[str, Any]:
    log = logging.getLogger("pipeline")
    log.info(f"Starting pipeline | month={month} | limit={limit} | dry_run={dry_run}")
    # Validate and normalize month to YYYYMM for querying
    month_yyyymm = _to_yyyymm(month)

    # Load allowed categories
    with open(cfg.categories_path, "r", encoding="utf-8") as f:
        categories: List[str] = json.load(f)
        if not isinstance(categories, list) or not all(isinstance(x, str) for x in categories):
            raise ValueError("allowed_categories.json must be a JSON array of strings")
        if not categories:
            raise ValueError("allowed_categories.json must not be empty")
    log.info(f"Loaded allowed categories | count={len(categories)}")

    # BigQuery client and iterator
    bq_client = bigquery.Client(project=cfg.gcp_project_id)
    log.info(
        f"Querying BigQuery | table={cfg.table_id} | month={month_yyyymm} | limit={limit} | org_ids={org_ids if org_ids else 'ALL'}"
    )
    row_iter = iter_invoices_by_month(bq_client, cfg.table_id, month_yyyymm, org_ids, limit)

    # Classify
    log.info(f"Initializing Gemini classifier | model={cfg.gemini_model} | location={cfg.gcp_location}")
    classifier = GeminiClassifier(
        project=cfg.gcp_project_id,
        location=cfg.gcp_location,
        model_name=cfg.gemini_model,
        categories=categories,
    )

    # Optional cross-run cache
    cache_bucket = os.getenv("CACHE_GCS_BUCKET")
    cache_blob = os.getenv("CACHE_GCS_BLOB")
    cache: Dict[str, tuple[str | None, float | None, str | None]] = {}
    if cache_bucket and cache_blob:
        try:
            cache = load_cache_from_gcs(cache_bucket, cache_blob)
            log.info(f"Loaded cache | entries={len(cache)} from gs://{cache_bucket}/{cache_blob}")
        except Exception:
            log.warning("Cache load failed; proceeding without cache")

    # Streaming: classify and write CSV chunk-by-chunk
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_name = f"product-category_{month}_{ts}.csv"
    out_dir = Path(os.getenv("OUTPUT_DIR", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / file_name

    row_chunk_size = int(os.getenv("ROW_CHUNK_SIZE", "500") or 500)
    min_desc = int(os.getenv("MIN_DESC_CHARS", "2") or 2)
    pred_cols = ["predict category", "score", "short justification"]
    log.info(f"Streaming CSV | path={str(local_path)} | row_chunk_size={row_chunk_size}")
    total_rows = 0
    missing_score_count = 0
    missing_any_count = 0
    written = 0

    with open(local_path, "w", encoding="utf-8", newline="") as f:
        writer = None
        chunk: List[Dict[str, Any]] = []
        for row in row_iter:
            chunk.append(row)
            if len(chunk) >= row_chunk_size:
                written += _process_chunk(
                    classifier,
                    chunk,
                    writer_ref := [writer],
                    f,
                    pred_cols,
                    progress_every,
                    batch_size,
                    concurrency,
                    deduplicate,
                    min_desc,
                    cache,
                    log,
                )
                writer = writer_ref[0]
                total_rows += len(chunk)
                chunk = []
        if chunk:
            written += _process_chunk(
                classifier,
                chunk,
                writer_ref := [writer],
                f,
                pred_cols,
                progress_every,
                batch_size,
                concurrency,
                deduplicate,
                min_desc,
                cache,
                log,
            )
            writer = writer_ref[0]
            total_rows += len(chunk)
        log.info(f"CSV written | rows={written}")

    # Save cache back if enabled
    if cache_bucket and cache_blob:
        try:
            save_cache_to_gcs(cache_bucket, cache_blob, cache)
            log.info(f"Saved cache | entries={len(cache)} to gs://{cache_bucket}/{cache_blob}")
        except Exception:
            log.warning("Cache save failed; continuing")

    # Upload to GCS unless dry_run
    gcs_uri = None
    if not dry_run:
        # Upload directly under the configured prefix without month subfolder
        blob_path = f"{cfg.gcs_output_prefix.rstrip('/')}/{file_name}"
        log.info(f"Uploading to GCS | bucket={cfg.gcs_bucket} | blob={blob_path}")
        gcs_uri = upload_to_gcs(cfg.gcs_bucket, blob_path, str(local_path))
        log.info(f"Uploaded to GCS | gcs_uri={gcs_uri}")

    return {
        "month": month,
        "total_rows": written,
        "processed": written,
        "local_csv": str(local_path),
        "gcs_uri": gcs_uri,
    }


def _process_chunk(
    classifier: GeminiClassifier,
    rows: List[Dict[str, Any]],
    writer_ref: List[csv.DictWriter | None],
    f,
    pred_cols: List[str],
    progress_every: int,
    batch_size: int,
    concurrency: int,
    deduplicate: bool,
    min_desc: int,
    cache: Dict[str, tuple[str | None, float | None, str | None]],
    log: logging.Logger,
) -> int:
    # Prepare descriptions with guard
    descs: List[str] = []
    empties: List[int] = []
    for i, r in enumerate(rows):
        d = str(r.get("item_description") or "").strip()
        if len(d) < min_desc:
            empties.append(i)
            descs.append("")
        else:
            descs.append(d)

    # Use cache to short-circuit
    to_classify_idx: List[int] = []
    preds: List[Dict[str, Any] | None] = [None] * len(rows)
    for i, d in enumerate(descs):
        if i in empties:
            preds[i] = {"c1": None, "s1": None, "j": "missing/too short description"}
        else:
            k = _norm_local(d)
            if cache and k in cache:
                c1, s1, j = cache[k]
                preds[i] = {"c1": c1, "s1": s1, "j": j}
            else:
                to_classify_idx.append(i)

    if to_classify_idx:
        texts = [descs[i] for i in to_classify_idx]
        cls = classifier.classify_batch(
            texts,
            progress_every=progress_every,
            batch_size=batch_size,
            concurrency=concurrency,
            deduplicate=deduplicate,
        )
        for i, pr in zip(to_classify_idx, cls):
            preds[i] = pr
            k = _norm_local(descs[i])
            cache[k] = (pr.get("c1"), pr.get("s1"), pr.get("j"))

    # Write out
    writer = writer_ref[0]
    written = 0
    for i, r in enumerate(rows):
        pr = preds[i] or {"c1": None, "s1": None, "j": None}
        rr = dict(r)
        rr["predict category"] = pr.get("c1")
        rr["score"] = pr.get("s1")
        rr["short justification"] = pr.get("j")
        if writer is None:
            base_fields = [k for k in rr.keys() if k not in pred_cols]
            fieldnames = base_fields + pred_cols
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer_ref[0] = writer
        writer.writerow(rr)
        written += 1
        if written % max(1, progress_every) == 0 or i == len(rows) - 1:
            log.info(f"CSV progress | chunk_written={written}/{len(rows)}")
    return written


def _norm_local(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()
