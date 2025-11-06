from __future__ import annotations

import argparse
import json
import logging

from .config import load_config
from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export categorized product data: BigQuery -> Gemini -> CSV -> GCS",
    )
    parser.add_argument("--month", required=True, help="Month to query, format YYYYMM (legacy MM-YYYY also accepted)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of rows")
    parser.add_argument("--dry-run", action="store_true", help="Do not upload to GCS")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="How often to log progress counts (default: from .env CLASSIFY_PROGRESS_EVERY or 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of items per model call (default: from .env CLASSIFY_BATCH_SIZE or 8)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent model calls (default: from .env CLASSIFY_CONCURRENCY or 4)",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication of repeated descriptions",
    )

    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config()
    # Resolve performance knobs: CLI > .env > defaults
    batch_size = args.batch_size if args.batch_size is not None else (cfg.classify_batch_size or 8)
    concurrency = args.concurrency if args.concurrency is not None else (cfg.classify_concurrency or 4)
    progress_every = args.progress_every if args.progress_every is not None else (cfg.classify_progress_every or 1)
    result = run_pipeline(
        cfg,
        month=args.month,
        limit=args.limit,
        dry_run=args.dry_run,
        progress_every=max(1, progress_every),
        batch_size=max(1, batch_size),
        concurrency=max(1, concurrency),
        deduplicate=(not args.no_dedupe),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
