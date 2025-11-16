"""Quick local prompt sanity test (no FastAPI required).

Usage:
  1) Ensure environment variables are set (see .env):
     GCP_PROJECT_ID, GCP_LOCATION, GEMINI_MODEL,
     CATEGORIES_PATH, GOOGLE_APPLICATION_CREDENTIALS, TABLE_ID,
     (optional) GENAI_BACKEND, GOOGLE_API_KEY/GENAI_API_KEY.
  2) pip install -r requirements.txt
  3) python quick_prompt_test.py

It prints model outputs for a fixed batch of sample items.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from app.classifier import GeminiClassifier
from app.config import load_config


def main() -> None:
    cfg = load_config()
    # Load categories
    cats_path = Path(cfg.categories_path)
    categories: List[str] = json.loads(cats_path.read_text(encoding="utf-8"))

    clf = GeminiClassifier(
        project=cfg.gcp_project_id,
        location=cfg.gcp_location,
        model_name=cfg.gemini_model,
        categories=categories,
    )

    samples = [
        "Tea cup : wooden",
    ]

    print("Running sample batch...\n")
    results = clf.classify_batch(samples, batch_size=5, concurrency=2, deduplicate=False)
    for text, res in zip(samples, results):
        print(f"{text}\n  -> category: {res.get('c1')} | score: {res.get('s1')} | j: {res.get('j')}\n")


if __name__ == "__main__":
    # Optional: set defaults for local dry-run
    os.environ.setdefault("PROMPT_MODE", "compact")
    main()

