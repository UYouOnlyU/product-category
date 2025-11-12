from __future__ import annotations

import json
import io
import re
from typing import Dict, Tuple
from google.cloud import storage


_SPACE_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    t = (text or "").strip()
    t = _SPACE_RE.sub(" ", t)
    return t.casefold()


def load_cache_from_gcs(bucket: str, blob_name: str) -> Dict[str, Tuple[str | None, float | None, str | None]]:
    client = storage.Client()
    b = client.bucket(bucket)
    blob = b.blob(blob_name)
    if not blob.exists():
        return {}
    data = blob.download_as_bytes()
    try:
        # JSONL mapping: each line {"k": <norm>, "c1": ..., "s1": ..., "j": ...}
        cache: Dict[str, Tuple[str | None, float | None, str | None]] = {}
        for line in io.BytesIO(data).read().decode("utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            k = str(obj.get("k", ""))
            c1 = obj.get("c1")
            s1 = obj.get("s1")
            j = obj.get("j")
            cache[k] = (c1, float(s1) if s1 is not None else None, j)
        return cache
    except Exception:
        return {}


def save_cache_to_gcs(bucket: str, blob_name: str, cache: Dict[str, Tuple[str | None, float | None, str | None]]) -> None:
    client = storage.Client()
    b = client.bucket(bucket)
    blob = b.blob(blob_name)
    buf = io.StringIO()
    for k, (c1, s1, j) in cache.items():
        buf.write(json.dumps({"k": k, "c1": c1, "s1": s1, "j": j}, ensure_ascii=False))
        buf.write("\n")
    blob.upload_from_string(buf.getvalue(), content_type="application/jsonl")

