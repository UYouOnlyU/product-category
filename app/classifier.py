from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import os
import warnings

try:
    # vertexai is provided by google-cloud-aiplatform
    from vertexai import init as vertexai_init  # type: ignore
    try:
        # Prefer stable import path if available
        from vertexai.generative_models import GenerativeModel, GenerationConfig  # type: ignore
    except Exception:  # fallback for older SDKs
        from vertexai.preview.generative_models import GenerativeModel, GenerationConfig  # type: ignore
    # Optionally silence known deprecation warning from the SDK to keep logs clean
    warnings.filterwarnings(
        "ignore",
        message=r".*deprecated.*genai-vertexai-sdk.*",
        category=UserWarning,
    )
except Exception:  # pragma: no cover - import-time fallback for environments without lib
    vertexai_init = None
    GenerativeModel = None  # type: ignore

from rapidfuzz import process, fuzz


def normalize_categories(categories: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for c in categories:
        mapping[_norm(c)] = c
    return mapping


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).casefold()


def _extract_json_array(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_score(v: object) -> float | None:
    if v is None:
        return None
    try:
        s = str(v).strip()
        if s.endswith('%'):
            f = float(s[:-1]) / 100.0
        else:
            f = float(s)
    except Exception:
        return None
    if f < 0:
        f = 0.0
    if f > 1:
        f = 1.0
    return f


class GeminiClassifier:
    def __init__(self, project: str, location: str, model_name: str, categories: List[str]):
        # Choose backend: vertexai (default) or googleai
        backend = (os.getenv("GENAI_BACKEND") or "vertexai").strip().lower()
        self._provider = None
        self._ga_client = None
        self._model = None
        self._model_name = model_name

        if backend == "googleai":
            try:
                from google import genai  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "google-genai is not installed. Install it or set GENAI_BACKEND=vertexai"
                ) from e
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Missing GOOGLE_API_KEY/GENAI_API_KEY for google-ai backend"
                )
            self._ga_client = genai.Client(api_key=api_key)
            self._provider = "googleai"
        else:
            if vertexai_init is None or GenerativeModel is None:
                # If vertexai not available, try google-ai automatically
                try:
                    from google import genai  # type: ignore
                    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
                    if not api_key:
                        raise RuntimeError(
                            "Vertex AI SDK unavailable and GOOGLE_API_KEY not set for google-ai fallback"
                        )
                    self._ga_client = genai.Client(api_key=api_key)
                    self._provider = "googleai"
                except Exception as e:
                    raise RuntimeError(
                        "No supported Generative AI backend available. Install google-cloud-aiplatform or google-genai."
                    ) from e
            else:
                vertexai_init(project=project, location=location)
                self._model = GenerativeModel(model_name)
                self._provider = "vertexai"
        self._categories = categories
        self._norm_map = normalize_categories(categories)

        # Pre-build catalog for fuzzy matching
        self._choices = list(self._norm_map.keys())

        # Build category ID mapping (C01, C02, ...) to reduce ambiguity and tokens
        self._codes: List[str] = [f"C{idx+1:02d}" for idx in range(len(categories))]
        self._id_to_name: Dict[str, str] = {code: name for code, name in zip(self._codes, categories)}
        self._name_to_id: Dict[str, str] = {name: code for code, name in self._id_to_name.items()}

        cats_lines = "\n".join(f"- {code}: {name}" for code, name in self._id_to_name.items())
        # System instruction: require IDs only, JSON only, with disambiguation rules
        self._system_instruction = (
            "You are a strict product categorizer.\n"
            "Use ONLY the category IDs from the allowed list (e.g., C01, C02).\n"
            "Choose the two most relevant categories per item and output JSON ONLY.\n"
            "Scores must be numeric between 0 and 1.\n\n"
            "Disambiguation rules:\n"
            "- Color words (orange, red, green, etc.) are colors by default unless explicit edible context (e.g., 'kg', 'fresh', 'fruit', 'juice', 'menu').\n"
            "- PPE/cleaning terms (glove, gloves, mask, gown, sanitizer, detergent, mop, etc.) must NOT map to food or beverage categories.\n"
            "- Meat terms (pork, bacon, beef, chicken, lamb, ham, sausage) map to 'Meat and Poultry', not produce or beverages.\n"
            "- Seafood terms (fish, shrimp, prawn, squid, crab, salmon, tuna) map to 'Seafood'.\n"
            "- Alcohol terms (beer, wine, whisky, spirits, liquor) map to alcohol categories, not non-alcoholic beverages.\n"
            "- Terms implying food (kg, g, ml, litre, pack, fresh, frozen, dried, sliced, smoked, fillet) prefer food categories over equipment/services.\n"
            "- Items mentioning 'juice' or edible fruits (e.g., 'orange', 'tangerine') MUST map to an appropriate beverage/food category, never hardware or pharmaceuticals.\n"
            "- Seaweed terms (seaweed, nori, kombu, wakame, kaiso, tosaka) map to 'Seafood' (or the closest edible sea vegetable category).\n\n"
            f"Allowed categories (ID: Name):\n{cats_lines}\n"
        )

    def classify_batch(
        self,
        descriptions: List[str],
        progress_every: int = 1,
        batch_size: int = 8,
        concurrency: int = 4,
        deduplicate: bool = True,
    ) -> List[Dict[str, object]]:
        log = logging.getLogger("classifier")

        total = len(descriptions)
        if total == 0:
            return []

        # Optional deduplication: classify unique descriptions only
        if deduplicate:
            norm_order: List[str] = []
            uniq_descs: List[str] = []
            seen: set[str] = set()
            for d in descriptions:
                k = _norm(d)
                norm_order.append(k)
                if k not in seen:
                    uniq_descs.append(d)
                    seen.add(k)
        else:
            uniq_descs = descriptions
            norm_order = [_norm(d) for d in descriptions]

        chunks: List[List[str]] = [
            uniq_descs[i : i + max(1, batch_size)] for i in range(0, len(uniq_descs), max(1, batch_size))
        ]

        done = 0
        preds_unique: List[Dict[str, object]] = []

        def do_chunk(chunk: List[str]) -> List[Dict[str, object]]:
            return self._classify_chunk(chunk)

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            future_map = {pool.submit(do_chunk, ch): idx for idx, ch in enumerate(chunks)}
            for fut in as_completed(future_map):
                res = fut.result()
                preds_unique.extend(res)
                done += len(res)
                interval = max(1, progress_every)
                if done % interval == 0 or done >= len(uniq_descs):
                    log.info(f"Classification progress | done={done}/{len(uniq_descs)} (unique)")

        # Map back to original order
        mapping: Dict[str, Dict[str, object]] = {}
        for d, p in zip(uniq_descs, preds_unique):
            mapping[_norm(d)] = p
        final: List[Dict[str, object]] = [mapping[k] for k in norm_order]
        return final

    def _gen_text(self, parts: List[str], want_json: bool = False) -> str:
        try:
            if self._provider == "vertexai":
                if want_json:
                    cfg = GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    )
                    resp = self._model.generate_content(parts, generation_config=cfg)
                else:
                    resp = self._model.generate_content(parts)
                return (getattr(resp, "text", "") or "").strip()
            elif self._provider == "googleai":
                cfg = {"temperature": 0.0}
                if want_json:
                    cfg["response_mime_type"] = "application/json"
                # google-genai: pass list of strings as contents
                resp = self._ga_client.models.generate_content(
                    model=self._model_name, contents=parts, config=cfg
                )
                return (getattr(resp, "text", "") or "").strip()
        except Exception:
            return ""
        return ""

    def _classify_chunk(self, descriptions: List[str]) -> List[Dict[str, object]]:
        if not descriptions:
            return []

        items = "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))
        prompt = (
            "You will receive a numbered list of item descriptions.\n"
            "For each item, choose the two most relevant category IDs (e.g., C01) from the allowed list.\n"
            "Return ONLY a JSON array with one object per item, no extra text.\n"
            "Each object must be: {\"c1\": <ID>, \"s1\": <0..1>, \"c2\": <ID>, \"s2\": <0..1>}\n\n"
            f"Items:\n{items}\n\n"
            "Respond with a JSON array of objects as specified."
        )
        try:
            text = self._gen_text([self._system_instruction, prompt], want_json=True)
            json_str = _extract_json_array(text)
            arr = json.loads(json_str)
            if isinstance(arr, list) and len(arr) == len(descriptions):
                out: List[Dict[str, object]] = []
                for obj in arr:
                    c1, s1, c2, s2 = self._validate_top2(obj)
                    out.append({"c1": c1, "s1": s1, "c2": c2, "s2": s2})
                return out
        except Exception:
            pass
        # Retry once with a shorter strict prompt
        try:
            strict = (
                "Return ONLY a JSON array of objects, one per item. Use category IDs only (e.g., C01). "
                "Each object: {\"c1\": <ID>, \"s1\": <0..1>, \"c2\": <ID>, \"s2\": <0..1>}"
            )
            text = self._gen_text([self._system_instruction, strict, items], want_json=True)
            json_str = _extract_json_array(text)
            arr = json.loads(json_str)
            if isinstance(arr, list) and len(arr) == len(descriptions):
                out = []
                for obj in arr:
                    c1, s1, c2, s2 = self._validate_top2(obj)
                    out.append({"c1": c1, "s1": s1, "c2": c2, "s2": s2})
                return out
        except Exception:
            pass
        # Fallback to single calls for this chunk (no guessing; return nulls if model fails)
        out: List[Dict[str, object]] = []
        for d in descriptions:
            c1, s1, c2, s2 = self._classify_single_top2(d)
            out.append({"c1": c1, "s1": s1, "c2": c2, "s2": s2})
        return out

    def _classify_single_top2(self, description: str) -> Tuple[str | None, float | None, str | None, float | None]:
        prompt = (
            f"Item description: {description}\n"
            "Choose the two most relevant category IDs from the allowed list (e.g., C01).\n"
            "Return ONLY JSON object: {\"c1\": <ID>, \"s1\": <0..1>, \"c2\": <ID>, \"s2\": <0..1>}"
        )
        try:
            text = self._gen_text([self._system_instruction, prompt], want_json=True)
            obj = json.loads(_extract_json_object(text))
            return self._validate_top2(obj)
        except Exception:
            # No guessing: return nulls to indicate missing/invalid model output
            return None, None, None, None

    # Removed unused single-label method to reduce surface area

    def _validate_top2(self, obj: object) -> Tuple[str | None, float | None, str | None, float | None]:
        if not isinstance(obj, dict):
            return None, None, None, None
        c1 = self._resolve_id_or_name(str(obj.get("c1", "")))
        c2_raw = str(obj.get("c2", ""))
        c2 = self._resolve_id_or_name(c2_raw)
        if c1 is None and c2 is None:
            return None, None, None, None
        if c2 == c1 and c1 is not None:
            c2 = self._second_best_different(c1, hint=c2_raw)
        s1 = _parse_score(obj.get("s1"))
        s2 = _parse_score(obj.get("s2"))
        return c1, s1, c2, s2

    def _second_best_different(self, first: str, hint: str | None = None) -> str:
        # Pick best other category different from first using fuzzy match against hint (or first)
        choices = [k for k in self._choices if self._norm_map[k] != first]
        if not choices:
            return first
        query = _norm(hint or first)
        match = None
        if choices:
            match, _, _ = process.extractOne(query, choices, scorer=fuzz.WRatio)
        return self._norm_map[match] if match else self._categories[0]

    def _post_validate_label(self, model_text: str) -> str:
        # Normalize and try exact normalization match
        if model_text:
            key = _norm(model_text)
            if key in self._norm_map:
                return self._norm_map[key]
        # Fallback to fuzzy match against allowed list
        query = _norm(model_text) if model_text else ""
        match, score, _ = process.extractOne(
            query, self._choices, scorer=fuzz.WRatio
        ) if self._choices else (None, 0, None)
        if match is not None and score >= 60:
            return self._norm_map[match]
        # If still not good, pick the first category as consistent fallback
        return self._categories[0]

    def _resolve_id_or_name(self, value: str) -> str:
        v = value.strip()
        # If it's an ID like C01, map to name
        if v in self._id_to_name:
            return self._id_to_name[v]
        # Otherwise treat as free text and validate
        return self._post_validate_label(v)
