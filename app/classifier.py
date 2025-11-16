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
    # Normalize to one decimal place (0.0, 0.1, ..., 1.0)
    return round(f, 1)


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

        cats_lines = "\n".join(f"- {code}: {name}" for code, name in self._id_to_name.items())
        # Build system prompt (compact by default) to reduce tokens
        compact = (os.getenv("PROMPT_MODE") or "compact").strip().lower() == "compact"
        if compact:
            self._system_instruction = (
                "Classify products into allowed categories by ID only.\n"
                "Output JSON only. One object per item: {\"i\":<n>,\"c1\":<ID>,\"s1\":<0..1 with one decimal place (0.0..1.0)>,\"j\":<<=20w>}\n"
                "Rules: edible terms -> food/beverage; PPE/cleaning -> not food; seaweed -> seafood.\n"
                "If description has 'Name : details', use the product name before ':' as the primary signal (e.g., 'Coffee Bean : Breakfast Blend' -> use 'Coffee Bean').\n"
                "Quick examples (target categories): condensed/sweetened milk -> Dairy and Eggs; soy milk -> Dairy and Eggs; lemon/paprika/peppercorn powders -> Fruit and Vegetables; tomato/fish sauce -> Food Preparation Ingredients; tortillas/wraps -> Bakery; tea or drink syrups -> Nonalcoholic Beverages.\n"
                "More examples: mounted stone (shank cone) -> Plants and Flowers; projector screen -> Computers and Communications; furikake (seaweed/bonito flakes) -> Prepared Food and Ingredients; pickled vegetable mix (fukujinzuke) -> Fruit and Vegetables; sports/athletic gloves -> Sport and Recreation.\n\n"
                f"Allowed (ID: Name):\n{cats_lines}\n"
            )
        else:
            self._system_instruction = (
                "You are a strict product categorizer.\n"
                "Use ONLY the category IDs from the allowed list (e.g., C01, C02).\n"
                "Choose the most relevant category per item and output JSON ONLY.\n"
                "Score s1 must be a decimal probability in [0,1] with one decimal place (0.0, 0.1, ..., 1.0).\n\n"
                "Disambiguation rules:\n"
                "- If a description is 'Name : details', use the product name before ':' as the main cue; trailing details may refine but must not override the base product (e.g., 'Coffee Bean : Breakfast Blend' -> treat as Coffee Bean).\n"
                "- Quick examples: condensed/sweetened/evap milk or milk powder -> Dairy and Eggs; soy milk -> Dairy and Eggs; lemon/paprika/peppercorn powders -> Fruit and Vegetables; tomato/fish/soy/BBQ sauces -> Food Preparation Ingredients (not seafood/meals); tortillas/wraps/flatbreads -> Bakery; tea/coffee and drink syrups -> Nonalcoholic Beverages.\n"
                "- More examples: mounted stone (shank/cone) -> Plants and Flowers; projector screen -> Computers and Communications; furikake (seaweed + bonito flakes) -> Prepared Food and Ingredients; pickled vegetable mix (fukujinzuke) -> Fruit and Vegetables; sports/athletic gloves -> Sport and Recreation.\n"
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

        # Optional prompt-size optimization: truncate long descriptions
        try:
            max_len_env = os.getenv("MAX_DESC_CHARS")
            max_len = int(max_len_env) if max_len_env else 0
        except Exception:
            max_len = 0
        if max_len and max_len > 0:
            descriptions = [
                (d if len(d) <= max_len else d[: max_len]) if isinstance(d, str) else ""
                for d in descriptions
            ]

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

        step = max(1, batch_size)
        chunks: List[List[str]] = []
        starts: List[int] = []
        i = 0
        while i < len(uniq_descs):
            ch = uniq_descs[i : i + step]
            chunks.append(ch)
            starts.append(i)
            i += len(ch)

        done = 0
        preds_unique: List[Dict[str, object] | None] = [None] * len(uniq_descs)

        def do_chunk(chunk: List[str]) -> List[Dict[str, object]]:
            return self._classify_chunk(chunk)

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            future_map = {pool.submit(do_chunk, ch): idx for idx, ch in enumerate(chunks)}
            for fut in as_completed(future_map):
                idx = future_map[fut]
                start = starts[idx]
                res = fut.result()
                preds_unique[start : start + len(res)] = res
                done += len(res)
                interval = max(1, progress_every)
                if done % interval == 0 or done >= len(uniq_descs):
                    log.info(f"Classification progress | done={done}/{len(uniq_descs)} (unique)")

        # Map back to original order
        if deduplicate:
            mapping: Dict[str, Dict[str, object]] = {}
            for d, p in zip(uniq_descs, preds_unique):
                if p is None:
                    p = {"c1": None, "s1": None, "j": None}
                mapping[_norm(d)] = p
            final: List[Dict[str, object]] = [mapping[k] for k in norm_order]
        else:
            # No dedupe: preds_unique already aligned 1:1 with descriptions
            final = [p if p is not None else {"c1": None, "s1": None, "j": None} for p in preds_unique]
        return final

    def _gen_text(self, parts: List[str], want_json: bool = False) -> str:
        # Basic retry with jitter for transient errors
        import time, random
        max_attempts = max(1, int(os.getenv("GENAI_RETRIES", "2") or 2))
        base_ms = max(50, int(os.getenv("GENAI_RETRY_BASE_MS", "200") or 200))
        last_text = ""
        try:
            temp_env = os.getenv("GENAI_TEMPERATURE")
            temperature = float(temp_env) if temp_env is not None else 0.0
        except Exception:
            temperature = 0.0
        for attempt in range(1, max_attempts + 1):
            try:
                if self._provider == "vertexai":
                    if want_json:
                        cfg = GenerationConfig(
                            response_mime_type="application/json",
                            temperature=temperature,
                        )
                        resp = self._model.generate_content(parts, generation_config=cfg)
                    else:
                        if temperature and temperature > 0:
                            cfg = GenerationConfig(temperature=temperature)
                            resp = self._model.generate_content(parts, generation_config=cfg)
                        else:
                            resp = self._model.generate_content(parts)
                    return (getattr(resp, "text", "") or "").strip()
                elif self._provider == "googleai":
                    cfg = {"temperature": temperature}
                    if want_json:
                        cfg["response_mime_type"] = "application/json"
                    resp = self._ga_client.models.generate_content(
                        model=self._model_name, contents=parts, config=cfg
                    )
                    return (getattr(resp, "text", "") or "").strip()
            except Exception:
                # Sleep with full jitter
                if attempt < max_attempts:
                    time.sleep((base_ms / 1000.0) * (2 ** (attempt - 1)) * random.uniform(0.5, 1.5))
        return last_text

    def _classify_chunk(self, descriptions: List[str]) -> List[Dict[str, object]]:
        if not descriptions:
            return []

        items = "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))
        prompt = (
            "You will receive a numbered list of item descriptions.\n"
            "For each item, choose the most relevant category ID (e.g., C01) from the allowed list.\n"
            "If an item looks like 'Name : details', use the product name before ':' as the primary cue; details may refine but not override.\n"
            "Return ONLY a JSON array with one object per item, no extra text.\n"
            "Each object must be: {\"i\": <item number>, \"c1\": <ID>, \"s1\": <0..1 with one decimal place>, \"j\": <short justification in <= 20 words>}\n"
            "Ensure i matches the numbered list below so results can be mapped back exactly.\n\n"
            f"Items:\n{items}\n\n"
            "Respond with a JSON array of objects as specified."
        )
        try:
            text = self._gen_text([self._system_instruction, prompt], want_json=True)
            json_str = _extract_json_array(text)
            arr = json.loads(json_str)
            if isinstance(arr, list):
                # Prefer index-based assembly to guarantee alignment
                indexed: List[Dict[str, object] | None] = [None] * len(descriptions)
                all_have_index = True
                seen = set()
                for obj in arr:
                    try:
                        idx = int(str((obj.get("i") if isinstance(obj, dict) else None) or "0"))
                    except Exception:
                        idx = 0
                    if 1 <= idx <= len(descriptions) and idx not in seen:
                        c1, s1 = self._validate_top1(obj)
                        j = self._get_justification(obj)
                        indexed[idx - 1] = {"c1": c1, "s1": s1, "j": j}
                        seen.add(idx)
                    else:
                        all_have_index = False
                if all_have_index and all(v is not None for v in indexed):
                    return [v for v in indexed if v is not None]
                # Fallback to sequential mapping if indices are missing
                if len(arr) == len(descriptions):
                    out: List[Dict[str, object]] = []
                    for obj in arr:
                        c1, s1 = self._validate_top1(obj)
                        j = self._get_justification(obj)
                        out.append({"c1": c1, "s1": s1, "j": j})
                    return out
        except Exception:
            pass
        # Retry once with a shorter strict prompt
        try:
            strict = (
                "Return ONLY a JSON array of objects, one per item. Use category IDs only (e.g., C01). "
                "Each object: {\"i\": <item number>, \"c1\": <ID>, \"s1\": <0..1 with one decimal place>, \"j\": <short justification in <= 20 words>}"
            )
            text = self._gen_text([self._system_instruction, strict, items], want_json=True)
            json_str = _extract_json_array(text)
            arr = json.loads(json_str)
            if isinstance(arr, list):
                indexed: List[Dict[str, object] | None] = [None] * len(descriptions)
                all_have_index = True
                seen = set()
                for obj in arr:
                    try:
                        idx = int(str((obj.get("i") if isinstance(obj, dict) else None) or "0"))
                    except Exception:
                        idx = 0
                    if 1 <= idx <= len(descriptions) and idx not in seen:
                        c1, s1 = self._validate_top1(obj)
                        j = self._get_justification(obj)
                        indexed[idx - 1] = {"c1": c1, "s1": s1, "j": j}
                        seen.add(idx)
                    else:
                        all_have_index = False
                if all_have_index and all(v is not None for v in indexed):
                    return [v for v in indexed if v is not None]
                if len(arr) == len(descriptions):
                    out: List[Dict[str, object]] = []
                    for obj in arr:
                        c1, s1 = self._validate_top1(obj)
                        j = self._get_justification(obj)
                        out.append({"c1": c1, "s1": s1, "j": j})
                    return out
        except Exception:
            pass
        # Fallback to single calls for this chunk (no guessing; return nulls if model fails)
        out: List[Dict[str, object]] = []
        for d in descriptions:
            c1, s1, j = self._classify_single_top1(d)
            out.append({"c1": c1, "s1": s1, "j": j})
        return out

    def _classify_single_top1(self, description: str) -> Tuple[str | None, float | None, str | None]:
        prompt = (
            f"Item description: {description}\n"
            "Choose the most relevant category ID from the allowed list (e.g., C01).\n"
            "Return ONLY JSON object: {\"c1\": <ID>, \"s1\": <0..1 with one decimal place>, \"j\": <short justification in <= 20 words>}"
        )
        try:
            text = self._gen_text([self._system_instruction, prompt], want_json=True)
            obj = json.loads(_extract_json_object(text))
            c1, s1 = self._validate_top1(obj)
            j = self._get_justification(obj)
            return c1, s1, j
        except Exception:
            # No guessing: return nulls to indicate missing/invalid model output
            return None, None, None

    # Removed unused single-label method to reduce surface area

    def _validate_top1(self, obj: object) -> Tuple[str | None, float | None]:
        if not isinstance(obj, dict):
            return None, None
        c1 = self._resolve_id_or_name(str(obj.get("c1", "")))
        s1 = _parse_score(obj.get("s1"))
        return c1, s1

    def _get_justification(self, obj: object) -> str | None:
        if isinstance(obj, dict):
            v = obj.get("j") or obj.get("justification") or obj.get("reason")
            if isinstance(v, str):
                t = v.strip()
            elif v is not None:
                t = str(v)
            else:
                return None
            return t[:200]
        return None

    # Removed: second-best helper (no longer used in single-category mode)

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
