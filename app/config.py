import os
from dataclasses import dataclass
from typing import Optional


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(override=False)


@dataclass(frozen=True)
class Config:
    gcp_project_id: str
    gcp_location: str
    gemini_model: str
    service_account_path: Optional[str]
    table_id: str
    categories_path: str
    gcs_bucket: str
    gcs_output_prefix: str
    classify_batch_size: Optional[int]
    classify_concurrency: Optional[int]
    classify_progress_every: Optional[int]
    invoice_date_field: str
    org_field: str
    item_description_field: str


def load_config() -> Config:
    _load_dotenv_if_available()

    service_account = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if service_account:
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", service_account)

    cfg = Config(
        gcp_project_id=_require("GCP_PROJECT_ID"),
        gcp_location=_require("GCP_LOCATION"),
        gemini_model=_require("GEMINI_MODEL"),
        service_account_path=service_account,
        table_id=_require("TABLE_ID"),
        categories_path=_require("CATEGORIES_PATH"),
        gcs_bucket=_require("GCS_BUCKET"),
        gcs_output_prefix=_require("GCS_OUTPUT_PREFIX"),
        classify_batch_size=_get_int("CLASSIFY_BATCH_SIZE"),
        classify_concurrency=_get_int("CLASSIFY_CONCURRENCY"),
        classify_progress_every=_get_int("CLASSIFY_PROGRESS_EVERY"),
        invoice_date_field=_get_str("INVOICE_DATE_FIELD", "check_invoice_date"),
        org_field=_get_str("ORG_FIELD", "check_organization"),
        item_description_field=_get_str("ITEM_DESCRIPTION_FIELD", "item_description"),
    )
    return cfg


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


def _get_int(key: str) -> Optional[int]:
    val = os.getenv(key)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        raise RuntimeError(f"Environment variable {key} must be an integer if set")


def _get_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default
