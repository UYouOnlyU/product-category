Cloud Run deployment guide

Overview
- Container exposes a FastAPI service with:
  - POST /run: trigger pipeline for a month (MM-YYYY)
  - GET /healthz: health check
- Uses Workload Identity: no JSON keys in the image. The Cloud Run service account authorizes access to BigQuery, Vertex AI, and GCS.

Prerequisites
- gcloud CLI installed and authenticated
- Project ID available (replace <PROJECT_ID>)
- Vertex AI region chosen (e.g., us-central1)

Enable APIs
```
gcloud config set project <PROJECT_ID>
gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com
```

Create Artifact Registry (one time)
```
gcloud artifacts repositories create pcv-repo \
  --repository-format=docker \
  --location=us-central1
```

Build and push image
```
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/<PROJECT_ID>/pcv-repo/product-category-api:latest
```

Create/choose Cloud Run service account
```
gcloud iam service-accounts create product-category-sa \
  --display-name "Product Category API"

# Grant minimal roles (tighten to dataset/bucket scope if desired)
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member serviceAccount:product-category-sa@<PROJECT_ID>.iam.gserviceaccount.com \
  --role roles/aiplatform.user

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member serviceAccount:product-category-sa@<PROJECT_ID>.iam.gserviceaccount.com \
  --role roles/bigquery.readSessionUser

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member serviceAccount:product-category-sa@<PROJECT_ID>.iam.gserviceaccount.com \
  --role roles/bigquery.dataViewer

# Storage (restrict to target bucket in production)
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member serviceAccount:product-category-sa@<PROJECT_ID>.iam.gserviceaccount.com \
  --role roles/storage.objectAdmin
```

Deploy to Cloud Run
```
gcloud run deploy product-category-api \
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/pcv-repo/product-category-api:latest \
  --region us-central1 \
  --port 8080 \
  --service-account product-category-sa@<PROJECT_ID>.iam.gserviceaccount.com \
  --timeout=900s \
  --memory=1Gi \
  --set-env-vars \
    GCP_PROJECT_ID=<PROJECT_ID>,\
    GCP_LOCATION=us-central1,\
    GEMINI_MODEL=gemini-1.5-pro-002,\
    TABLE_ID=<PROJECT_ID>.<DATASET>.<TABLE>,\
    CATEGORIES_PATH=/app/allowed_categories.json,\
    GCS_BUCKET=<BUCKET>,\
    GCS_OUTPUT_PREFIX=MBTH/product-category,\
    CLASSIFY_BATCH_SIZE=8,\
    CLASSIFY_CONCURRENCY=4,\
    FRONTEND_ORIGINS=https://your-frontend.example.com,\
    API_KEY=<strong_random_value>,\
    GENAI_BACKEND=vertexai

# For public testing only (remove for authenticated-only):
#   --allow-unauthenticated
```

Note on auth
- Recommended: keep the service authenticated. Grant callers `roles/run.invoker` or front with a backend/IAP.
- Test with: `curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" ...`

Call the API
```
SERVICE_URL=$(gcloud run services describe product-category-api --region us-central1 \
  --format='value(status.url)')

# Authenticated example
ID_TOKEN=$(gcloud auth print-identity-token)
curl -X POST "$SERVICE_URL/run" \
  -H "Authorization: Bearer $ID_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "month": "202509",
    "limit": 100,
    "dry_run": false,
    "batch_size": 8,
    "concurrency": 4
  }'
```

Frontend integration
- CORS: configure allowed origins via `FRONTEND_ORIGINS` (comma-separated). Use `*` for testing only.
- Simple API key: set `API_KEY` and send it from the browser with header `X-API-Key: <value>` or `Authorization: Bearer <value>`.
- For stronger auth, keep the service private (no `--allow-unauthenticated`) and front it with API Gateway or Identity-Aware Proxy.

Operational tips
- Adjust `CLASSIFY_BATCH_SIZE` and `CLASSIFY_CONCURRENCY` based on Vertex AI quotas and latency.
- Backend selection: Set `GENAI_BACKEND=vertexai` (default) to use Vertex AI with GCP identity. To use Google AI with API key, set `GENAI_BACKEND=googleai` and `GOOGLE_API_KEY=<key>` and ensure `google-genai` is installed in the image.
- If jobs may exceed 15 minutes, increase `--timeout` or consider an async task pattern.
- Logs are available in Cloud Logging (look for `pipeline`, `classifier`, `bq`, `gcs`, `server`).

