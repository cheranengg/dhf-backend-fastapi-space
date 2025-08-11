# dhf-backend-fastapi-space
DHF Remediation
# DHF Backend – Hugging Face Spaces

This is the backend for the Design History File (DHF) Automation system.
It exposes endpoints for:
- Hazard Analysis (HA)
- Design Verification Protocol (DVP)
- Trace Matrix (TM)
- Guardrail validation

## 1. Space Type & Hardware
- **Space Type:** Docker
- **Hardware:** Select GPU (T4 or A10G, depending on model size)

## 2. Environment Variables
Set these in your HF Space **Settings → Variables and secrets**:

| Key                      | Value / Description |
|--------------------------|----------------------|
| `BACKEND_TOKEN`          | Any strong API key (same used by frontend Streamlit app) |
| `HA_MODEL_MERGED_DIR`    | Path in repo where merged HA model is stored (e.g., `/app/models/ha_merged`) |
| `DVP_MODEL_DIR`          | Path in repo where fine-tuned DVP model is stored |
| `TM_MODEL_DIR`           | Path in repo where fine-tuned TM model is stored |
| `SERVICE_ACCOUNT_JSON`   | Google Drive API service account JSON content |
| `DRIVE_FOLDER_ID`        | Google Drive folder ID for outputs |
| `OUTPUT_DIR`             | Output dir inside container (e.g., `/app/outputs`) |

> **Tip:** For large model files, use HF `git-lfs` and keep them in `/models`.

## 3. Run Locally (Optional)
```bash
docker build -t dhf-backend .
docker run -it -p 7860:7860 \
  -e BACKEND_TOKEN=change-me \
  -e HA_MODEL_MERGED_DIR=/app/models/ha_merged \
  -e DVP_MODEL_DIR=/app/models/dvp \
  -e TM_MODEL_DIR=/app/models/tm \
  dhf-backend


curl -s -H "Authorization: Bearer change-me" https://<space-url>/health
curl -s -X POST https://<space-url>/hazard-analysis \
  -H "Authorization: Bearer change-me" -H "Content-Type: application/json" \
  -d '{"requirements":[{"requirement_id":"REQ-001","requirements":"Infusion system shall detect air-in-line within 1 second"}]}'





