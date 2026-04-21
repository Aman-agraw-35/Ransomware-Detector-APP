# SeqDefender Backend (FastAPI)

## Run

1. Create/activate environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r backend/requirements.txt
```

3. Start API:

```powershell

```

## Endpoint

- POST /analyze
  - multipart/form-data
  - field name: file
  - supported: .exe, .dll
  - returns: verdict, confidence, attention labels/weights, SHAP top features, diagnostics

## Artifacts expected

By default, the API loads these files from artifacts_phase2:

- bilstm_attention_model.keras
- bilstm_attention_with_weights.keras
- scaler.pkl
- feature_names.json
- group_config.json

You can override paths with environment variables from backend/.env.example.
