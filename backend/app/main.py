from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .ml_service import MLService
from .schemas import AnalyzeResponse


service: MLService | None = None
service_init_error: str | None = None
service_init_lock = Lock()
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".exe", ".dll"}


def _init_service() -> None:
    """Load MLService + warm-up TF graph in a background thread at startup."""
    global service, service_init_error
    try:
        svc = MLService()
        svc.warm_up()
        service = svc
        service_init_error = None
    except Exception as exc:  # noqa: BLE001
        service_init_error = str(exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    thread = threading.Thread(target=_init_service, daemon=True, name="ml-loader")
    thread.start()
    yield


app = FastAPI(title="SeqDefender API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> MLService:
    global service, service_init_error

    if service is not None:
        return service

    with service_init_lock:
        if service is not None:
            return service
        try:
            service = MLService()
            service_init_error = None
            return service
        except Exception as exc:
            service_init_error = str(exc)
            raise


@app.get("/health")
def health() -> dict:
    if service is not None:
        mock = service._mock
        mode = "heuristic" if mock else "bilstm_model"
    else:
        from pathlib import Path  # noqa: PLC0415
        root = Path(__file__).resolve().parents[2]
        has_artifacts = (root / "artifacts_phase2" / "bilstm_attention_model.keras").exists()
        mock = not has_artifacts
        mode = "bilstm_model" if has_artifacts else "heuristic"
    return {
        "status": "ok",
        "service_ready": service is not None,
        "inference_mode": mode,
        "mock_mode": mock,
        "service_init_error": service_init_error,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename.")

    extension = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .exe and .dll files are supported.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max size is 50 MB.")

    try:
        runtime_service = get_service()
        result = await run_in_threadpool(runtime_service.analyze_file, payload, file.filename)
        return AnalyzeResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
