#!/usr/bin/env python3
"""
Training server — sirve el dashboard HTML de monitoreo de entrenamiento.

Usage:
    python scripts/training_server.py
    python scripts/training_server.py --port 8765

Endpoints:
    GET /              → dashboard HTML
    GET /status        → training_status.json
    GET /results       → loop_engineering.json
    GET /runs          → lista de all_runs
    GET /canonical     → métricas del modelo canónico
    GET /wc_fixtures   → World Cup 2026 group fixtures
    GET /wc_predictions→ World Cup 2026 group predictions
    POST /wc_predict   → predict one match
    POST /wc_regenerate→ regenerate group-stage predictions
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from scripts.agents.check_gpu_usage import audit as audit_gpu
from scripts.training_orchestrator import REPORT_PATH as ORCHESTRATOR_REPORT_PATH
from scripts.wc2026_engine import (
    load_fixtures,
    load_predictions,
    predict_single,
    regenerate_predictions,
)

ROOT = Path(__file__).parent.parent
DASHBOARD_DIR = ROOT / "dashboard"
MODELS_DIR = ROOT / "data" / "models"
STATUS_PATH = MODELS_DIR / "training_status.json"
RESULTS_PATH = MODELS_DIR / "loop_engineering.json"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

app = FastAPI(title="Mondial-Xboost Training Monitor")

# Proceso de entrenamiento adaptivo en segundo plano.
_adaptive_process: subprocess.Popen | None = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    index_path = DASHBOARD_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Dashboard not found</h1>"


@app.get("/status")
async def status() -> JSONResponse:
    if STATUS_PATH.exists():
        data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        return JSONResponse(content=data)
    return JSONResponse(content={"status": "idle", "message": "No training status found"})


@app.get("/results")
async def results() -> JSONResponse:
    if RESULTS_PATH.exists():
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        return JSONResponse(content=data)
    return JSONResponse(content={"batches": [], "all_runs": [], "best": None})


@app.get("/runs")
async def runs() -> JSONResponse:
    if RESULTS_PATH.exists():
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        return JSONResponse(content=data.get("all_runs", []))
    return JSONResponse(content=[])


@app.get("/canonical")
async def canonical() -> JSONResponse:
    def _load_manifest(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _extract_metrics(data: dict | None) -> dict:
        if not data:
            return {}
        metrics = data.get("metrics", {}) or {}
        # Accept either top-level keys or nested under "metrics".
        return {
            "accuracy": metrics.get("accuracy"),
            "log_loss": metrics.get("log_loss"),
            "top_feature": metrics.get("top_feature"),
            "source": metrics.get("source"),
        }

    data = _load_manifest(MANIFEST_PATH)
    metrics = _extract_metrics(data)

    # Fallback: if the canonical manifest has no metrics, look for experiment
    # manifests that belong to the same base model and pick the best accuracy.
    if metrics.get("accuracy") is None and data:
        base_name = data.get("model_name", "xgboost_football")
        best_fallback: dict | None = None
        for candidate in MODELS_DIR.glob(f"model_manifest_{base_name}_exp_*.json"):
            cand_data = _load_manifest(candidate)
            cand_metrics = _extract_metrics(cand_data)
            acc = cand_metrics.get("accuracy")
            if acc is None:
                continue
            if best_fallback is None or acc > best_fallback.get("accuracy", 0.0):
                best_fallback = {
                    "accuracy": acc,
                    "log_loss": cand_metrics.get("log_loss"),
                    "top_feature": cand_metrics.get("top_feature"),
                    "source": cand_metrics.get("source"),
                    "manifest": candidate.name,
                }
        if best_fallback:
            metrics = best_fallback

    return JSONResponse(content={
        "name": data.get("model_name", "xgboost_football") if data else "xgboost_football",
        "accuracy": metrics.get("accuracy"),
        "log_loss": metrics.get("log_loss"),
        "top_feature": metrics.get("top_feature"),
        "source": metrics.get("source"),
        "manifest": metrics.get("manifest"),
        "trained_at": data.get("trained_at") if data else None,
    })


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@app.post("/train/adaptive")
def train_adaptive(body: dict = Body(default={})) -> JSONResponse:
    """Inicia el entrenamiento adaptivo en un proceso separado."""
    global _adaptive_process
    if _adaptive_process is not None and _adaptive_process.poll() is None:
        return JSONResponse(content={"status": "running", "message": "El entrenamiento adaptivo ya está corriendo."})

    cmd = [sys.executable, str(ROOT / "scripts" / "training_orchestrator.py")]
    if body.get("max_auto_batches") is not None:
        cmd.extend(["--max-auto-batches", str(body["max_auto_batches"])])
    if body.get("trials_per_batch") is not None:
        cmd.extend(["--trials-per-batch", str(body["trials_per_batch"])])
    if body.get("max_trials") is not None:
        cmd.extend(["--max-trials", str(body["max_trials"])])
    if body.get("name"):
        cmd.extend(["--name", str(body["name"])])
    if body.get("no_cv"):
        cmd.append("--no-cv")
    if body.get("cv_folds") is not None:
        cmd.extend(["--cv-folds", str(body["cv_folds"])])
    if body.get("cv_embargo") is not None:
        cmd.extend(["--cv-embargo", str(body["cv_embargo"])])
    if body.get("no_gpu"):
        cmd.append("--no-gpu")
    if body.get("promote"):
        cmd.append("--promote")

    _adaptive_process = subprocess.Popen(cmd, cwd=ROOT)
    return JSONResponse(content={"status": "started", "pid": _adaptive_process.pid})


@app.get("/train/status")
def train_status() -> JSONResponse:
    """Devuelve el estado del entrenamiento adaptivo y su último reporte."""
    running = _adaptive_process is not None and _adaptive_process.poll() is None
    report = {}
    if ORCHESTRATOR_REPORT_PATH.exists():
        report = json.loads(ORCHESTRATOR_REPORT_PATH.read_text(encoding="utf-8"))
    return JSONResponse(content={"running": running, "report": report})


@app.get("/gpu")
def gpu_status() -> JSONResponse:
    """Estado de GPU/CUDA disponible para XGBoost."""
    return JSONResponse(content=audit_gpu())


@app.get("/wc_fixtures")
async def wc_fixtures() -> JSONResponse:
    return JSONResponse(content=load_fixtures())


@app.get("/wc_predictions")
async def wc_predictions() -> JSONResponse:
    return JSONResponse(content=load_predictions())


@app.post("/wc_predict")
def wc_predict(body: dict = Body(...)) -> JSONResponse:
    try:
        prediction = predict_single(
            home_team=body.get("home_team", ""),
            away_team=body.get("away_team", ""),
            date=body.get("date", ""),
            model_name=body.get("model", "xgboost_football"),
        )
        return JSONResponse(content=prediction)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)


@app.post("/wc_regenerate")
def wc_regenerate(body: dict = Body(default={})) -> JSONResponse:
    try:
        predictions = regenerate_predictions(body.get("model", "xgboost_football"))
        return JSONResponse(content={"predictions": predictions, "count": len(predictions)})
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)


# Mount static files (JS, CSS) from dashboard directory.
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


def _warm_up_cache() -> None:
    """Pre-compute historical features so first prediction is fast."""
    try:
        start = time.time()
        from scripts.wc2026_engine import warm_up
        warm_up()
        print(f"WC2026 cache ready in {time.time() - start:.1f}s")
    except Exception as exc:
        print(f"WC2026 warm-up failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Training dashboard server for Mondial-Xboost")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()

    print(f"Training dashboard: http://{args.host}:{args.port}")
    print("Presioná Ctrl+C para detener.")
    threading.Thread(target=_warm_up_cache, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
