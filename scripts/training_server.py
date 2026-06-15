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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent.parent
DASHBOARD_DIR = ROOT / "dashboard"
MODELS_DIR = ROOT / "data" / "models"
STATUS_PATH = MODELS_DIR / "training_status.json"
RESULTS_PATH = MODELS_DIR / "loop_engineering.json"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

app = FastAPI(title="Mondial-Xboost Training Monitor")

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


# Mount static files (JS, CSS) from dashboard directory.
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


def main() -> int:
    parser = argparse.ArgumentParser(description="Training dashboard server for Mondial-Xboost")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()

    print(f"Training dashboard: http://{args.host}:{args.port}")
    print("Presioná Ctrl+C para detener.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
