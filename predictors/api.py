"""
FastAPI bridge for the C# app and the ML dashboard.

Endpoints:
  GET  /health
  GET  /dashboard/stats
  GET  /dashboard/metrics
  GET  /dashboard/models
  GET  /dashboard/features
  POST /train
  POST /predict
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from predictors.feature_engineering import build_features, load_historical_results
from predictors.model_manifest import MODELS_DIR, load_manifest
from predictors.random_forest_engine import RandomForestFootballPredictor
from predictors.xgboost_engine import XGBoostFootballPredictor, train_and_save
from scripts.wc2026_engine import (
    load_fixtures,
    load_predictions,
    predict_single,
    regenerate_predictions,
)


app = FastAPI(title="Mondial-Xboost ML Bridge")

# Serve dashboard static files
_DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
app.mount("/static", StaticFiles(directory=str(_DASHBOARD_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(_DASHBOARD_DIR / "index.html"))


@app.get("/debug/files")
async def debug_files() -> dict[str, Any]:
    """Temporary helper to inspect the deployed filesystem."""
    cwd = Path.cwd()
    task = Path("/var/task")
    models_rel = Path(os.getenv("MODELS_DIR", "predictors/models"))
    models_abs = (cwd / models_rel).resolve()
    result: dict[str, Any] = {
        "cwd": str(cwd),
        "models_env": str(models_rel),
        "models_abs": str(models_abs),
        "models_exists": models_abs.exists(),
        "var_task_exists": task.exists(),
    }
    if task.exists():
        try:
            result["var_task_listing"] = [str(p) for p in list(task.rglob("*")) if p.is_file()][:200]
        except Exception as exc:
            result["var_task_listing_error"] = str(exc)
    if models_abs.exists():
        result["models_files"] = [str(p.relative_to(models_abs)) for p in models_abs.iterdir()]
    return result


# Cache the latest trained model in memory
_predictors: dict[str, XGBoostFootballPredictor | RandomForestFootballPredictor] = {}


def _get_predictor(engine: str = "xgboost") -> XGBoostFootballPredictor | RandomForestFootballPredictor:
    global _predictors
    if engine in _predictors:
        return _predictors[engine]

    model_name = os.getenv("MODEL_NAME", f"{engine}_football")
    try:
        if engine == "random_forest":
            predictor = RandomForestFootballPredictor.load(model_name)
        else:
            predictor = XGBoostFootballPredictor.load(model_name)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"No trained model available for engine={engine}: {exc}") from exc

    _predictors[engine] = predictor
    return predictor


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    engine: str


class TrainResponse(BaseModel):
    status: str
    metrics: dict[str, Any]
    paths: dict[str, str]


class PredictRequest(BaseModel):
    historical_path: str | None = None
    fixtures: list[dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: list[dict[str, Any]]


class WCPredictRequest(BaseModel):
    home_team: str
    away_team: str
    date: str | None = None


class WCRegenerateRequest(BaseModel):
    model: str = "xgboost_football"


@app.get("/health", response_model=HealthResponse)
async def health(engine: str = "xgboost") -> HealthResponse:
    loaded = False
    try:
        _get_predictor(engine)
        loaded = True
    except HTTPException:
        pass
    return HealthResponse(status="ok", model_loaded=loaded, engine=engine)


@app.post("/train", response_model=TrainResponse)
async def train(min_date: str = "2010-01-01", engine: str = "xgboost") -> TrainResponse:
    try:
        if engine == "random_forest":
            from predictors.random_forest_engine import train_and_save as rf_train
            result = rf_train(min_date=min_date)
            _predictors["random_forest"] = RandomForestFootballPredictor.load()
        else:
            result = train_and_save(min_date=min_date)
            _predictors["xgboost"] = XGBoostFootballPredictor.load()
        return TrainResponse(status="trained", metrics=result["metrics"], paths=result["paths"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, engine: str = "xgboost") -> PredictResponse:
    try:
        historical = load_historical_results(req.historical_path)
        fixtures = pd.DataFrame(req.fixtures)
        features = build_features(historical, fixtures)
        predictor = _get_predictor(engine)
        predictions = predictor.predict(features)
        return PredictResponse(predictions=predictions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------


def _dataset_path() -> Path:
    """Return the canonical historical results CSV path."""
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "MondialXboost.Web" / "Data" / "historical_results.csv",
        root / "data" / "raw" / "historical_results.csv",
        root / "data" / "historical_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("historical_results.csv not found")


@app.get("/dashboard/stats")
async def dashboard_stats() -> dict[str, Any]:
    """Return high-level dataset statistics for the dashboard."""
    try:
        df = load_historical_results()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load dataset: {exc}") from exc

    total_matches = len(df)
    teams = sorted(set(df["home_team"].dropna().unique()) | set(df["away_team"].dropna().unique()))
    date_range = {
        "min": df["date"].min().isoformat() if not df["date"].isna().all() else None,
        "max": df["date"].max().isoformat() if not df["date"].isna().all() else None,
    }

    # Outcome distribution from historical scores.
    def _outcome(row: pd.Series) -> str:
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            return "Unknown"
        if row["home_score"] > row["away_score"]:
            return "Home"
        if row["home_score"] < row["away_score"]:
            return "Away"
        return "Draw"

    outcomes = df.apply(_outcome, axis=1).value_counts().to_dict()

    return {
        "total_matches": total_matches,
        "teams": len(teams),
        "team_list": teams,
        "date_range": date_range,
        "outcome_distribution": {
            "home": int(outcomes.get("Home", 0)),
            "draw": int(outcomes.get("Draw", 0)),
            "away": int(outcomes.get("Away", 0)),
        },
    }


@app.get("/dashboard/metrics")
async def dashboard_metrics() -> dict[str, Any]:
    """Return canonical model metrics from the manifest."""
    manifest = load_manifest()
    if not manifest:
        raise HTTPException(status_code=503, detail="No model manifest available")

    metrics = manifest.get("metrics", {})
    return {
        "model_name": manifest.get("model_name"),
        "trained_at": manifest.get("trained_at"),
        "accuracy": metrics.get("accuracy"),
        "log_loss": metrics.get("log_loss"),
        "top_feature": metrics.get("top_feature"),
        "feature_count": len(manifest.get("feature_cols", [])),
    }


@app.get("/dashboard/models")
async def dashboard_models() -> dict[str, Any]:
    """List trained model artifacts in data/models."""
    try:
        models = []
        for path in sorted(MODELS_DIR.glob("*_outcome.pkl")):
            meta_path = path.with_name(path.name.replace("_outcome.pkl", "_meta.json"))
            name = path.name.replace("_outcome.pkl", "")
            created = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    created = meta.get("trained_at", created)
                except Exception:
                    pass
            size_mb = round(path.stat().st_size / (1024 * 1024), 2)
            models.append({
                "name": name,
                "size_mb": size_mb,
                "created": created,
                "feature_cols": meta.get("feature_cols", []),
            })
        return {"models": models}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/dashboard/features")
async def dashboard_features() -> dict[str, Any]:
    """Return feature importance for the canonical XGBoost model."""
    try:
        predictor = _get_predictor("xgboost")
    except HTTPException as exc:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc.detail}") from exc

    try:
        # CalibratedClassifierCV wraps the raw XGBClassifier.
        outcome_model = predictor.outcome_model
        if hasattr(outcome_model, "calibrated_classifiers_"):
            estimator = outcome_model.calibrated_classifiers_[0].estimator
        else:
            estimator = outcome_model

        importances = estimator.feature_importances_
        features = predictor.feature_cols
        feature_importance = [
            {"feature": f, "importance": round(float(imp), 6)}
            for f, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        ]
        return {"features": feature_importance}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not compute feature importance: {exc}") from exc


# ---------------------------------------------------------------------------
# World Cup 2026 endpoints (also consumed by the dashboard)
# ---------------------------------------------------------------------------


@app.get("/wc_fixtures")
async def wc_fixtures() -> list[dict[str, Any]]:
    """Return the 2026 World Cup group-stage fixtures."""
    try:
        return load_fixtures()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load fixtures: {exc}") from exc


@app.get("/wc_predictions")
async def wc_predictions() -> list[dict[str, Any]]:
    """Return saved predictions for the 2026 World Cup group stage."""
    try:
        return load_predictions()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load predictions: {exc}") from exc


@app.post("/wc_predict")
async def wc_predict(req: WCPredictRequest) -> dict[str, Any]:
    """Predict a single WC 2026 match using the canonical model."""
    try:
        date = req.date or "2026-07-15"
        return predict_single(req.home_team, req.away_team, date)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/wc_regenerate")
async def wc_regenerate(req: WCRegenerateRequest | None = None) -> dict[str, Any]:
    """Regenerate predictions for the whole group stage."""
    try:
        model = req.model if req else "xgboost_football"
        predictions = regenerate_predictions(model)
        return {"predictions": predictions, "count": len(predictions)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/wc_tournament")
async def wc_tournament() -> dict[str, Any]:
    """Return the pre-computed full 2026 World Cup simulation."""
    try:
        path = Path(__file__).resolve().parent.parent / "data" / "wc2026_tournament.json"
        if not path.exists():
            raise FileNotFoundError("Tournament snapshot not found")
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load tournament snapshot: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
