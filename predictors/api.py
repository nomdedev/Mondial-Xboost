"""
FastAPI bridge for the C# app to call the Python ML predictor.

Endpoints:
  GET  /health
  POST /train
  POST /predict
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from predictors.feature_engineering import build_features, load_historical_results
from predictors.random_forest_engine import RandomForestFootballPredictor
from predictors.xgboost_engine import XGBoostFootballPredictor, train_and_save

app = FastAPI(title="Mondial-Xboost ML Bridge")

# Serve dashboard static files
_DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
app.mount("/static", StaticFiles(directory=str(_DASHBOARD_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(_DASHBOARD_DIR / "index.html"))


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
