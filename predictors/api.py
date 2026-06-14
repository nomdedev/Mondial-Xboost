"""
FastAPI bridge for the C# app to call the Python XGBoost predictor.

Endpoints:
  POST /health
  POST /train
  POST /predict
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predictors.feature_engineering import build_features, load_historical_results
from predictors.xgboost_engine import XGBoostFootballPredictor, train_and_save

app = FastAPI(title="Mondial-Xboost ML Bridge")

# Cache the latest trained model in memory
_predictor: XGBoostFootballPredictor | None = None


def _get_predictor() -> XGBoostFootballPredictor:
    global _predictor
    if _predictor is None:
        model_name = os.getenv("XGBOOST_MODEL_NAME", "xgboost_football")
        try:
            _predictor = XGBoostFootballPredictor.load(model_name)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"No trained model available: {exc}") from exc
    return _predictor


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


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
async def health() -> HealthResponse:
    loaded = False
    try:
        _get_predictor()
        loaded = True
    except HTTPException:
        pass
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/train", response_model=TrainResponse)
async def train(min_date: str = "2010-01-01") -> TrainResponse:
    try:
        result = train_and_save(min_date=min_date)
        global _predictor
        _predictor = XGBoostFootballPredictor.load()
        return TrainResponse(status="trained", metrics=result["metrics"], paths=result["paths"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    try:
        historical = load_historical_results(req.historical_path)
        fixtures = pd.DataFrame(req.fixtures)
        features = build_features(historical, fixtures)
        predictor = _get_predictor()
        predictions = predictor.predict(features)
        return PredictResponse(predictions=predictions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
