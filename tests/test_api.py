"""Integration tests for the FastAPI ML bridge."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from predictors.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["engine"] == "xgboost"
    assert "model_loaded" in data


def _train_test_model(client, engine: str, model_name: str):
    import predictors.api as api_module
    from predictors.feature_engineering import build_training_dataset

    if engine == "random_forest":
        from predictors.random_forest_engine import RandomForestFootballPredictor as Predictor
    else:
        from predictors.xgboost_engine import XGBoostFootballPredictor as Predictor

    api_module._predictors = {}

    train = build_training_dataset(min_date="2018-01-01")
    predictor = Predictor(random_state=2026, n_estimators=10, max_depth=2)
    predictor.fit(train, calibrate=False)
    predictor.save(model_name)


def test_predict_endpoint(client, tmp_path, monkeypatch):
    import predictors.api as api_module

    monkeypatch.setenv("MODEL_NAME", "test_api_model")
    api_module._predictors = {}

    _train_test_model(client, "xgboost", "test_api_model")

    fixtures = [
        {"date": "2026-06-15", "home_team": "Argentina", "away_team": "Brazil", "neutral": True},
        {"date": "2026-06-16", "home_team": "Germany", "away_team": "France", "neutral": True},
    ]
    response = client.post("/predict", json={"fixtures": fixtures})
    assert response.status_code == 200, response.text

    data = response.json()
    predictions = data["predictions"]
    assert len(predictions) == len(fixtures)

    for p in predictions:
        total = p["prob_away_win"] + p["prob_draw"] + p["prob_home_win"]
        assert 0.99 <= total <= 1.01
        assert p["top_pick"] in {"Home", "Draw", "Away"}
        assert p["expected_home_goals"] >= 0
        assert p["expected_away_goals"] >= 0


def test_predict_endpoint_random_forest(client, tmp_path, monkeypatch):
    import predictors.api as api_module

    monkeypatch.setenv("MODEL_NAME", "test_api_rf_model")
    api_module._predictors = {}

    _train_test_model(client, "random_forest", "test_api_rf_model")

    fixtures = [
        {"date": "2026-06-15", "home_team": "Argentina", "away_team": "Brazil", "neutral": True},
    ]
    response = client.post("/predict?engine=random_forest", json={"fixtures": fixtures})
    assert response.status_code == 200, response.text

    data = response.json()
    predictions = data["predictions"]
    assert len(predictions) == 1

    p = predictions[0]
    total = p["prob_away_win"] + p["prob_draw"] + p["prob_home_win"]
    assert 0.99 <= total <= 1.01
    assert p["top_pick"] in {"Home", "Draw", "Away"}
