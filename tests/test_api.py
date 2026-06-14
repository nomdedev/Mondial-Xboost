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
    # May be true or false depending on whether a model was trained in this env
    assert "model_loaded" in data


def test_predict_endpoint(client, tmp_path, monkeypatch):
    # If no model is available, train a tiny one on the fly
    # Use a dedicated test model name to avoid overwriting the canonical production model.
    import predictors.api as api_module
    from predictors.feature_engineering import build_training_dataset
    from predictors.xgboost_engine import XGBoostFootballPredictor

    monkeypatch.setenv("XGBOOST_MODEL_NAME", "test_api_model")
    api_module._predictor = None

    train = build_training_dataset(min_date="2018-01-01")
    predictor = XGBoostFootballPredictor(random_state=2026, n_estimators=10, max_depth=2)
    predictor.fit(train, calibrate=False)
    predictor.save("test_api_model")

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
