"""Tests for the Random Forest football predictor engine."""

from __future__ import annotations

import pytest

from predictors.feature_engineering import build_training_dataset
from predictors.random_forest_engine import RandomForestFootballPredictor


@pytest.fixture(scope="module")
def tiny_training_dataset():
    return build_training_dataset(min_date="2018-01-01")


def test_rf_predictor_fits_and_predicts(tiny_training_dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    from predictors import random_forest_engine
    random_forest_engine.MODELS_DIR = tmp_path

    predictor = RandomForestFootballPredictor(random_state=2026, n_estimators=10, max_depth=3)
    metrics = predictor.fit(tiny_training_dataset, calibrate=False)

    assert metrics["n_train"] > 0
    assert "feature_importance" in metrics

    sample = tiny_training_dataset.head(3).copy()
    predictions = predictor.predict(sample)

    assert len(predictions) == 3
    for p in predictions:
        total = p["prob_away_win"] + p["prob_draw"] + p["prob_home_win"]
        assert 0.99 <= total <= 1.01
        assert p["top_pick"] in {"Home", "Draw", "Away"}
        assert p["expected_home_goals"] >= 0
        assert p["expected_away_goals"] >= 0


def test_rf_save_and_load(tiny_training_dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    from predictors import random_forest_engine
    random_forest_engine.MODELS_DIR = tmp_path

    predictor = RandomForestFootballPredictor(random_state=2026, n_estimators=10, max_depth=3)
    predictor.fit(tiny_training_dataset, calibrate=False)
    predictor.save("test_rf")

    loaded = RandomForestFootballPredictor.load("test_rf")
    sample = tiny_training_dataset.head(2).copy()
    original = predictor.predict(sample)
    restored = loaded.predict(sample)

    assert len(original) == len(restored)
    for o, r in zip(original, restored):
        assert o["top_pick"] == r["top_pick"]
