import pytest

from predictors.feature_engineering import build_training_dataset
from predictors.xgboost_engine import XGBoostFootballPredictor


@pytest.fixture(scope="module")
def trained_predictor():
    train = build_training_dataset(min_date="2018-01-01")
    predictor = XGBoostFootballPredictor(random_state=2026)
    predictor.fit(train, calibrate=False)
    return predictor, train


def test_predict_output_shape(trained_predictor):
    predictor, train = trained_predictor
    preds = predictor.predict(train.head(10))
    assert len(preds) == 10
    for p in preds:
        assert {"prob_home_win", "prob_draw", "prob_away_win"}.issubset(p.keys())
        total = p["prob_home_win"] + p["prob_draw"] + p["prob_away_win"]
        assert 0.99 <= total <= 1.01


def test_top_pick_values(trained_predictor):
    predictor, train = trained_predictor
    preds = predictor.predict(train.head(10))
    for p in preds:
        assert p["top_pick"] in {"Home", "Draw", "Away"}


def test_save_and_load(trained_predictor, tmp_path):
    predictor, _ = trained_predictor
    # Save to a dedicated test model name to avoid overwriting the canonical model.
    predictor.save("test_xgboost_model")
    loaded = XGBoostFootballPredictor.load("test_xgboost_model")
    assert loaded.outcome_model is not None
    assert loaded.home_goals_model is not None
    assert loaded.away_goals_model is not None


def test_feature_cols_match_engineering(trained_predictor):
    predictor, train = trained_predictor
    from predictors.feature_engineering import FEATURE_COLS
    assert predictor.feature_cols == FEATURE_COLS
    prepared = predictor._prepare_x(train.head(1))
    assert list(prepared.columns) == FEATURE_COLS
