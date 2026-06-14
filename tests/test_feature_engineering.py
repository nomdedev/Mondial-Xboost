import pandas as pd
import pytest

from predictors.feature_engineering import (
    build_features,
    build_training_dataset,
    compute_elo_ratings,
    load_historical_results,
)


@pytest.fixture
def sample_historical():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "home_team": ["A", "B", "A", "C"],
        "away_team": ["B", "C", "C", "A"],
        "home_score": [2, 1, 0, 3],
        "away_score": [1, 1, 0, 0],
        "tournament": ["Friendly"] * 4,
        "city": ["X"] * 4,
        "country": ["Y"] * 4,
        "neutral": [False, False, False, False],
    })


def test_load_historical_results():
    df = load_historical_results()
    assert not df.empty
    assert {"date", "home_team", "away_team", "home_score", "away_score"}.issubset(df.columns)


def test_compute_elo_ratings(sample_historical):
    df = compute_elo_ratings(sample_historical)
    assert "home_elo_before" in df.columns
    assert "away_elo_before" in df.columns
    # After A beats B, A's Elo should increase
    assert df.iloc[0]["home_elo_before"] == 1500.0
    assert df.iloc[0]["away_elo_before"] == 1500.0


def test_build_features_no_leakage(sample_historical):
    fixtures = sample_historical.iloc[[-1]].copy()
    features = build_features(sample_historical, fixtures)
    assert len(features) == 1
    # Team C played last match on 2020-04-01; features for C vs A should use data before that
    assert features.iloc[0]["home_team"] == "C"
    assert features.iloc[0]["away_team"] == "A"


def test_build_features_returns_expected_columns(sample_historical):
    fixtures = sample_historical.iloc[[-1]].copy()
    features = build_features(sample_historical, fixtures)
    expected = {
        "elo_diff", "home_points_avg_5", "away_points_avg_5",
        "h2h_wins_diff", "h2h_goals_avg", "h2h_last_result", "neutral",
    }
    assert expected.issubset(features.columns)


def test_compute_elo_respects_neutral():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "home_team": ["A", "A"],
        "away_team": ["B", "B"],
        "home_score": [1, 1],
        "away_score": [1, 1],
        "neutral": [True, False],
    })
    rated = compute_elo_ratings(df)
    # Neutral match first: both teams start at the default rating, so elo_diff is raw (no +100)
    assert rated.iloc[0]["home_elo_before"] == rated.iloc[0]["away_elo_before"]
    assert rated.iloc[0]["elo_diff"] == rated.iloc[0]["home_elo_before"] - rated.iloc[0]["away_elo_before"]
    # Non-neutral match later: elo_diff still equals the raw rating difference
    assert rated.iloc[1]["elo_diff"] == rated.iloc[1]["home_elo_before"] - rated.iloc[1]["away_elo_before"]


def test_build_training_dataset_no_leakage():
    df = build_training_dataset(min_date="2018-01-01")
    assert not df.empty
    assert "outcome" in df.columns
    assert "home_score" in df.columns
    assert "away_score" in df.columns
    # No NaN in targets
    assert df["outcome"].notna().all()
