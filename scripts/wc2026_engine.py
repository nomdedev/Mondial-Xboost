"""WC 2026 prediction engine used by the training dashboard.

This module exposes a clean API to load fixtures/predictions, predict single
matches or the whole group stage, and (in later phases) compute group standings
and knockout brackets.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from predictors.feature_engineering import (  # noqa: E402
    FEATURE_COLS,
    _build_team_history,
    _compute_h2h,
    _compute_team_rolling,
    compute_elo_ratings,
    load_historical_results,
)
from predictors.random_forest_engine import RandomForestFootballPredictor  # noqa: E402
from predictors.xgboost_engine import XGBoostFootballPredictor  # noqa: E402

FIXTURES_PATH = ROOT / "data" / "wc2026_fixtures.json"
PREDICTIONS_PATH = ROOT / "data" / "wc2026_predictions.json"
RESULTS_PATH = ROOT / "data" / "wc2026_results.json"
FEATURES_PATH = ROOT / "data" / "wc2026_features.parquet"


def load_fixtures() -> list[dict]:
    """Load the 2026 World Cup group-stage fixtures."""
    if not FIXTURES_PATH.exists():
        return []
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def load_predictions() -> list[dict]:
    """Load saved predictions for the group stage."""
    if not PREDICTIONS_PATH.exists():
        return []
    return json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))


def _serialize_predictions(predictions: list[dict]) -> list[dict]:
    """Return a JSON-serializable copy of predictions."""
    serializable: list[dict] = []
    for p in predictions:
        p_copy = dict(p)
        date = p_copy.get("date")
        if isinstance(date, str):
            p_copy["date"] = date[:10]
        else:
            p_copy["date"] = str(date)[:10]
        serializable.append(p_copy)
    return serializable


PREPARED_DIR = ROOT / "data" / "prepared"


def _load_prepared_historical() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load pre-computed historical features from data/prepared/ if available."""
    historical_path = PREPARED_DIR / "historical.csv.gz"
    long_path = PREPARED_DIR / "long.csv.gz"
    h2h_path = PREPARED_DIR / "h2h.csv.gz"
    if not all(p.exists() for p in (historical_path, long_path, h2h_path)):
        return None

    historical = pd.read_csv(historical_path, parse_dates=["date"], compression="gzip")
    long = pd.read_csv(long_path, parse_dates=["date"], compression="gzip")
    historical_h2h = pd.read_csv(h2h_path, parse_dates=["date"], compression="gzip")
    return historical, long, historical_h2h


@lru_cache(maxsize=1)
def _prepared_historical() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare historical data once and cache it.

    In production (Vercel) we load pre-computed CSVs to avoid the ~70s cost of
    recomputing rolling stats and H2H on every cold start. In development we
    fall back to computing them on demand.

    Returns (elo_historical, rolling_long, h2h_historical).
    """
    prepared = _load_prepared_historical()
    if prepared is not None:
        return prepared

    historical = load_historical_results()
    historical["date"] = pd.to_datetime(historical["date"])
    historical = compute_elo_ratings(historical)
    long = _build_team_history(historical)
    long = _compute_team_rolling(long)
    historical_h2h = _compute_h2h(historical)
    return historical, long, historical_h2h


def _last_elo(before: pd.DataFrame, team: str, col: str) -> float:
    home_elos = before[before["home_team"] == team][["date", col]].rename(columns={col: "elo"})
    away_elos = before[before["away_team"] == team][["date", col]].rename(columns={col: "elo"})
    all_elos = pd.concat([home_elos, away_elos]).sort_values("date")
    return float(all_elos.iloc[-1]["elo"]) if not all_elos.empty else 1500.0


def _latest_team_stats(long: pd.DataFrame, team: str, date: pd.Timestamp) -> dict:
    team_long = long[(long["team"] == team) & (long["date"] < date)].sort_values("date")
    if team_long.empty:
        return {
            "points_avg_5": 0.5,
            "points_avg_10": 0.5,
            "goals_scored_avg_10": 1.3,
            "goals_conceded_avg_10": 1.3,
            "win_rate_10": 0.0,
            "draw_rate_10": 0.0,
            "loss_rate_10": 0.0,
            "matches_played": 0,
        }
    last = team_long.iloc[-1]
    return {
        "points_avg_5": float(last.get("points_avg_5", 0.5)),
        "points_avg_10": float(last.get("points_avg_10", 0.5)),
        "goals_scored_avg_10": float(last.get("goals_scored_avg_10", 1.3)),
        "goals_conceded_avg_10": float(last.get("goals_conceded_avg_10", 1.3)),
        "win_rate_10": float(last.get("win_rate_10", 0.0)),
        "draw_rate_10": float(last.get("draw_rate_10", 0.0)),
        "loss_rate_10": float(last.get("loss_rate_10", 0.0)),
        "matches_played": int(last.get("matches_played", 0)),
    }


def build_features_for_fixtures(fixtures: list[dict]) -> pd.DataFrame:
    """Build feature matrix for a list of fixtures using cached historical data."""
    if not fixtures:
        return pd.DataFrame(columns=FEATURE_COLS + ["home_team", "away_team", "date"])

    historical, long, historical_h2h = _prepared_historical()
    fixtures_df = pd.DataFrame(fixtures).copy()
    fixtures_df["date"] = pd.to_datetime(fixtures_df["date"])

    elo_rows = []
    stat_rows = []
    h2h_rows = []

    for _, fx in fixtures_df.iterrows():
        date = fx["date"]
        home, away = fx["home_team"], fx["away_team"]
        before = historical[historical["date"] < date]

        elo_rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_elo_before": _last_elo(before, home, "home_elo_before"),
            "away_elo_before": _last_elo(before, away, "away_elo_before"),
            "home_elo_recent": _last_elo(before, home, "home_elo_recent"),
            "away_elo_recent": _last_elo(before, away, "away_elo_recent"),
        })

        home_stats = _latest_team_stats(long, home, date)
        away_stats = _latest_team_stats(long, away, date)
        stat_rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            **{f"home_{k}": v for k, v in home_stats.items()},
            **{f"away_{k}": v for k, v in away_stats.items()},
        })

        before_h2h = historical_h2h[historical_h2h["date"] < date]
        mask = before_h2h.apply(
            lambda r: sorted([r["home_team"], r["away_team"]]) == sorted([home, away]),
            axis=1,
        )
        pair = before_h2h[mask]
        if not pair.empty:
            last = pair.iloc[-1]
            h2h_last_result = float(last["h2h_last_result"])
            h2h_goals_avg = float(last["h2h_goals_avg"])
            h2h_wins_diff = float(last["h2h_wins_diff"])
            h2h_years_since = (date - last["date"]).days / 365.25
        else:
            h2h_last_result, h2h_goals_avg, h2h_wins_diff, h2h_years_since = 0.0, 1.3, 0.0, 20.0

        h2h_rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "h2h_last_result": h2h_last_result,
            "h2h_goals_avg": h2h_goals_avg,
            "h2h_wins_diff": h2h_wins_diff,
            "h2h_years_since": h2h_years_since,
        })

    elo_df = pd.DataFrame(elo_rows)
    stats_df = pd.DataFrame(stat_rows)
    h2h_df = pd.DataFrame(h2h_rows)

    features = (
        fixtures_df[["date", "home_team", "away_team", "neutral"]]
        .merge(elo_df, on=["date", "home_team", "away_team"], how="left")
        .merge(stats_df, on=["date", "home_team", "away_team"], how="left")
        .merge(h2h_df, on=["date", "home_team", "away_team"], how="left")
    )

    features["elo_diff"] = features["home_elo_before"] - features["away_elo_before"]
    features["elo_diff_recent"] = features["home_elo_recent"] - features["away_elo_recent"]

    # Fill defaults matching XGBoostFootballPredictor expectations.
    features["elo_diff"] = features["elo_diff"].fillna(0.0)
    features["elo_diff_recent"] = features["elo_diff_recent"].fillna(0.0)
    for col in ["home_points_avg_5", "home_points_avg_10", "away_points_avg_5", "away_points_avg_10"]:
        features[col] = features[col].fillna(0.5)
    for col in [
        "home_win_rate_10", "home_draw_rate_10", "home_loss_rate_10",
        "away_win_rate_10", "away_draw_rate_10", "away_loss_rate_10", "h2h_last_result",
    ]:
        features[col] = features[col].fillna(0.0)
    for col in [
        "home_goals_scored_avg_10", "home_goals_conceded_avg_10",
        "away_goals_scored_avg_10", "away_goals_conceded_avg_10", "h2h_goals_avg",
    ]:
        features[col] = features[col].fillna(1.3)
    for col in ["home_matches_played", "away_matches_played"]:
        features[col] = features[col].fillna(0).astype(int)
    features["h2h_wins_diff"] = features["h2h_wins_diff"].fillna(0.0)
    features["h2h_years_since"] = features["h2h_years_since"].fillna(20.0)

    # The predictor's FEATURE_COLS may include newer features not yet
    # computed by the historical dataset pipeline. Add sensible defaults
    # so the model receives a complete feature vector.
    feature_defaults = {
        "home_momentum_3": 0.0,
        "away_momentum_3": 0.0,
        "home_sos_5": 1500.0,
        "away_sos_5": 1500.0,
        "home_points_weighted_10": 0.5,
        "away_points_weighted_10": 0.5,
        "tournament_importance": 1.0,
    }
    for col, default in feature_defaults.items():
        if col not in features.columns:
            features[col] = default

    return features[FEATURE_COLS + ["home_team", "away_team", "date"]]


# In-memory cache for the loaded predictor to avoid repeated pickle reads.
_predictor_cache: dict[str, Any] = {}


def _load_predictor(model_name: str = "xgboost_football"):
    """Load the requested predictor engine (cached)."""
    if model_name not in _predictor_cache:
        if model_name.startswith("random_forest"):
            _predictor_cache[model_name] = RandomForestFootballPredictor.load(model_name)
        else:
            _predictor_cache[model_name] = XGBoostFootballPredictor.load(model_name)
    return _predictor_cache[model_name]


def predict_fixture_list(fixtures: list[dict], model_name: str = "xgboost_football") -> list[dict]:
    """Predict a list of fixtures using the chosen model."""
    if not fixtures:
        return []

    features = build_features_for_fixtures(fixtures)
    predictor = _load_predictor(model_name)
    predictions = predictor.predict(features)

    fixtures_df = pd.DataFrame(fixtures).copy()
    meta_cols = ["date", "home_team", "away_team", "group", "neutral"]
    available = [c for c in meta_cols if c in fixtures_df.columns]
    if available:
        fixtures_df["date"] = pd.to_datetime(fixtures_df["date"]).dt.strftime("%Y-%m-%d")
        meta = fixtures_df[available].to_dict(orient="records")
        for pred, meta_row in zip(predictions, meta, strict=True):
            for key, value in meta_row.items():
                if key not in pred:
                    pred[key] = value

    return _serialize_predictions(predictions)


def predict_single(
    home_team: str,
    away_team: str,
    date: str,
    model_name: str = "xgboost_football",
) -> dict:
    """Predict a single match."""
    fixtures = [
        {"date": date, "home_team": home_team, "away_team": away_team, "neutral": True}
    ]
    predictions = predict_fixture_list(fixtures, model_name)
    return predictions[0]


def _save_features(features: pd.DataFrame) -> None:
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)


def _load_cached_features(fixtures: list[dict]) -> pd.DataFrame | None:
    """Return cached features if they match the current fixtures."""
    if not FEATURES_PATH.exists():
        return None
    try:
        cached = pd.read_parquet(FEATURES_PATH)
        fixtures_df = pd.DataFrame(fixtures)
        if len(cached) != len(fixtures_df):
            return None
        # Quick check: same home/away/date tuples in same order.
        for col in ["home_team", "away_team", "date"]:
            if col not in cached.columns or col not in fixtures_df.columns:
                return None
            fixtures_df[col] = fixtures_df[col].astype(str)
            cached[col] = cached[col].astype(str)
            if not fixtures_df[col].reset_index(drop=True).equals(cached[col].reset_index(drop=True)):
                return None
        return cached[FEATURE_COLS + ["home_team", "away_team", "date"]]
    except Exception:
        return None


def regenerate_predictions(model_name: str = "xgboost_football") -> list[dict]:
    """Regenerate and persist predictions for the whole group stage."""
    fixtures = load_fixtures()
    features = _load_cached_features(fixtures)
    if features is None:
        features = build_features_for_fixtures(fixtures)
        _save_features(features)

    predictor = _load_predictor(model_name)
    predictions = predictor.predict(features)

    fixtures_df = pd.DataFrame(fixtures).copy()
    meta_cols = ["date", "home_team", "away_team", "group", "neutral"]
    available = [c for c in meta_cols if c in fixtures_df.columns]
    if available:
        fixtures_df["date"] = pd.to_datetime(fixtures_df["date"]).dt.strftime("%Y-%m-%d")
        meta = fixtures_df[available].to_dict(orient="records")
        for pred, meta_row in zip(predictions, meta, strict=True):
            for key, value in meta_row.items():
                if key not in pred:
                    pred[key] = value

    predictions = _serialize_predictions(predictions)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return predictions


def warm_up() -> None:
    """Pre-compute cached historical data so first prediction is fast."""
    _prepared_historical()
