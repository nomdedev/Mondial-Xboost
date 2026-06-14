"""
Backtest framework for Mondial-Xboost predictors on previous World Cups.

For each World Cup, trains a model on all historical matches before the
tournament starts, then predicts every match of that tournament.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from predictors.feature_engineering import build_features, load_historical_results
from predictors.xgboost_engine import XGBoostFootballPredictor

BACKTEST_DIR = Path(__file__).parent
RESULTS_DIR = BACKTEST_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    wc_year: int
    n_matches: int
    log_loss: float
    brier_score: float
    top_pick_accuracy: float
    calibration_error: float
    roi_simulated: float

    def to_dict(self):
        return {
            "wc_year": self.wc_year,
            "n_matches": self.n_matches,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "top_pick_accuracy": self.top_pick_accuracy,
            "calibration_error": self.calibration_error,
            "roi_simulated": self.roi_simulated,
        }


def _outcome_to_vector(outcome: int) -> np.ndarray:
    v = np.zeros(3)
    v[outcome] = 1.0
    return v


def _expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Compute ECE for multiclass outcomes."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == outcomes).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if i == 0:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_confidence = confidences[mask].mean()
        avg_accuracy = correct[mask].mean()
        ece += mask.sum() * abs(avg_confidence - avg_accuracy)

    return ece / len(probs)


def backtest_world_cup(
    wc_year: int,
    historical: pd.DataFrame | None = None,
    min_train_date: str = "2010-01-01",
    random_state: int = 2026,
    predictor_kwargs: dict[str, Any] | None = None,
) -> BacktestResult:
    """Backtest XGBoost on a single World Cup."""
    if historical is None:
        historical = load_historical_results()

    # Exclude qualifiers; keep only FIFA World Cup finals matches
    wc_mask = historical["tournament"] == "FIFA World Cup"
    wc_matches = historical[wc_mask & (historical["date"].dt.year == wc_year)].copy()
    if wc_matches.empty:
        raise ValueError(f"No World Cup {wc_year} matches found in historical data")

    wc_matches = wc_matches.sort_values("date").reset_index(drop=True)
    wc_start = wc_matches["date"].min()

    # Train on all history before this World Cup starts. Use the full historical
    # record for look-back features (Elo, H2H) and limit training rows by min_date.
    train_fixtures = historical[
        (historical["date"] < wc_start) & (historical["date"] >= pd.to_datetime(min_train_date))
    ][["date", "home_team", "away_team", "home_score", "away_score", "neutral"]].copy()
    train_features = build_features(
        historical,
        train_fixtures,
        min_date=min_train_date,
    )
    train_features = train_features.dropna(subset=["outcome", "home_score", "away_score"]).copy()

    if len(train_features) < 500:
        raise ValueError(f"Not enough training data before WC {wc_year}: {len(train_features)} rows")

    print(f"  WC {wc_year}: training on {len(train_features)} matches before {wc_start.date()}")
    kwargs = predictor_kwargs or {}
    predictor = XGBoostFootballPredictor(random_state=random_state, **kwargs)
    predictor.fit(train_features)

    # Predict all WC matches
    fixtures = wc_matches[["date", "home_team", "away_team", "neutral"]].copy()
    features = build_features(historical, fixtures)
    predictions = predictor.predict(features)

    probs = np.array([[p["prob_away_win"], p["prob_draw"], p["prob_home_win"]] for p in predictions])
    outcomes = wc_matches.apply(
        lambda r: 2 if r["home_score"] > r["away_score"] else 1 if r["home_score"] == r["away_score"] else 0,
        axis=1,
    ).values
    y_true = np.array([_outcome_to_vector(o) for o in outcomes])

    logloss = log_loss(y_true, probs)
    brier = np.mean([brier_score_loss(y_true[:, i], probs[:, i]) for i in range(3)])
    top_pick_acc = np.mean(np.argmax(probs, axis=1) == outcomes)
    ece = _expected_calibration_error(probs, outcomes)

    # Simulated ROI: bet 1 unit on top pick if confidence > 0.45
    roi = 0.0
    bets = 0
    for i, prob in enumerate(probs):
        pick = np.argmax(prob)
        confidence = prob[pick]
        if confidence > 0.45:
            bets += 1
            if pick == outcomes[i]:
                roi += (1.0 / confidence) - 1.0
            else:
                roi -= 1.0
    roi_pct = roi / bets if bets > 0 else 0.0

    return BacktestResult(
        wc_year=wc_year,
        n_matches=len(probs),
        log_loss=logloss,
        brier_score=brier,
        top_pick_accuracy=top_pick_acc,
        calibration_error=ece,
        roi_simulated=roi_pct,
    )


def run_all_backtests(
    years: list[int] | None = None,
    predictor_kwargs: dict[str, Any] | None = None,
) -> list[BacktestResult]:
    """Run backtests for multiple World Cups."""
    years = years or [2010, 2014, 2018, 2022]
    historical = load_historical_results()
    results = []
    for year in years:
        try:
            print(f"Backtesting World Cup {year}...")
            result = backtest_world_cup(year, historical, predictor_kwargs=predictor_kwargs)
            results.append(result)
            print(f"  log-loss={result.log_loss:.4f}, brier={result.brier_score:.4f}, acc={result.top_pick_accuracy:.2%}, ROI={result.roi_simulated:.2%}")
        except Exception as ex:
            print(f"  Failed: {ex}")

    if results:
        summary = {
            "backtests": [r.to_dict() for r in results],
            "average": {
                "log_loss": float(np.mean([r.log_loss for r in results])),
                "brier_score": float(np.mean([r.brier_score for r in results])),
                "top_pick_accuracy": float(np.mean([r.top_pick_accuracy for r in results])),
                "calibration_error": float(np.mean([r.calibration_error for r in results])),
                "roi_simulated": float(np.mean([r.roi_simulated for r in results])),
            },
        }
        path = RESULTS_DIR / "world_cup_backtest_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {path}")
    return results


if __name__ == "__main__":
    run_all_backtests()
