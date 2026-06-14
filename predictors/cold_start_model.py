"""
Cold-start predictor for matches where at least one team lacks recent history.

The canonical XGBoostFootballPredictor relies on rolling team stats that are
unreliable when a team has played few recent matches. This model is trained on
static/generalizable features only (Elo, H2H, tournament, experience) and is
used as a fallback for low-confidence fixtures.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

COLD_START_FEATURE_COLS = [
    "elo_diff",
    "elo_diff_recent",
    "neutral",
    "h2h_last_result",
    "h2h_years_since",
    "home_matches_played",
    "away_matches_played",
    "home_recent_matches",
    "away_recent_matches",
    "recent_matches_diff",
    "tournament_p_home",
    "tournament_p_draw",
    "tournament_p_away",
]


class ColdStartPredictor:
    """Predict outcomes for fixtures with sparse recent team history."""

    def __init__(
        self,
        cold_threshold: int = 10,
        random_state: int = 2026,
    ):
        self.cold_threshold = cold_threshold
        self.random_state = random_state
        self.model: xgb.XGBClassifier | None = None
        self.tournament_encoding: dict[str, float] | None = None

    def _encode_tournament(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Target-encode tournament using the full outcome distribution."""
        df = df.copy()
        if "tournament" not in df.columns:
            df["tournament"] = "Unknown"
        if fit:
            counts = df["tournament"].value_counts()
            global_probs = df["outcome"].value_counts(normalize=True).to_dict()
            grouped = df.groupby("tournament")["outcome"]
            # Smoothed probabilities for each outcome class.
            smoothed = {}
            for outcome in (0, 1, 2):
                p = grouped.apply(lambda s: (s == outcome).mean())
                global_p = global_probs.get(outcome, 1 / 3)
                smoothed[outcome] = (p * counts + global_p * 10) / (counts + 10)
            self.tournament_encoding = {
                t: {
                    0: float(smoothed[0].get(t, global_probs.get(0, 1 / 3))),
                    1: float(smoothed[1].get(t, global_probs.get(1, 1 / 3))),
                    2: float(smoothed[2].get(t, global_probs.get(2, 1 / 3))),
                }
                for t in counts.index
            }
        encoding = self.tournament_encoding or {}
        defaults = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
        df["tournament_p_away"] = df["tournament"].map(
            lambda t: encoding.get(t, defaults)[0]
        )
        df["tournament_p_draw"] = df["tournament"].map(
            lambda t: encoding.get(t, defaults)[1]
        )
        df["tournament_p_home"] = df["tournament"].map(
            lambda t: encoding.get(t, defaults)[2]
        )
        return df

    def _prepare_x(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the cold-start feature matrix."""
        df = self._encode_tournament(df, fit=False)
        df["recent_matches_diff"] = df["home_recent_matches"] - df["away_recent_matches"]
        return df[COLD_START_FEATURE_COLS].fillna(0.0)

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train the cold-start model on a balanced mix of cold and warm fixtures."""
        df = df.copy()
        df["home_cold"] = df["home_recent_matches"] < self.cold_threshold
        df["away_cold"] = df["away_recent_matches"] < self.cold_threshold
        df["is_cold"] = df["home_cold"] | df["away_cold"]

        cold_df = df[df["is_cold"]].copy()
        warm_df = df[~df["is_cold"]].copy()

        n_cold = len(cold_df)
        if n_cold == 0:
            raise ValueError("No cold-start fixtures found in training data")

        # Balance warm examples to avoid the model learning only from cold cases.
        if len(warm_df) > n_cold:
            warm_df = warm_df.sample(n=n_cold, random_state=self.random_state)

        train_df = pd.concat([cold_df, warm_df], ignore_index=True)
        train_df = self._encode_tournament(train_df, fit=True)
        train_df["recent_matches_diff"] = (
            train_df["home_recent_matches"] - train_df["away_recent_matches"]
        )

        x = train_df[COLD_START_FEATURE_COLS].fillna(0.0)
        y = train_df["outcome"].astype(int)

        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        self.model.fit(x, y, verbose=False)

        importances = dict(zip(COLD_START_FEATURE_COLS, self.model.feature_importances_.tolist()))
        top_feature = max(importances, key=importances.get)

        return {
            "n_cold": int(n_cold),
            "n_warm": int(len(warm_df)),
            "n_train": len(train_df),
            "top_feature": top_feature,
            "top_feature_importance": float(importances[top_feature]),
        }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return (n, 3) probability array [away, draw, home]."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        x = self._prepare_x(df)
        return self.model.predict_proba(x)

    def is_cold_start(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean series marking cold-start fixtures."""
        return (df["home_recent_matches"] < self.cold_threshold) | (
            df["away_recent_matches"] < self.cold_threshold
        )

    def save(self, name: str = "cold_start") -> Path:
        """Persist the cold-start model."""
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, name: str = "cold_start") -> ColdStartPredictor:
        """Load a persisted cold-start model."""
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
