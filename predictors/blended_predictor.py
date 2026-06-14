"""
Blended predictor: canonical XGBoost + cold-start fallback.

Uses a soft blend between the canonical model and the cold-start model based on
how much recent history each team has. When both teams have plenty of recent
matches the canonical model dominates; when history is sparse the cold-start
model gets more weight.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from predictors.cold_start_model import ColdStartPredictor
from predictors.xgboost_engine import XGBoostFootballPredictor


class BlendedFootballPredictor:
    """Combine canonical and cold-start predictions with a soft blend."""

    def __init__(
        self,
        canonical_name: str = "xgboost_football",
        cold_start_name: str = "cold_start",
        cold_threshold: int = 10,
        blend_steepness: float = 1.0,
    ):
        self.canonical = XGBoostFootballPredictor.load(canonical_name)
        self.cold_start = ColdStartPredictor.load(cold_start_name)
        self.cold_threshold = cold_threshold
        self.blend_steepness = blend_steepness

    def _cold_weight(self, df: pd.DataFrame) -> np.ndarray:
        """Return a weight in [0, 1] for the cold-start model per fixture."""
        home_deficit = np.clip(
            (self.cold_threshold - df["home_recent_matches"].to_numpy()) / self.cold_threshold,
            0,
            1,
        )
        away_deficit = np.clip(
            (self.cold_threshold - df["away_recent_matches"].to_numpy()) / self.cold_threshold,
            0,
            1,
        )
        # If either team is cold, increase cold weight.
        max_deficit = np.maximum(home_deficit, away_deficit)
        # Sigmoid-like smooth transition.
        return 1.0 / (1.0 + np.exp(-self.blend_steepness * (max_deficit - 0.5) * 6))

    def predict(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Return predictions, blending canonical and cold-start models."""
        # Start from canonical predictions (includes expected goals)
        predictions = self.canonical.predict(df)

        # Cold-start outcome probabilities
        cold_probs = self.cold_start.predict_proba(df)
        weights = self._cold_weight(df).reshape(-1, 1)

        for idx, pred in enumerate(predictions):
            canonical_probs = np.array([
                pred["prob_away_win"],
                pred["prob_draw"],
                pred["prob_home_win"],
            ])
            blended = weights[idx] * cold_probs[idx] + (1 - weights[idx]) * canonical_probs
            blended = blended / blended.sum()
            pred["prob_away_win"] = float(blended[0])
            pred["prob_draw"] = float(blended[1])
            pred["prob_home_win"] = float(blended[2])
            pred["top_pick"] = ["Away", "Draw", "Home"][int(np.argmax(blended))]

        return predictions

    @classmethod
    def load(
        cls,
        canonical_name: str = "xgboost_football",
        cold_start_name: str = "cold_start",
    ) -> BlendedFootballPredictor:
        """Load a blended predictor from persisted components."""
        return cls(canonical_name=canonical_name, cold_start_name=cold_start_name)
