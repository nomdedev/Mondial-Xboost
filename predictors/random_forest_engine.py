"""
Random Forest predictor engine for football match outcomes.

Mirrors the XGBoostFootballPredictor API so it can be swapped in via the
--engine selector in train/predict/api CLI flows.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from predictors.feature_engineering import FEATURE_COLS
from predictors.model_manifest import build_manifest, hash_dataset, save_manifest
from predictors.xgboost_engine import _fillna

MODELS_DIR = Path(
    os.getenv("MODELS_DIR", str(Path(__file__).parent.parent / "data" / "models"))
)
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # Read-only filesystems (e.g. Vercel) may already contain the model files
    # in the deployment bundle; creation is only required when saving models.
    pass


def _clean_params(model) -> dict[str, Any]:
    """Extract primitive hyperparameters from a scikit-learn model."""
    if model is None or not hasattr(model, "get_params"):
        return {}
    params = model.get_params()
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            cleaned[key] = list(value)
        elif isinstance(value, dict):
            cleaned[key] = {k: v for k, v in value.items() if isinstance(v, (str, int, float, bool, type(None)))}
    return cleaned


class RandomForestFootballPredictor:
    """Trains and predicts football match outcomes with Random Forest."""

    def __init__(
        self,
        random_state: int = 2026,
        n_estimators: int = 500,
        max_depth: int | None = 16,
        min_samples_leaf: int = 5,
        calibrate: bool = True,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.calibrate = calibrate
        self.outcome_model: RandomForestClassifier | CalibratedClassifierCV | None = None
        self.home_goals_model: RandomForestRegressor | None = None
        self.away_goals_model: RandomForestRegressor | None = None
        self.feature_cols = FEATURE_COLS

    def _prepare_x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.feature_cols].copy()
        return _fillna(x)

    def fit(
        self,
        train_df: pd.DataFrame,
        calibrate: bool | None = None,
    ) -> dict[str, Any]:
        """Train models on the supplied feature dataframe."""
        if calibrate is None:
            calibrate = self.calibrate

        train_df = train_df.dropna(subset=["outcome", "home_score", "away_score"]).copy()

        x = self._prepare_x(train_df)
        y_outcome = train_df["outcome"].astype(int)

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        clf.fit(x, y_outcome)

        if calibrate:
            self.outcome_model = CalibratedClassifierCV(clf, method="isotonic", cv=3)
            self.outcome_model.fit(x, y_outcome)
        else:
            self.outcome_model = clf

        self.home_goals_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.away_goals_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.home_goals_model.fit(x, train_df["home_score"])
        self.away_goals_model.fit(x, train_df["away_score"])

        importance = pd.Series(
            self.outcome_model.calibrated_classifiers_[0].estimator.feature_importances_
            if calibrate else clf.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)

        return {
            "n_train": len(x),
            "feature_importance": importance.to_dict(),
            "classes": [0, 1, 2],
        }

    def predict(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Return probability distribution and expected goals for each fixture."""
        if self.outcome_model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")

        x = self._prepare_x(df)
        probs = self.outcome_model.predict_proba(x)
        probs = probs / probs.sum(axis=1, keepdims=True)
        home_goals = self.home_goals_model.predict(x).round(2)
        away_goals = self.away_goals_model.predict(x).round(2)

        records = []
        for idx, row in enumerate(df.itertuples(index=False)):
            records.append({
                "home_team": row.home_team,
                "away_team": row.away_team,
                "date": row.date.isoformat() if hasattr(row.date, "isoformat") else row.date,
                "prob_away_win": float(probs[idx, 0]),
                "prob_draw": float(probs[idx, 1]),
                "prob_home_win": float(probs[idx, 2]),
                "expected_home_goals": float(round(home_goals[idx], 2)),
                "expected_away_goals": float(round(away_goals[idx], 2)),
                "top_pick": ["Away", "Draw", "Home"][int(np.argmax(probs[idx]))],
            })
        return records

    def save(self, name: str = "random_forest_football", metrics: dict[str, Any] | None = None) -> dict[str, Path]:
        """Persist models, metadata, and a manifest entry."""
        paths = {
            "outcome": MODELS_DIR / f"{name}_outcome.pkl",
            "home_goals": MODELS_DIR / f"{name}_home_goals.pkl",
            "away_goals": MODELS_DIR / f"{name}_away_goals.pkl",
            "meta": MODELS_DIR / f"{name}_meta.json",
        }
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(paths["outcome"], "wb") as f:
            pickle.dump(self.outcome_model, f)
        with open(paths["home_goals"], "wb") as f:
            pickle.dump(self.home_goals_model, f)
        with open(paths["away_goals"], "wb") as f:
            pickle.dump(self.away_goals_model, f)
        meta = {
            "feature_cols": self.feature_cols,
            "random_state": self.random_state,
            "name": name,
        }
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        manifest = build_manifest(
            name=name,
            feature_cols=self.feature_cols,
            hyperparameters={
                "outcome": _clean_params(self.outcome_model),
                "home_goals": _clean_params(self.home_goals_model),
                "away_goals": _clean_params(self.away_goals_model),
            },
            dataset_hash=hash_dataset(),
            calibration=True,
            metrics=metrics,
            extra={"random_state": self.random_state},
        )
        named_manifest = MODELS_DIR / f"model_manifest_{name}.json"
        save_manifest(manifest, path=named_manifest)
        return paths

    @classmethod
    def load(cls, name: str = "random_forest_football") -> RandomForestFootballPredictor:
        """Load a persisted predictor."""
        paths = {
            "outcome": MODELS_DIR / f"{name}_outcome.pkl",
            "home_goals": MODELS_DIR / f"{name}_home_goals.pkl",
            "away_goals": MODELS_DIR / f"{name}_away_goals.pkl",
            "meta": MODELS_DIR / f"{name}_meta.json",
        }
        with open(paths["meta"], encoding="utf-8") as f:
            meta = json.load(f)
        predictor = cls(random_state=meta.get("random_state", 2026))
        predictor.feature_cols = meta["feature_cols"]
        with open(paths["outcome"], "rb") as f:
            predictor.outcome_model = pickle.load(f)
        with open(paths["home_goals"], "rb") as f:
            predictor.home_goals_model = pickle.load(f)
        with open(paths["away_goals"], "rb") as f:
            predictor.away_goals_model = pickle.load(f)
        return predictor


def train_and_save(min_date: str = "2010-01-01") -> dict[str, Any]:
    """Convenience CLI entrypoint: build features, train, evaluate, save."""
    from predictors.feature_engineering import build_training_dataset, save_features
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import train_test_split

    print("Building training dataset...")
    full = build_training_dataset(min_date=min_date)
    save_features(full, "train_historical")

    print(f"Splitting {len(full)} rows into train/test...")
    train, test = train_test_split(
        full,
        test_size=0.2,
        random_state=2026,
        stratify=full["outcome"],
    )

    print(f"Training on {len(train)} rows...")
    predictor = RandomForestFootballPredictor(random_state=2026)
    fit_result = predictor.fit(train, calibrate=True)

    x_test = predictor._prepare_x(test)
    probs = predictor.outcome_model.predict_proba(x_test)
    preds = probs.argmax(axis=1)
    feature_importance = fit_result.get("feature_importance", {})
    top_feature = max(feature_importance.items(), key=lambda kv: kv[1])[0] if feature_importance else None

    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": float(accuracy_score(test["outcome"], preds)),
        "log_loss": float(log_loss(test["outcome"], probs)),
        "top_feature": top_feature,
        "feature_importance": feature_importance,
    }

    print("Saving model...")
    paths = predictor.save(metrics=metrics)
    print(f"Saved models to {MODELS_DIR}")

    return {"metrics": metrics, "paths": {k: str(v) for k, v in paths.items()}}


if __name__ == "__main__":
    result = train_and_save()
    print(json.dumps(result, indent=2, default=str))
