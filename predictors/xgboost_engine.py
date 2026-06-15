"""
XGBoost predictor engine for football match outcomes.

Trains multiclass (1X2) and optional regression models (goals) on features built
by feature_engineering.py. Persists models to disk for the FastAPI bridge.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from predictors.feature_engineering import FEATURE_COLS
from predictors.model_manifest import build_manifest, hash_dataset, save_manifest

MODELS_DIR = Path(
    os.getenv("MODELS_DIR", str(Path(__file__).parent.parent / "data" / "models"))
)
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # Read-only filesystems (e.g. Vercel) may already contain the model files
    # in the deployment bundle; creation is only required when saving models.
    pass


def _xgboost_device() -> str:
    """Return the XGBoost device from the environment (cuda or cpu)."""
    return "cuda" if os.getenv("XGBOOST_DEVICE", "cpu").lower() == "cuda" else "cpu"


def _xgboost_can_run_cuda() -> bool:
    """Intenta entrenar un booster en CUDA y confirma que no haya fallback a CPU."""
    try:
        info = xgb.build_info()
        if not info.get("USE_CUDA", False):
            return False
        X = np.random.RandomState(42).rand(100, 4)
        y = np.random.RandomState(43).randint(0, 3, size=100)
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "device": "cuda",
            "tree_method": "hist",
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 3,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            xgb.train(params, dtrain, num_boost_round=5)
            for warning in w:
                msg = str(warning.message).lower()
                if "no visible gpu" in msg or "changed from gpu to cpu" in msg:
                    return False
        return True
    except Exception:
        return False


def detect_xgboost_device(prefer_gpu: bool = True) -> str:
    """Auto-detectar GPU CUDA; respetar XGBOOST_DEVICE si está seteada."""
    env = os.getenv("XGBOOST_DEVICE", "auto").lower()
    if env == "cpu":
        return "cpu"
    if env == "cuda":
        return "cuda"
    if not prefer_gpu:
        return "cpu"
    return "cuda" if _xgboost_can_run_cuda() else "cpu"


def default_n_jobs(device: str) -> int:
    """n_jobs recomendado: 1 en GPU para evitar contención, -1 en CPU."""
    return 1 if device == "cuda" else -1


def _fillna(x: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with sensible defaults."""
    x = x.copy()
    # Elo diff default 0 (even teams)
    x["elo_diff"] = x["elo_diff"].fillna(0.0)
    # Form / goals defaults to neutral / average
    for col in [
        "home_points_avg_5", "home_points_avg_10",
        "away_points_avg_5", "away_points_avg_10",
        "home_win_rate_10", "home_draw_rate_10", "home_loss_rate_10",
        "away_win_rate_10", "away_draw_rate_10", "away_loss_rate_10",
        "h2h_last_result",
    ]:
        x[col] = x[col].fillna(0.5 if "rate" in col or "points" in col else 0.0)
    for col in [
        "home_goals_scored_avg_10", "home_goals_conceded_avg_10",
        "away_goals_scored_avg_10", "away_goals_conceded_avg_10",
        "h2h_goals_avg",
    ]:
        x[col] = x[col].fillna(1.3)  # global average-ish
    x["home_matches_played"] = x["home_matches_played"].fillna(0).astype(int)
    x["away_matches_played"] = x["away_matches_played"].fillna(0).astype(int)
    x["h2h_wins_diff"] = x["h2h_wins_diff"].fillna(0.0)
    x["h2h_years_since"] = x["h2h_years_since"].fillna(20.0)
    x["neutral"] = x["neutral"].fillna(False).astype(bool)
    # New features
    for col in ["home_momentum_3", "away_momentum_3"]:
        if col in x.columns:
            x[col] = x[col].fillna(0.0)
    for col in ["home_sos_5", "away_sos_5"]:
        if col in x.columns:
            x[col] = x[col].fillna(1500.0)
    for col in ["home_points_weighted_10", "away_points_weighted_10"]:
        if col in x.columns:
            x[col] = x[col].fillna(0.5)
    for col in ["tournament_importance"]:
        if col in x.columns:
            x[col] = x[col].fillna(1.0)
    return x


def _clean_params(model) -> dict[str, Any]:
    """Extract primitive hyperparameters from a scikit-learn / xgboost model."""
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


class XGBoostFootballPredictor:
    """Trains and predicts football match outcomes."""

    def __init__(
        self,
        random_state: int = 2026,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.1,
        calibrate: bool = True,
        device: str | None = None,
        tree_method: str = "hist",
        max_bin: int = 256,
        n_jobs: int | None = None,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.calibrate = calibrate
        self.device = device if device is not None else detect_xgboost_device()
        self.tree_method = tree_method
        self.max_bin = max_bin
        self.n_jobs = n_jobs if n_jobs is not None else default_n_jobs(self.device)
        self.outcome_model: xgb.XGBClassifier | None = None
        self.home_goals_model: xgb.XGBRegressor | None = None
        self.away_goals_model: xgb.XGBRegressor | None = None
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

        # Drop rows with missing targets
        train_df = train_df.dropna(subset=["outcome", "home_score", "away_score"]).copy()

        x = self._prepare_x(train_df)
        y_outcome = train_df["outcome"].astype(int)

        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state,
            eval_metric="mlogloss",
            n_jobs=self.n_jobs,
            device=self.device,
            tree_method=self.tree_method,
            max_bin=self.max_bin,
        )

        clf.fit(x, y_outcome, verbose=False)

        if calibrate:
            self.outcome_model = CalibratedClassifierCV(clf, method="isotonic", cv=3)
            self.outcome_model.fit(x, y_outcome)
        else:
            self.outcome_model = clf

        # Goal regression models
        self.home_goals_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=self.random_state,
            objective="reg:squarederror",
            n_jobs=self.n_jobs,
            device=self.device,
            tree_method=self.tree_method,
            max_bin=self.max_bin,
        )
        self.away_goals_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=self.random_state,
            objective="reg:squarederror",
            n_jobs=self.n_jobs,
            device=self.device,
            tree_method=self.tree_method,
            max_bin=self.max_bin,
        )
        self.home_goals_model.fit(x, train_df["home_score"])
        self.away_goals_model.fit(x, train_df["away_score"])

        # Feature importance summary
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
        # Normalize so probabilities sum exactly to 1 (avoids sklearn warnings
        # and keeps the distribution valid after rounding).
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

    def save(self, name: str = "xgboost_football", metrics: dict[str, Any] | None = None) -> dict[str, Path]:
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
        if name == "xgboost_football":
            save_manifest(manifest)
        else:
            named_manifest = MODELS_DIR / f"model_manifest_{name}.json"
            save_manifest(manifest, path=named_manifest)
        return paths

    @classmethod
    def load(cls, name: str = "xgboost_football") -> XGBoostFootballPredictor:
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
    """Convenience CLI entrypoint: build features, train, save."""
    from predictors.feature_engineering import build_training_dataset, save_features

    print("Building training dataset...")
    train = build_training_dataset(min_date=min_date)
    save_features(train, "train_historical")

    print(f"Training on {len(train)} rows...")
    predictor = XGBoostFootballPredictor(random_state=2026)
    metrics = predictor.fit(train, calibrate=True)

    print("Saving model...")
    paths = predictor.save()
    print(f"Saved models to {MODELS_DIR}")

    return {"metrics": metrics, "paths": {k: str(v) for k, v in paths.items()}}


if __name__ == "__main__":
    result = train_and_save()
    print(json.dumps(result, indent=2, default=str))
