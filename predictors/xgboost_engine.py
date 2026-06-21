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


def _compute_temporal_weights(
    dates: pd.Series,
    halflife_years: float | None,
    reference_date: pd.Timestamp | None = None,
) -> np.ndarray | None:
    """Return sample weights decaying exponentially with match age.

    A match exactly ``halflife_years`` old gets half the weight of the most
    recent match. If ``halflife_years`` is None, all weights are equal (None).
    """
    if halflife_years is None or halflife_years <= 0:
        return None

    reference = reference_date or dates.max()
    days = (reference - pd.to_datetime(dates)).dt.days.astype(float)
    # Clip at 0 to avoid future-dated fixtures receiving weight > 1
    days = np.maximum(days, 0.0)
    halflife_days = halflife_years * 365.25
    # Normalize so the most recent match has weight 1.0
    weights = np.exp(-np.log(2) * days / halflife_days)
    return weights.astype(np.float32)


def _fillna(x: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with sensible defaults; create missing columns."""
    x = x.copy()

    defaults = {
        "elo_diff": 0.0,
        "elo_diff_recent": 0.0,
        "home_points_avg_5": 0.5,
        "home_points_avg_10": 0.5,
        "away_points_avg_5": 0.5,
        "away_points_avg_10": 0.5,
        "home_win_rate_10": 0.0,
        "home_draw_rate_10": 0.0,
        "home_loss_rate_10": 0.0,
        "away_win_rate_10": 0.0,
        "away_draw_rate_10": 0.0,
        "away_loss_rate_10": 0.0,
        "h2h_last_result": 0.0,
        "home_goals_scored_avg_10": 1.3,
        "home_goals_conceded_avg_10": 1.3,
        "away_goals_scored_avg_10": 1.3,
        "away_goals_conceded_avg_10": 1.3,
        "h2h_goals_avg": 1.3,
        "home_matches_played": 0,
        "away_matches_played": 0,
        "h2h_wins_diff": 0.0,
        "h2h_years_since": 20.0,
        "neutral": False,
        "home_momentum_3": 0.0,
        "away_momentum_3": 0.0,
        "home_sos_5": 1500.0,
        "away_sos_5": 1500.0,
        "home_points_weighted_10": 0.5,
        "away_points_weighted_10": 0.5,
        "tournament_importance": 1.0,
    }

    for col, default in defaults.items():
        if col not in x.columns:
            x[col] = default
        if col in ("home_matches_played", "away_matches_played"):
            x[col] = x[col].fillna(default).astype(int)
        elif col == "neutral":
            x[col] = x[col].fillna(default).astype(bool)
        else:
            x[col] = x[col].fillna(default)
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
        x = _fillna(df)
        return x[self.feature_cols]

    def fit(
        self,
        train_df: pd.DataFrame,
        calibrate: bool | None = None,
        temporal_decay_halflife_years: float | None = None,
    ) -> dict[str, Any]:
        """Train models on the supplied feature dataframe.

        If ``temporal_decay_halflife_years`` is set, older matches receive
        exponentially lower sample weights (half weight at the given age).
        """
        if calibrate is None:
            calibrate = self.calibrate

        # Drop rows with missing targets
        train_df = train_df.dropna(subset=["outcome", "home_score", "away_score"]).copy()

        x = self._prepare_x(train_df)
        y_outcome = train_df["outcome"].astype(int)
        sample_weight = _compute_temporal_weights(
            train_df["date"], temporal_decay_halflife_years
        )

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

        clf.fit(x, y_outcome, sample_weight=sample_weight, verbose=False)

        if calibrate:
            self.outcome_model = CalibratedClassifierCV(clf, method="isotonic", cv=3)
            self.outcome_model.fit(x, y_outcome, sample_weight=sample_weight)
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


def train_and_save(
    min_date: str = "2010-01-01",
    temporal_decay_halflife_years: float | None = None,
) -> dict[str, Any]:
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
    if temporal_decay_halflife_years:
        print(f"Temporal decay half-life: {temporal_decay_halflife_years} years")
    predictor = XGBoostFootballPredictor(random_state=2026)
    fit_result = predictor.fit(
        train,
        calibrate=True,
        temporal_decay_halflife_years=temporal_decay_halflife_years,
    )

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
        "temporal_decay_halflife_years": temporal_decay_halflife_years,
    }

    print("Saving model...")
    name = "xgboost_football"
    if temporal_decay_halflife_years:
        name = f"xgboost_football_decay{temporal_decay_halflife_years}y"
    paths = predictor.save(name=name, metrics=metrics)
    print(f"Saved models to {MODELS_DIR}")

    return {"metrics": metrics, "paths": {k: str(v) for k, v in paths.items()}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-date", default="2010-01-01")
    parser.add_argument("--temporal-decay-halflife-years", type=float, default=None)
    args = parser.parse_args()
    result = train_and_save(
        min_date=args.min_date,
        temporal_decay_halflife_years=args.temporal_decay_halflife_years,
    )
    print(json.dumps(result, indent=2, default=str))
