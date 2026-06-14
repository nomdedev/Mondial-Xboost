"""
XGBoost Predictor — Football Match Prediction Engine
=====================================================
Trains XGBoost models to predict match outcomes, goals, and betting markets.

Models:
- outcome: Multiclass (home win / draw / away win)
- home_goals: Regression
- away_goals: Regression
- btts: Binary classification
- over_2_5: Binary classification

Usage:
    from xgboost_predictor import XGBoostPredictor
    predictor = XGBoostPredictor()
    predictor.train()
    predictions = predictor.predict(fixture_features)
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix
)


class XGBoostPredictor:
    MODELS_DIR = Path("data/models")
    FEATURES_DIR = Path("data/features")
    
    # Model configurations
    MODEL_CONFIGS = {
        "outcome": {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "home_goals": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 150,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "away_goals": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 150,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "btts": {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 150,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "over_2_5": {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 150,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
    }
    
    def __init__(self):
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, xgb.XGBModel] = {}
        self.feature_names: List[str] = []
        self.training_metrics: Dict = {}
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load X_train and y_train from parquet files."""
        # Find latest files
        X_files = sorted(self.FEATURES_DIR.glob("X_train_*.parquet"))
        y_files = sorted(self.FEATURES_DIR.glob("y_train_*.parquet"))
        
        if not X_files or not y_files:
            raise FileNotFoundError("No training data found. Run ETL and feature engineering first.")
        
        X = pd.read_parquet(X_files[-1])
        y = pd.read_parquet(y_files[-1])
        
        self.feature_names = list(X.columns)
        
        return X, y
    
    def train_outcome_model(self, X: pd.DataFrame, y: pd.DataFrame) -> xgb.XGBClassifier:
        """Train multiclass model for match outcome."""
        print("\n[1/5] Training outcome model...")
        
        config = self.MODEL_CONFIGS["outcome"].copy()
        objective = config.pop("objective")
        eval_metric = config.pop("eval_metric")
        num_class = config.pop("num_class")
        
        model = xgb.XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            num_class=num_class,
            **config
        )
        
        # Prepare data
        y_outcome = y["outcome"].astype(int)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_outcome, test_size=0.2, random_state=42, stratify=y_outcome
        )
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_proba)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y_outcome, cv=cv, scoring="accuracy")
        
        self.training_metrics["outcome"] = {
            "accuracy": accuracy,
            "log_loss": ll,
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_importance": dict(zip(self.feature_names, model.feature_importances_.tolist()))
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Log-loss: {ll:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return model
    
    def train_goals_model(self, X: pd.DataFrame, y: pd.DataFrame, target: str) -> xgb.XGBRegressor:
        """Train regression model for goals."""
        print(f"\n[{'2' if target == 'home_goals' else '3'}/5] Training {target} model...")
        
        config = self.MODEL_CONFIGS[target].copy()
        objective = config.pop("objective")
        eval_metric = config.pop("eval_metric")
        
        model = xgb.XGBRegressor(
            objective=objective,
            eval_metric=eval_metric,
            **config
        )
        
        y_goals = y[target].astype(float)
        
        # Remove rows with NaN targets
        mask = ~y_goals.isna()
        X_clean = X[mask]
        y_clean = y_goals[mask]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.training_metrics[target] = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "feature_importance": dict(zip(self.feature_names, model.feature_importances_.tolist()))
        }
        
        print(f"  RMSE: {np.sqrt(mse):.4f}")
        print(f"  MAE: {mae:.4f}")
        
        return model
    
    def train_binary_model(self, X: pd.DataFrame, y: pd.DataFrame, target: str) -> xgb.XGBClassifier:
        """Train binary classification model."""
        print(f"\n[{'4' if target == 'btts' else '5'}/5] Training {target} model...")
        
        config = self.MODEL_CONFIGS[target].copy()
        objective = config.pop("objective")
        eval_metric = config.pop("eval_metric")
        
        model = xgb.XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            **config
        )
        
        y_binary = y[target].astype(int)
        
        mask = ~y_binary.isna()
        X_clean = X[mask]
        y_clean = y_binary[mask]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)
        
        self.training_metrics[target] = {
            "accuracy": accuracy,
            "brier_score": brier,
            "feature_importance": dict(zip(self.feature_names, model.feature_importances_.tolist()))
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        
        return model
    
    def train(self):
        """Train all models."""
        print("=== XGBoost Training Start ===")
        
        X, y = self.load_training_data()
        print(f"Loaded training data: X={X.shape}, y={y.shape}")
        
        # Train models
        self.models["outcome"] = self.train_outcome_model(X, y)
        self.models["home_goals"] = self.train_goals_model(X, y, "home_goals")
        self.models["away_goals"] = self.train_goals_model(X, y, "away_goals")
        self.models["btts"] = self.train_binary_model(X, y, "btts")
        self.models["over_2_5"] = self.train_binary_model(X, y, "over_2_5")
        
        # Save models
        self.save_models()
        
        # Save metrics
        self.save_metrics()
        
        print("\n=== XGBoost Training Complete ===")
        
        return self.training_metrics
    
    def save_models(self):
        """Save trained models to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            path = self.MODELS_DIR / f"{name}_{timestamp}.json"
            model.save_model(str(path))
            print(f"Saved {name} model to {path}")
        
        # Save feature names
        with open(self.MODELS_DIR / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
    
    def load_models(self):
        """Load latest models from disk."""
        for target in self.MODEL_CONFIGS.keys():
            model_files = sorted(self.MODELS_DIR.glob(f"{target}_*.json"))
            if not model_files:
                raise FileNotFoundError(f"No model found for {target}")
            
            latest = model_files[-1]
            
            if target in ["outcome", "btts", "over_2_5"]:
                model = xgb.XGBClassifier()
            else:
                model = xgb.XGBRegressor()
            
            model.load_model(str(latest))
            self.models[target] = model
            print(f"Loaded {target} model from {latest}")
        
        # Load feature names
        with open(self.MODELS_DIR / "feature_names.json") as f:
            self.feature_names = json.load(f)
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.MODELS_DIR / f"metrics_{timestamp}.json"
        
        with open(path, "w") as f:
            json.dump(self.training_metrics, f, indent=2)
        
        print(f"Saved metrics to {path}")
    
    def predict(self, features: pd.DataFrame) -> Dict[str, any]:
        """Predict match outcomes from features."""
        if not self.models:
            self.load_models()
        
        # Ensure features have same columns as training
        features = features.reindex(columns=self.feature_names, fill_value=0)
        
        # Predict
        outcome_proba = self.models["outcome"].predict_proba(features)[0]
        home_goals = self.models["home_goals"].predict(features)[0]
        away_goals = self.models["away_goals"].predict(features)[0]
        btts_proba = self.models["btts"].predict_proba(features)[0, 1]
        over_2_5_proba = self.models["over_2_5"].predict_proba(features)[0, 1]
        
        # Calculate derived probabilities
        home_win_prob = outcome_proba[2]
        draw_prob = outcome_proba[1]
        away_win_prob = outcome_proba[0]
        
        # Most likely score (round goals)
        most_likely_score = (round(max(0, home_goals)), round(max(0, away_goals)))
        
        return {
            "outcome_probabilities": {
                "home_win": float(home_win_prob),
                "draw": float(draw_prob),
                "away_win": float(away_win_prob)
            },
            "expected_goals": {
                "home": float(home_goals),
                "away": float(away_goals)
            },
            "most_likely_score": most_likely_score,
            "btts_probability": float(btts_proba),
            "over_2_5_probability": float(over_2_5_proba),
            "model_confidence": max(outcome_proba)
        }
    
    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict for multiple matches."""
        if not self.models:
            self.load_models()
        
        features_df = features_df.reindex(columns=self.feature_names, fill_value=0)
        
        outcome_proba = self.models["outcome"].predict_proba(features_df)
        home_goals = self.models["home_goals"].predict(features_df)
        away_goals = self.models["away_goals"].predict(features_df)
        btts_proba = self.models["btts"].predict_proba(features_df)[:, 1]
        over_2_5_proba = self.models["over_2_5"].predict_proba(features_df)[:, 1]
        
        results = pd.DataFrame({
            "home_win_prob": outcome_proba[:, 2],
            "draw_prob": outcome_proba[:, 1],
            "away_win_prob": outcome_proba[:, 0],
            "expected_home_goals": home_goals,
            "expected_away_goals": away_goals,
            "btts_prob": btts_proba,
            "over_2_5_prob": over_2_5_proba,
            "model_confidence": np.max(outcome_proba, axis=1)
        })
        
        return results


# ===== CLI =====

if __name__ == "__main__":
    import sys
    
    predictor = XGBoostPredictor()
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        metrics = predictor.train()
        
        print("\n=== Training Summary ===")
        for model, metric in metrics.items():
            print(f"\n{model}:")
            for key, value in metric.items():
                if key != "feature_importance" and key != "confusion_matrix":
                    print(f"  {key}: {value}")
    
    else:
        print("Usage: python xgboost_predictor.py train")
        print("\nXGBoostPredictor OK")
