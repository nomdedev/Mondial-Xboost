"""
Train XGBoost models from the terminal.

Usage:
    python scripts/train.py                      # canonical model with defaults
    python scripts/train.py --name mundial2026   # custom model name
    python scripts/train.py --loop --trials 50   # hyperparameter tuning
    python scripts/train.py --loop --auto        # 10 batches back-to-back
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows terminals so accented chars render without crashing.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from predictors.feature_engineering import FEATURE_COLS, build_training_dataset, save_features
from predictors.random_forest_engine import RandomForestFootballPredictor
from predictors.xgboost_engine import XGBoostFootballPredictor

# ANSI colors
C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "red": "\033[91m",
    "dim": "\033[2m",
}


def print_header(text: str) -> None:
    print(f"\n{C['bold']}{'=' * 70}{C['reset']}")
    print(f"{C['cyan']}{text}{C['reset']}")
    print(f"{C['bold']}{'=' * 70}{C['reset']}")


def print_metric(label: str, value: str, color: str = "reset") -> None:
    print(f"  {label:20s}: {C[color]}{value}{C['reset']}")


def train_canonical(
    name: str,
    min_date: str = "2010-01-01",
    elo_decay_halflife_years: float | None = None,
    elo_recent_years: float | None = 8.0,
    engine: str = "xgboost",
) -> None:
    """Train the canonical predictor and save it."""
    print_header(f"Mondial-Xboost — Entrenamiento Canónico ({engine})")

    start = time.time()
    print(f"{C['dim']}Construyendo dataset de entrenamiento...{C['reset']}")
    train = build_training_dataset(
        min_date=min_date,
        elo_decay_halflife_years=elo_decay_halflife_years,
        elo_recent_years=elo_recent_years,
    )
    save_features(train, "train_historical")

    print(f"\n{C['bold']}Dataset{C['reset']}")
    print_metric("Filas", f"{len(train):,}")
    print_metric("Features", f"{len([c for c in train.columns if c in FEATURE_COLS])}")
    print_metric("Rango", f"{train['date'].min().date()} -> {train['date'].max().date()}")

    if engine == "random_forest":
        print(f"\n{C['dim']}Entrenando RandomForestFootballPredictor...{C['reset']}")
        predictor = RandomForestFootballPredictor(random_state=2026)
    else:
        print(f"\n{C['dim']}Entrenando XGBoostFootballPredictor...{C['reset']}")
        predictor = XGBoostFootballPredictor(random_state=2026)
    metrics = predictor.fit(train, calibrate=True)

    # Training-set diagnostics for the manifest (labelled as training metrics)
    x_train = predictor._prepare_x(train)
    y_train = train["outcome"].astype(int)
    train_probs = predictor.outcome_model.predict_proba(x_train)
    train_preds = np.argmax(train_probs, axis=1)
    training_metrics = {
        "source": "training",
        "log_loss": float(round(log_loss(y_train, train_probs), 4)),
        "accuracy": float(round(accuracy_score(y_train, train_preds), 4)),
        "top_feature": max(metrics["feature_importance"], key=metrics["feature_importance"].get),
    }

    paths = predictor.save(name, metrics=training_metrics)

    elapsed = time.time() - start
    print(f"\n{C['bold']}Modelo guardado{C['reset']}")
    for key, path in paths.items():
        print_metric(key, str(path))

    print(f"\n{C['bold']}Métricas de entrenamiento{C['reset']}")
    print_metric("Filas usadas", f"{metrics['n_train']:,}")
    top_feature = max(metrics["feature_importance"], key=metrics["feature_importance"].get)
    print_metric("Feature top", f"{top_feature} ({metrics['feature_importance'][top_feature]:.4f})")
    print_metric("Tiempo", f"{elapsed:.1f}s", "yellow")

    print(f"\n{C['green']}[OK] Modelo '{name}' entrenado y guardado.{C['reset']}")


def train_loop(trials: int, auto: bool) -> None:
    """Run the hyperparameter tuning loop."""
    from scripts.loop_engineering import run_batch

    if auto:
        print_header("Mondial-Xboost — Loop Engineering (10 batches)")
        for batch in range(1, 11):
            run_batch(batch, trials)
    else:
        print_header("Mondial-Xboost — Loop Engineering")
        run_batch(1, trials)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ML models for Mondial-Xboost")
    parser.add_argument("--loop", action="store_true", help="Run hyperparameter tuning with Optuna")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per batch (default: 50)")
    parser.add_argument("--auto", action="store_true", help="Run 10 batches automatically")
    parser.add_argument("--name", type=str, default="xgboost_football", help="Model name for canonical training")
    parser.add_argument("--min-date", type=str, default="2010-01-01", help="Minimum date for training data")
    parser.add_argument("--elo-decay", type=float, default=None, help="Elo temporal decay half-life in years")
    parser.add_argument("--elo-recent", type=float, default=8.0, help="Recent Elo window in years")
    parser.add_argument("--engine", type=str, default="xgboost", choices=["xgboost", "random_forest"], help="ML engine to train (default: xgboost)")
    args = parser.parse_args()

    try:
        if args.loop:
            train_loop(args.trials, args.auto)
        else:
            train_canonical(
                args.name,
                args.min_date,
                elo_decay_halflife_years=args.elo_decay,
                elo_recent_years=args.elo_recent,
                engine=args.engine,
            )
        return 0
    except KeyboardInterrupt:
        print(f"\n{C['yellow']}[WARN] Entrenamiento interrumpido por el usuario.{C['reset']}")
        return 130
    except Exception as exc:
        print(f"\n{C['red']}[ERROR] {exc}{C['reset']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
