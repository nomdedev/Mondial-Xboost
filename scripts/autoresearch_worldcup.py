"""
Autoresearch loop for Mondial-Xboost.

Iterates over XGBoost hyperparameters, measuring performance on previous
World Cups. Keeps the configuration that improves target metrics.

Run:
    python scripts/autoresearch_worldcup.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np

from backtest.world_cup_backtest import run_all_backtests

EXPERIMENTS_DIR = Path(__file__).parent.parent / ".agents" / "logs" / "autoresearch"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Experiment:
    id: str
    description: str
    predictor_kwargs: dict[str, Any]
    metrics: dict[str, float]
    verdict: str  # KEEP, DISCARD, or PENDING


def _avg_metric(results: list, key: str) -> float:
    return float(np.mean([getattr(r, key) for r in results]))


def run_experiment(
    experiment_id: str,
    description: str,
    predictor_kwargs: dict[str, Any],
    years: list[int] | None = None,
) -> Experiment:
    """Run a single experiment."""
    try:
        results = run_all_backtests(years=years, predictor_kwargs=predictor_kwargs)
        metrics = {
            "log_loss": _avg_metric(results, "log_loss"),
            "brier_score": _avg_metric(results, "brier_score"),
            "top_pick_accuracy": _avg_metric(results, "top_pick_accuracy"),
            "calibration_error": _avg_metric(results, "calibration_error"),
            "roi_simulated": _avg_metric(results, "roi_simulated"),
        }
    except Exception as ex:
        metrics = {"error": str(ex)}

    return Experiment(
        id=experiment_id,
        description=description,
        predictor_kwargs=predictor_kwargs,
        metrics=metrics,
        verdict="PENDING",
    )


def evaluate_experiments() -> list[Experiment]:
    """Run baseline + variations and decide KEEP/DISCARD."""
    years = [2014, 2018, 2022]

    experiments = []

    # Baseline
    baseline = run_experiment(
        "EXP-000",
        "Baseline: max_depth=4, lr=0.05, n_estimators=300, no calibration",
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "calibrate": False},
        years=years,
    )
    experiments.append(baseline)

    # Variation 1: simpler trees, slower learning
    experiments.append(run_experiment(
        "EXP-001",
        "Conservative: max_depth=3, lr=0.03, n_estimators=400",
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 400, "calibrate": False},
        years=years,
    ))

    # Variation 2: add calibration
    experiments.append(run_experiment(
        "EXP-002",
        "Baseline + isotonic calibration",
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "calibrate": True},
        years=years,
    ))

    # Variation 3: more regularization
    experiments.append(run_experiment(
        "EXP-003",
        "More regularization: reg_lambda=2.0, reg_alpha=0.5",
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "reg_lambda": 2.0, "reg_alpha": 0.5, "calibrate": False},
        years=years,
    ))

    # Decide verdicts against baseline
    baseline_ll = baseline.metrics.get("log_loss", float("inf"))
    baseline_brier = baseline.metrics.get("brier_score", float("inf"))

    for exp in experiments[1:]:
        if "error" in exp.metrics:
            exp.verdict = "DISCARD"
            continue
        ll = exp.metrics.get("log_loss", float("inf"))
        brier = exp.metrics.get("brier_score", float("inf"))
        # Keep if log-loss improves and brier does not degrade more than 5%
        if ll < baseline_ll and brier < baseline_brier * 1.05:
            exp.verdict = "KEEP"
        else:
            exp.verdict = "DISCARD"

    return experiments


def save_experiments(experiments: list[Experiment]) -> Path:
    from datetime import datetime

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = EXPERIMENTS_DIR / f"worldcup_{timestamp}.json"
    data = {
        "experiments": [
            {
                "id": e.id,
                "description": e.description,
                "predictor_kwargs": e.predictor_kwargs,
                "metrics": e.metrics,
                "verdict": e.verdict,
            }
            for e in experiments
        ]
    }
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def main():
    print("=" * 70)
    print("AUTORESEARCH — World Cup Backtest Loop")
    print("=" * 70)

    experiments = evaluate_experiments()

    print("\nResults:")
    print(f"{'ID':<10} {'Description':<50} {'log-loss':<10} {'Brier':<10} {'Verdict':<10}")
    for e in experiments:
        if "error" in e.metrics:
            print(f"{e.id:<10} {e.description:<50} {'ERROR':<10} {'ERROR':<10} {e.verdict:<10}")
        else:
            ll = e.metrics.get("log_loss", -1)
            brier = e.metrics.get("brier_score", -1)
            print(f"{e.id:<10} {e.description:<50} {ll:<10.4f} {brier:<10.4f} {e.verdict:<10}")

    path = save_experiments(experiments)
    print(f"\nSaved experiments to {path}")


if __name__ == "__main__":
    main()
