#!/usr/bin/env python3
"""
Optuna Convergence Auditor — Mondial-Xboost
============================================

Decide si Optuna ya convergió o si conviene lanzar más trials.

Reglas:
- Si no hay suficientes trials: MORE_TRIALS.
- Si el mejor trial es peor que baseline y ya se corrieron muchos trials: BLOCK.
- Si el mejor trial no mejora en los últimos batches y la varianza de los
  top trials es baja: CONVERGED.
- En cualquier otro caso: MORE_TRIALS.

Salida JSON con status CONVERGED / MORE_TRIALS / BLOCK.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_JSON = ROOT / "data" / "models" / "loop_engineering.json"
BASELINE_JSON = ROOT / ".agents" / "logs" / "training-baseline.json"


def load_results() -> dict[str, Any]:
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return {"all_runs": [], "best": None}


def load_baseline_accuracy(default: float = 45.0) -> float:
    if BASELINE_JSON.exists():
        try:
            baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
            acc = baseline.get("backtest", {}).get("average", {}).get("top_pick_accuracy")
            if acc is not None:
                return float(acc) * 100
        except Exception:
            pass
    return default


def audit(
    min_trials: int = 30,
    min_batches: int = 2,
    improvement_threshold: float = 0.1,
    top_n: int = 10,
    max_trials_hard: int = 500,
) -> dict[str, Any]:
    data = load_results()
    runs = data.get("all_runs", [])
    best_global = data.get("best")

    baseline_acc = load_baseline_accuracy()

    if not runs:
        return {
            "status": "MORE_TRIALS",
            "suggested_trials": min_trials,
            "reason": "No hay runs registrados.",
        }

    total_runs = len(runs)

    # Peor caso: muchos trials y no superamos baseline.
    current_best = max(r["test_accuracy"] for r in runs)
    if total_runs >= min_trials and current_best < baseline_acc:
        return {
            "status": "BLOCK",
            "suggested_trials": 0,
            "reason": (
                f"Mejor trial ({current_best:.2f}%) no supera el baseline "
                f"({baseline_acc:.2f}%) después de {total_runs} trials."
            ),
        }

    if total_runs >= max_trials_hard:
        return {
            "status": "CONVERGED",
            "suggested_trials": 0,
            "reason": f"Se alcanzó el límite máximo de {max_trials_hard} trials.",
        }

    if total_runs < min_trials:
        return {
            "status": "MORE_TRIALS",
            "suggested_trials": min_trials - total_runs,
            "reason": f"Solo {total_runs} trials; se requieren al menos {min_trials}.",
        }

    batches = {r.get("batch", 0) for r in runs}
    if len(batches) < min_batches:
        return {
            "status": "MORE_TRIALS",
            "suggested_trials": min_trials,
            "reason": f"Solo {len(batches)} batch(s); se requieren al menos {min_batches}.",
        }

    # Detectar plateau: comparar el mejor de los últimos trials con el anterior.
    sorted_runs = sorted(runs, key=lambda r: (r.get("batch", 0), r.get("trial", 0)))
    recent_window = max(min_trials, total_runs // 3)
    recent = sorted_runs[-recent_window:]
    previous = sorted_runs[:-recent_window]

    best_recent = max(r["test_accuracy"] for r in recent)
    best_previous = max(r["test_accuracy"] for r in previous) if previous else 0.0
    improvement = best_recent - best_previous

    # Varianza entre los mejores trials.
    top_runs = sorted(runs, key=lambda r: r["test_accuracy"], reverse=True)[:top_n]
    std_top = float(np.std([r["test_accuracy"] for r in top_runs]))

    if improvement < improvement_threshold and std_top < 1.0:
        return {
            "status": "CONVERGED",
            "suggested_trials": 0,
            "reason": (
                f"No hay mejora significativa ({improvement:.3f}pp) en los últimos "
                f"{recent_window} trials y los top-{top_n} son estables (std={std_top:.3f})."
            ),
        }

    # Sugerir más trials proporcional a la mejora observada.
    suggested = max(min_trials, int(total_runs * 0.5))
    return {
        "status": "MORE_TRIALS",
        "suggested_trials": suggested,
        "reason": (
            f"Mejora reciente={improvement:.3f}pp, std top-{top_n}={std_top:.3f}. "
            "Aún no se detecta convergencia."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna Convergence Auditor")
    parser.add_argument("--min-trials", type=int, default=30, help="Mínimo de trials antes de evaluar convergencia")
    parser.add_argument("--min-batches", type=int, default=2, help="Mínimo de batches antes de evaluar convergencia")
    parser.add_argument("--improvement-threshold", type=float, default=0.1, help="Mejora mínima significativa (pp)")
    parser.add_argument("--max-trials", type=int, default=500, help="Límite absoluto de trials")
    parser.add_argument("--json", action="store_true", help="Imprimir solo JSON")
    args = parser.parse_args()

    result = audit(
        min_trials=args.min_trials,
        min_batches=args.min_batches,
        improvement_threshold=args.improvement_threshold,
        max_trials_hard=args.max_trials,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[{result['status']}] {result['reason']}")
        if result["suggested_trials"]:
            print(f"Trials sugeridos: {result['suggested_trials']}")

    return 0 if result["status"] in ("CONVERGED", "MORE_TRIALS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
