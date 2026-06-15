#!/usr/bin/env python3
"""
Cross-Validation Stability Auditor — Mondial-Xboost
=====================================================

Verifica que la purged cross-validation usada por Optuna sea estable.
Si la varianza entre folds es muy alta, el score del trial no es confiable.

Salida JSON con status PASS / WARNING / BLOCK.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_JSON = ROOT / "data" / "models" / "loop_engineering.json"


def load_results() -> dict[str, Any]:
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return {"all_runs": []}


def audit(std_threshold: float = 3.0, stability_min: float = 20.0) -> dict[str, Any]:
    data = load_results()
    runs = data.get("all_runs", [])
    if not runs:
        return {
            "status": "BLOCK",
            "message": "No hay runs de Optuna para evaluar.",
            "recommendation": "Correr al menos un batch de tuning.",
        }

    # Usar los trials del último batch para medir estabilidad actual.
    last_batch = max(r.get("batch", 0) for r in runs)
    last_runs = [r for r in runs if r.get("batch") == last_batch and r.get("cv_std_acc", 0) > 0]

    if not last_runs:
        return {
            "status": "BLOCK",
            "message": f"El batch {last_batch} no tiene métricas de CV.",
            "recommendation": "Asegurarse de que --no-cv no esté activado.",
        }

    mean_std = sum(r["cv_std_acc"] for r in last_runs) / len(last_runs)
    mean_stability = sum(r.get("cv_stability_ratio", 0) for r in last_runs) / len(last_runs)
    mean_cv_acc = sum(r.get("cv_mean_acc", 0) for r in last_runs) / len(last_runs)

    worst = max(last_runs, key=lambda r: r["cv_std_acc"])

    if mean_std > std_threshold * 1.5 or worst["cv_std_acc"] > std_threshold * 2:
        status = "BLOCK"
    elif mean_std > std_threshold:
        status = "WARNING"
    else:
        status = "PASS"

    recommendation = "CV estable."
    if status != "PASS":
        recommendation = (
            f"Alta varianza entre folds (mean_std={mean_std:.2f}%). "
            "Considerar más datos, reducir max_depth o aumentar el embargo temporal."
        )

    return {
        "status": status,
        "last_batch": last_batch,
        "n_trials": len(last_runs),
        "cv_mean_acc": round(mean_cv_acc, 4),
        "cv_mean_std": round(mean_std, 4),
        "cv_mean_stability_ratio": round(mean_stability, 4),
        "worst_trial_std": round(worst["cv_std_acc"], 4),
        "message": f"CV mean acc={mean_cv_acc:.2f}% std={mean_std:.2f}% stability={mean_stability:.2f}",
        "recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CV Stability Auditor")
    parser.add_argument("--std-threshold", type=float, default=3.0, help="Umbral de std entre folds (%)")
    parser.add_argument("--stability-min", type=float, default=20.0, help="Mínimo ratio estabilidad")
    parser.add_argument("--json", action="store_true", help="Imprimir solo JSON")
    args = parser.parse_args()

    result = audit(std_threshold=args.std_threshold, stability_min=args.stability_min)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[{result['status']}] {result['message']}")
        print(f"Recomendación: {result['recommendation']}")

    return 0 if result["status"] == "PASS" else 1 if result["status"] == "BLOCK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
