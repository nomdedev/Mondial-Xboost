#!/usr/bin/env python3
"""
Training Orchestrator — Mondial-Xboost
=======================================

Pipeline autónomo de entrenamiento XGBoost:

1. Valida datos con Data Council.
2. Corre batches de Optuna con CV temporal hasta convergencia o límite.
3. Audita estabilidad de CV y convergencia después de cada batch.
4. Entrena modelo final con la estrategia estabilizada.
5. Ejecuta backtest gate y smoke test.
6. Escribe reporte consolidado.

Uso:
    python scripts/training_orchestrator.py --max-auto-batches 5 --trials-per-batch 50
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32" and sys.stdout.isatty():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.agents.check_cv_stability import audit as audit_cv
from scripts.agents.check_gpu_usage import audit as audit_gpu
from scripts.agents.check_optuna_convergence import audit as audit_convergence
from scripts.auto_loop_engineering import (
    analyze_results,
    generate_strategy,
    load_json_results as load_loop_results,
    next_batch_number,
    train_final_model,
)
from scripts.loop_engineering import run_batch

REPORT_PATH = ROOT / ".agents" / "logs" / "training-orchestrator-report.json"
BASELINE_PATH = ROOT / ".agents" / "logs" / "training-baseline.json"

C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "red": "\033[91m",
    "dim": "\033[2m",
}


def _print_header(text: str) -> None:
    print(f"\n{C['bold']}{'=' * 70}{C['reset']}")
    print(f"{C['cyan']}{text}{C['reset']}")
    print(f"{C['bold']}{'=' * 70}{C['reset']}")


def _print_metric(label: str, value: str, color: str = "reset") -> None:
    print(f"  {label:24s}: {C[color]}{value}{C['reset']}")


def run_command(cmd: list[str] | str, timeout: int = 1200) -> tuple[int, str, str]:
    """Ejecuta un comando y devuelve (code, stdout, stderr)."""
    if isinstance(cmd, list):
        cmd = [sys.executable if c == "python" else c for c in cmd]
    proc = subprocess.run(cmd, cwd=ROOT, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def ensure_baseline() -> dict[str, Any]:
    """Corre Data Council y Backtest si no existe baseline."""
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    _print_header("Fase 0 — Baseline")
    baseline: dict[str, Any] = {"recorded_at": datetime.now(UTC).isoformat()}

    code, out, err = run_command([sys.executable, str(ROOT / "scripts" / "run_data_council.py")], timeout=600)
    baseline["data_council"] = {"code": code, "output": (out + "\n" + err).strip()}

    code, out, err = run_command([sys.executable, str(ROOT / "scripts" / "run_backtest_gate.py")], timeout=1200)
    baseline["backtest"] = {"code": code, "output": (out + "\n" + err).strip()}
    summary_path = ROOT / "backtest" / "results" / "world_cup_backtest_summary.json"
    if summary_path.exists():
        baseline["backtest"]["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, default=str), encoding="utf-8")
    return baseline


def data_council_gate() -> dict[str, Any]:
    _print_header("Data Council Gate")
    code, out, err = run_command([sys.executable, str(ROOT / "scripts" / "run_data_council.py")], timeout=600)
    combined = (out + "\n" + err).strip()
    print(combined[:2000])
    status = "PASS" if code == 0 else "BLOCK"
    return {"gate": "data_council", "status": status, "output": combined}


def gpu_gate() -> dict[str, Any]:
    _print_header("GPU Gate")
    result = audit_gpu()
    print(f"[{result['status']}] {result['message']}")
    return {"gate": "gpu", **result}


def cv_gate() -> dict[str, Any]:
    result = audit_cv()
    print(f"[{result['status']}] {result['message']}")
    print(f"  Recomendación: {result['recommendation']}")
    return {"gate": "cv_stability", **result}


def convergence_gate(min_trials: int, max_trials: int) -> dict[str, Any]:
    result = audit_convergence(min_trials=min_trials, max_trials=max_trials)
    print(f"[{result['status']}] {result['reason']}")
    if result.get("suggested_trials"):
        print(f"  Trials sugeridos: {result['suggested_trials']}")
    return {"gate": "optuna_convergence", **result}


def final_model_gate(experiment_name: str) -> dict[str, Any]:
    _print_header("Final Model Gate")
    data = load_loop_results()
    analysis = analyze_results(data)
    best = analysis["best_run"]
    strategy, adjustments = generate_strategy(best)
    paths, metrics, training_metrics = train_final_model(strategy, experiment_name)
    return {
        "gate": "final_model",
        "status": "PASS",
        "experiment_name": experiment_name,
        "strategy": strategy,
        "adjustments": adjustments,
        "paths": {k: str(v) for k, v in paths.items()},
        "metrics": metrics,
        "training_metrics": training_metrics,
    }


def backtest_gate() -> dict[str, Any]:
    _print_header("Backtest Gate")
    code, out, err = run_command([sys.executable, str(ROOT / "scripts" / "run_backtest_gate.py")], timeout=1200)
    combined = (out + "\n" + err).strip()
    print(combined[:2000])
    summary_path = ROOT / "backtest" / "results" / "world_cup_backtest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    status = summary.get("verdict", "UNKNOWN")
    return {"gate": "backtest", "status": status, "summary": summary, "output": combined}


def bridge_smoke_gate() -> dict[str, Any]:
    _print_header("Bridge Smoke Gate")
    code, out, err = run_command([sys.executable, str(ROOT / "scripts" / "run_bridge_smoke_test.py")], timeout=300)
    combined = (out + "\n" + err).strip()
    print(combined[:1500])
    status = "PASS" if code == 0 else "BLOCK"
    return {"gate": "bridge_smoke", "status": status, "output": combined}


def promote_to_canonical(experiment_name: str) -> dict[str, Any]:
    """Copia el experimento como xgboost_football canónico."""
    _print_header("Promoción a canónico")
    import shutil
    src = ROOT / "data" / "models"
    dst = src
    for suffix in ["_outcome.pkl", "_home_goals.pkl", "_away_goals.pkl", "_meta.json"]:
        shutil.copy(dst / f"{experiment_name}{suffix}", dst / f"xgboost_football{suffix}")
    print(f"  {C['green']}Modelo {experiment_name} promovido a xgboost_football{C['reset']}")
    return {"gate": "promote", "status": "PASS", "canonical": "xgboost_football"}


def run_orchestrator(
    max_auto_batches: int = 5,
    trials_per_batch: int = 50,
    max_trials: int = 500,
    cv: bool = True,
    cv_folds: int = 3,
    cv_embargo: int = 60,
    cv_lambda: float = 2.0,
    use_gpu: bool = True,
    study_name: str = "xgboost_loop",
    experiment_name: str | None = None,
    promote: bool = False,
) -> dict[str, Any]:
    _print_header("Mondial-Xboost — Training Orchestrator")
    start = time.time()

    report: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "config": {
            "max_auto_batches": max_auto_batches,
            "trials_per_batch": trials_per_batch,
            "max_trials": max_trials,
            "cv": cv,
            "cv_folds": cv_folds,
            "cv_embargo": cv_embargo,
            "cv_lambda": cv_lambda,
            "use_gpu": use_gpu,
            "study_name": study_name,
        },
        "gates": [],
    }

    # Baseline
    baseline = ensure_baseline()
    report["baseline"] = baseline

    # Data council gate
    dc = data_council_gate()
    report["gates"].append(dc)
    if dc["status"] == "BLOCK":
        return _finalize(report, start, "BLOCK", "Data Council bloqueó el pipeline.")

    # GPU gate
    gpu = gpu_gate()
    report["gates"].append(gpu)
    if gpu["status"] == "BLOCK":
        return _finalize(report, start, "BLOCK", "GPU en estado crítico.")

    # Adaptive Optuna loop
    converged = False
    for batch_idx in range(1, max_auto_batches + 1):
        _print_header(f"Adaptive Batch {batch_idx}/{max_auto_batches}")
        batch_num = next_batch_number()
        run_batch(
            batch_num=batch_num,
            n_trials=trials_per_batch,
            cv=cv,
            cv_folds=cv_folds,
            cv_embargo=cv_embargo,
            cv_lambda=cv_lambda,
            use_gpu=use_gpu,
            study_name=study_name,
        )

        # CV audit
        cv_audit = cv_gate()
        report["gates"].append(cv_audit)
        if cv_audit["status"] == "BLOCK":
            return _finalize(report, start, "BLOCK", "CV inestable.")

        # Convergence audit
        conv = convergence_gate(min_trials=trials_per_batch, max_trials=max_trials)
        report["gates"].append(conv)
        if conv["status"] == "BLOCK":
            return _finalize(report, start, "BLOCK", conv.get("reason", "Convergence auditor bloqueó."))
        if conv["status"] == "CONVERGED":
            converged = True
            break

    if not converged:
        print(f"\n{C['yellow']}[WARN] No se alcanzó convergencia después de {max_auto_batches} batches.{C['reset']}")

    # Final model
    exp_num = 1
    exp_dir = ROOT / "docs" / "vault" / "05-Research"
    if exp_dir.exists():
        nums = [int(p.stem.split("-")[1]) for p in exp_dir.glob("exp-*.md") if p.stem.split("-")[1].isdigit()]
        exp_num = max(nums, default=0) + 1
    final_name = experiment_name or f"xgboost_football_exp_{exp_num:02d}_orchestrated"
    final_gate = final_model_gate(final_name)
    report["gates"].append(final_gate)

    # Backtest gate
    bt = backtest_gate()
    report["gates"].append(bt)
    if bt["status"] == "BLOCK":
        return _finalize(report, start, "BLOCK", "Backtest gate bloqueó.")

    # Bridge smoke
    smoke = bridge_smoke_gate()
    report["gates"].append(smoke)
    if smoke["status"] == "BLOCK":
        return _finalize(report, start, "BLOCK", "Bridge smoke test falló.")

    # Promote
    if promote:
        prom = promote_to_canonical(final_name)
        report["gates"].append(prom)

    return _finalize(report, start, "PASS", "Pipeline completado exitosamente.")


def _finalize(report: dict[str, Any], start: float, status: str, message: str) -> dict[str, Any]:
    report["status"] = status
    report["message"] = message
    report["finished_at"] = datetime.now(UTC).isoformat()
    report["elapsed_seconds"] = round(time.time() - start, 1)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _print_header("Orquestador — Resumen")
    _print_metric("Status", status, "green" if status == "PASS" else "red")
    _print_metric("Mensaje", message)
    _print_metric("Reporte", str(REPORT_PATH.relative_to(ROOT)), "cyan")
    print("")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Training Orchestrator — XGBoost + Optuna + CV + agentes")
    parser.add_argument("--max-auto-batches", type=int, default=5, help="Máximo de batches automáticos")
    parser.add_argument("--trials-per-batch", type=int, default=50, help="Trials por batch")
    parser.add_argument("--max-trials", type=int, default=500, help="Máximo absoluto de trials")
    parser.add_argument("--no-cv", action="store_true", help="Desactiva CV temporal")
    parser.add_argument("--cv-folds", type=int, default=3, help="Folds de CV")
    parser.add_argument("--cv-embargo", type=int, default=60, help="Días de embargo")
    parser.add_argument("--cv-lambda", type=float, default=2.0, help="Penalización std en CV")
    parser.add_argument("--no-gpu", action="store_true", help="Forzar CPU")
    parser.add_argument("--study-name", type=str, default="xgboost_loop", help="Nombre del study Optuna")
    parser.add_argument("--name", type=str, default=None, help="Nombre del experimento final")
    parser.add_argument("--promote", action="store_true", help="Promover resultado a xgboost_football canónico")
    args = parser.parse_args()

    try:
        run_orchestrator(
            max_auto_batches=args.max_auto_batches,
            trials_per_batch=args.trials_per_batch,
            max_trials=args.max_trials,
            cv=not args.no_cv,
            cv_folds=args.cv_folds,
            cv_embargo=args.cv_embargo,
            cv_lambda=args.cv_lambda,
            use_gpu=not args.no_gpu,
            study_name=args.study_name,
            experiment_name=args.name,
            promote=args.promote,
        )
        return 0
    except KeyboardInterrupt:
        print(f"\n{C['yellow']}[WARN] Interrumpido por el usuario.{C['reset']}")
        return 130
    except Exception as exc:
        print(f"\n{C['red']}[ERROR] {exc}{C['reset']}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
