#!/usr/bin/env python3
"""
Auto Loop Engineering — Mondial-Xboost
=======================================

Orquesta un ciclo completo de loop engineering:

1. Corre N trials de Optuna sobre XGBoost.
2. Analiza los resultados con un score compuesto (accuracy - overfit - log_loss).
3. Genera una nueva estrategia estabilizada a partir del mejor trial.
4. Reentrena un modelo final con esa estrategia.
5. Documenta el experimento en docs/vault/05-Research/.
6. Opcionalmente ejecuta el World Cup backtest gate.

Usage:
    python scripts/auto_loop_engineering.py
    python scripts/auto_loop_engineering.py --trials 100 --name exp-04-auto-loop --backtest
    python scripts/auto_loop_engineering.py --trials 10 --tune-only
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows terminals.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from predictors.feature_engineering import FEATURE_COLS, build_training_dataset, save_features  # noqa: E402
from predictors.xgboost_engine import XGBoostFootballPredictor  # noqa: E402
from scripts.loop_engineering import load_json_results, run_batch  # noqa: E402
from scripts.training_monitor import monitor  # noqa: E402

MODELS_DIR = ROOT / "data" / "models"
RESEARCH_DIR = ROOT / "docs" / "vault" / "05-Research"
BACKTEST_SUMMARY = ROOT / "backtest" / "results" / "world_cup_backtest_summary.json"

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


def _print_header(text: str) -> None:
    print(f"\n{C['bold']}{'=' * 70}{C['reset']}")
    print(f"{C['cyan']}{text}{C['reset']}")
    print(f"{C['bold']}{'=' * 70}{C['reset']}")


def _print_metric(label: str, value: str, color: str = "reset") -> None:
    print(f"  {label:24s}: {C[color]}{value}{C['reset']}")


def next_experiment_number() -> int:
    """Return the next available experiment number from docs/vault/05-Research/."""
    max_num = 0
    if RESEARCH_DIR.exists():
        for path in RESEARCH_DIR.glob("exp-*.md"):
            parts = path.stem.split("-")
            if len(parts) >= 2 and parts[1].isdigit():
                max_num = max(max_num, int(parts[1]))
    return max_num + 1


def next_batch_number() -> int:
    """Return the next batch number based on existing loop_engineering.json."""
    data = load_json_results()
    batches = data.get("batches", [])
    if not batches:
        return 1
    return max(b.get("batch", 0) for b in batches) + 1


def composite_score(run: dict[str, Any]) -> float:
    """Score that rewards accuracy and penalizes overfit and log-loss."""
    test_acc = run.get("test_accuracy", 0.0) / 100.0
    gap = run.get("overfit_gap", 0.0) / 100.0
    ll = run.get("log_loss", 1.0)
    return test_acc - 0.5 * gap - 0.05 * ll


def analyze_results(data: dict[str, Any]) -> dict[str, Any]:
    """Analyze tuning results and return best run + statistics."""
    runs = data.get("all_runs", [])
    if not runs:
        raise ValueError("No hay runs de Optuna para analizar. Abortando.")

    best = max(runs, key=composite_score)
    top5 = sorted(runs, key=composite_score, reverse=True)[:5]

    return {
        "best_run": best,
        "top5": top5,
        "best_score": composite_score(best),
        "avg_test_accuracy": sum(r.get("test_accuracy", 0.0) for r in runs) / len(runs),
        "avg_overfit_gap": sum(r.get("overfit_gap", 0.0) for r in runs) / len(runs),
        "avg_log_loss": sum(r.get("log_loss", 0.0) for r in runs) / len(runs),
        "total_runs": len(runs),
    }


def generate_strategy(best_run: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Generate a stabilized strategy from the best Optuna trial."""
    params = dict(best_run.get("params", {}))

    strategy = {
        "n_estimators": int(params.get("n_estimators", 300)),
        "max_depth": int(params.get("max_depth", 4)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "reg_alpha": float(params.get("reg_alpha", 0.1)),
        "min_child_weight": int(params.get("min_child_weight", 1)),
        "gamma": float(params.get("gamma", 0.0)),
    }

    gap = best_run.get("overfit_gap", 0.0) / 100.0
    ll = best_run.get("log_loss", 1.0)
    adjustments: list[str] = []

    if gap > 0.10:
        adjustments.append(
            f"Overfit gap alto ({gap:.1%}): +reg_lambda, +reg_alpha, +min_child_weight, -max_depth"
        )
        strategy["reg_lambda"] = min(strategy["reg_lambda"] * 1.5, 10.0)
        strategy["reg_alpha"] = min(strategy["reg_alpha"] * 1.5, 10.0)
        strategy["min_child_weight"] = min(strategy["min_child_weight"] + 2, 10)
        strategy["max_depth"] = max(strategy["max_depth"] - 1, 3)

    if ll > 1.05:
        adjustments.append(f"Log loss alto ({ll:.3f}): -learning_rate, +n_estimators")
        strategy["learning_rate"] = max(strategy["learning_rate"] * 0.8, 0.01)
        strategy["n_estimators"] = min(int(strategy["n_estimators"] * 1.2), 1000)

    if strategy["subsample"] >= 0.99:
        adjustments.append("subsample=1.0: se reduce a 0.9 para bajar varianza")
        strategy["subsample"] = 0.9

    if strategy["colsample_bytree"] >= 0.99:
        adjustments.append("colsample_bytree=1.0: se reduce a 0.9")
        strategy["colsample_bytree"] = 0.9

    # Round for readability.
    for key in ("reg_lambda", "reg_alpha", "learning_rate", "subsample", "colsample_bytree", "gamma"):
        strategy[key] = round(strategy[key], 4)

    return strategy, adjustments


def train_final_model(strategy: dict[str, Any], experiment_name: str) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    """Train and save the final model with the generated strategy."""
    _print_header(f"Reentrenamiento final — {experiment_name}")

    print(f"{C['dim']}Construyendo dataset de entrenamiento...{C['reset']}")
    train = build_training_dataset(min_date="2010-01-01")
    save_features(train, f"train_historical_{experiment_name}")

    print(f"\n{C['bold']}Dataset{C['reset']}")
    _print_metric("Filas", f"{len(train):,}")
    _print_metric("Features", f"{len([c for c in train.columns if c in FEATURE_COLS])}")
    _print_metric("Rango", f"{train['date'].min().date()} -> {train['date'].max().date()}")

    print(f"\n{C['dim']}Hiperparámetros de la nueva estrategia:{C['reset']}")
    for key, value in strategy.items():
        _print_metric(key, str(value))

    predictor = XGBoostFootballPredictor(
        random_state=2026,
        n_estimators=strategy["n_estimators"],
        max_depth=strategy["max_depth"],
        learning_rate=strategy["learning_rate"],
        subsample=strategy["subsample"],
        colsample_bytree=strategy["colsample_bytree"],
        reg_lambda=strategy["reg_lambda"],
        reg_alpha=strategy["reg_alpha"],
        calibrate=True,
    )

    print(f"\n{C['dim']}Entrenando XGBoostFootballPredictor...{C['reset']}")
    metrics = predictor.fit(train, calibrate=True)

    # Training-set diagnostics for the manifest.
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

    print(f"\n{C['dim']}Guardando modelo y manifest...{C['reset']}")
    paths = predictor.save(experiment_name, metrics=training_metrics)

    print(f"\n{C['bold']}Modelo guardado{C['reset']}")
    for key, path in paths.items():
        _print_metric(key, str(path))

    print(f"\n{C['green']}[OK] Modelo '{experiment_name}' entrenado y guardado.{C['reset']}")
    return paths, metrics, training_metrics


def run_backtest_gate() -> dict[str, Any] | None:
    """Run the World Cup backtest gate and return its summary."""
    _print_header("World Cup Backtest Gate")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_backtest_gate.py")],
        cwd=ROOT,
        check=False,
    )
    if BACKTEST_SUMMARY.exists():
        summary = json.loads(BACKTEST_SUMMARY.read_text(encoding="utf-8"))
        summary["return_code"] = result.returncode
        return summary
    return None


def write_experiment_note(
    exp_num: int,
    experiment_name: str,
    analysis: dict[str, Any],
    strategy: dict[str, Any],
    adjustments: list[str],
    paths: dict[str, Path],
    metrics: dict[str, Any],
    training_metrics: dict[str, Any],
    backtest_summary: dict[str, Any] | None,
) -> Path:
    """Write a Markdown experiment note following the project convention."""
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    note_path = RESEARCH_DIR / f"exp-{exp_num:02d}-auto-loop.md"

    best = analysis["best_run"]
    top_feature = training_metrics.get("top_feature", "-")

    lines = [
        f"# Exp-{exp_num:02d}: Auto Loop Engineering — {experiment_name}",
        "",
        "## Hipótesis",
        "",
        "Un loop de tuning automatizado puede descubrir hiperparámetros superiores al canónico "
        "y estabilizarlos contra el overfitting, generando un modelo reproducible y documentado.",
        "",
        "## Cambios realizados",
        "",
        "- Se ejecutó un batch de Optuna sobre XGBoost.",
        f"- Se analizaron {analysis['total_runs']} trials con un score compuesto "
        "(accuracy - 0.5·overfit_gap - 0.05·log_loss).",
        "- Se generó una estrategia estabilizada a partir del mejor trial.",
        f"- Se reentrenó y guardó el modelo `{experiment_name}`.",
        "",
    ]

    if adjustments:
        lines.extend(["### Ajustes de estrategia aplicados", ""])
        for adj in adjustments:
            lines.append(f"- {adj}")
        lines.append("")

    lines.extend([
        "## Métricas",
        "",
        "### Tuning",
        "",
        "| Métrica | Valor |",
        "| --- | --- |",
        f"| Mejor test accuracy | {best.get('test_accuracy', 0.0):.2f}% |",
        f"| Log loss (mejor) | {best.get('log_loss', 0.0):.4f} |",
        f"| Brier (mejor) | {best.get('brier', 0.0):.4f} |",
        f"| Overfit gap (mejor) | {best.get('overfit_gap', 0.0):.2f}% |",
        f"| Walk-forward acc (mejor) | {best.get('walk_forward_acc', 0.0):.2f}% |",
        f"| Score compuesto (mejor) | {analysis['best_score']:.4f} |",
        f"| Test accuracy promedio | {analysis['avg_test_accuracy']:.2f}% |",
        f"| Overfit gap promedio | {analysis['avg_overfit_gap']:.2f}% |",
        f"| Log loss promedio | {analysis['avg_log_loss']:.4f} |",
        "",
        "### Modelo final (entrenamiento)",
        "",
        "| Métrica | Valor |",
        "| --- | --- |",
        f"| Filas usadas | {metrics.get('n_train', 0):,} |",
        f"| Accuracy entrenamiento | {training_metrics.get('accuracy', 0.0):.4f} |",
        f"| Log loss entrenamiento | {training_metrics.get('log_loss', 0.0):.4f} |",
        f"| Feature top | {top_feature} ({metrics.get('feature_importance', {}).get(top_feature, 0.0):.4f}) |",
        "",
    ])

    lines.extend([
        "### Hiperparámetros finales",
        "",
        "| Parámetro | Valor |",
        "| --- | --- |",
    ])
    for key, value in strategy.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    if backtest_summary:
        avg = backtest_summary.get("average", {})
        verdict = backtest_summary.get("verdict", "-")
        lines.extend([
            "### World Cup backtest",
            "",
            "| Métrica | Valor |",
            "| --- | --- |",
            f"| Verdict | {verdict} |",
            f"| Accuracy promedio | {avg.get('top_pick_accuracy', 0.0):.2%} |",
            f"| Log loss promedio | {avg.get('log_loss', 0.0):.4f} |",
            f"| Brier promedio | {avg.get('brier_score', 0.0):.4f} |",
            f"| ROI simulado | {avg.get('roi_simulated', 0.0):.2%} |",
            "",
        ])
        for tournament in backtest_summary.get("tournaments", []):
            lines.append(
                f"- WC {tournament.get('wc_year')}: "
                f"acc={tournament.get('top_pick_accuracy', 0.0):.2%}, "
                f"log_loss={tournament.get('log_loss', 0.0):.4f}, "
                f"roi={tournament.get('roi_simulated', 0.0):.2%}"
            )
        lines.append("")

    lines.extend([
        "## Conclusiones",
        "",
        f"- El mejor trial alcanzó {best.get('test_accuracy', 0.0):.2f}% de test accuracy.",
        f"- La estrategia estabilizada usa {strategy['n_estimators']} estimadores, "
        f"max_depth={strategy['max_depth']}, learning_rate={strategy['learning_rate']}.",
        f"- Feature más importante: **{top_feature}**.",
        "",
        "## Decisión",
        "",
        "_Pendiente de revisión humana: adoptar / descartar / iterar._",
        "",
        "## Comandos",
        "",
        "```bash",
        "# Reproducir este experimento",
        f"./mondial auto-loop --trials 100 --name {experiment_name}",
        "",
        "# Usar el modelo entrenado",
        f"./mondial predecir --home Brazil --away Morocco --model {experiment_name}",
        "```",
        "",
    ])

    note_path.write_text("\n".join(lines), encoding="utf-8")
    return note_path


def run_auto_loop(
    n_trials: int = 100,
    experiment_name: str | None = None,
    tune_only: bool = False,
    backtest: bool = False,
    walk_forward: bool = True,
    aggressive: bool = False,
    label_smoothing: bool = False,
) -> dict[str, Any]:
    """Run the full auto-loop engineering cycle."""
    _print_header("Mondial-Xboost — Auto Loop Engineering")

    exp_num = next_experiment_number()
    if experiment_name is None:
        experiment_name = f"xgboost_football_exp_{exp_num:02d}_auto_loop"

    print(f"{C['dim']}Experimento:{C['reset']} {experiment_name}")
    print(f"{C['dim']}Trials:{C['reset']} {n_trials}")
    print(f"{C['dim']}Walk-forward:{C['reset']} {walk_forward}")

    # 1. Tuning
    batch_num = next_batch_number()
    print(f"\n{C['bold']}1) Loop Engineering (Optuna){C['reset']}")
    print(f"  Agresivo: {aggressive} | Label Smoothing: {label_smoothing}")
    run_batch(batch_num=batch_num, n_trials=n_trials, walk_forward=walk_forward, aggressive=aggressive, label_smoothing=label_smoothing)
    monitor.set_phase("analysis")

    # 2. Analysis
    print(f"\n{C['bold']}2) Análisis de resultados{C['reset']}")
    data = load_json_results()
    analysis = analyze_results(data)
    best = analysis["best_run"]

    _print_metric("Total runs", str(analysis["total_runs"]))
    _print_metric("Mejor test accuracy", f"{best.get('test_accuracy', 0.0):.2f}%", "green")
    _print_metric("Mejor log loss", f"{best.get('log_loss', 0.0):.4f}")
    _print_metric("Mejor overfit gap", f"{best.get('overfit_gap', 0.0):.2f}%")
    _print_metric("Score compuesto", f"{analysis['best_score']:.4f}", "cyan")

    # 3. Strategy
    print(f"\n{C['bold']}3) Generación de nueva estrategia{C['reset']}")
    monitor.set_phase("strategy")
    strategy, adjustments = generate_strategy(best)
    if adjustments:
        for adj in adjustments:
            print(f"  • {adj}")
    else:
        print("  • Sin ajustes necesarios; se usa la estrategia del mejor trial.")

    if tune_only:
        print(f"\n{C['yellow']}[INFO] Modo tune-only: no se reentrena ni se documenta.{C['reset']}")
        return {
            "experiment_name": experiment_name,
            "analysis": analysis,
            "strategy": strategy,
            "adjustments": adjustments,
        }

    # 4. Retrain
    print(f"\n{C['bold']}4) Reentrenamiento con nueva estrategia{C['reset']}")
    monitor.set_phase("retraining")
    paths, metrics, training_metrics = train_final_model(strategy, experiment_name)

    # 5. Backtest (optional)
    backtest_summary = None
    if backtest:
        print(f"\n{C['bold']}5) Backtest gate{C['reset']}")
        backtest_summary = run_backtest_gate()
    else:
        print(f"\n{C['dim']}5) Backtest omitido (usar --backtest para correrlo){C['reset']}")

    # 6. Documentation
    print(f"\n{C['bold']}6) Documentación del experimento{C['reset']}")
    monitor.set_phase("documentation")
    note_path = write_experiment_note(
        exp_num=exp_num,
        experiment_name=experiment_name,
        analysis=analysis,
        strategy=strategy,
        adjustments=adjustments,
        paths=paths,
        metrics=metrics,
        training_metrics=training_metrics,
        backtest_summary=backtest_summary,
    )
    _print_metric("Nota del experimento", str(note_path.relative_to(ROOT)), "green")

    return {
        "experiment_name": experiment_name,
        "experiment_note": str(note_path),
        "analysis": analysis,
        "strategy": strategy,
        "adjustments": adjustments,
        "paths": {k: str(v) for k, v in paths.items()},
        "backtest_summary": backtest_summary,
    }


def _complete_monitor(result: dict[str, Any]) -> None:
    monitor.complete(
        f"Experimento {result['experiment_name']} completado. "
        f"Mejor test accuracy: {result['analysis']['best_run'].get('test_accuracy', 0.0):.2f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto Loop Engineering — tuning, análisis, estrategia, reentreno y documentación",
    )
    parser.add_argument("--trials", type=int, default=100, help="Trials de Optuna (default: 100)")
    parser.add_argument("--name", type=str, default=None, help="Nombre del experimento/modelo")
    parser.add_argument("--tune-only", action="store_true", help="Solo tuning; no reentrena ni documenta")
    parser.add_argument("--backtest", action="store_true", help="Ejecutar World Cup backtest gate al final")
    parser.add_argument("--no-walk-forward", action="store_true", help="Deshabilitar walk-forward validation")
    parser.add_argument("--aggressive", action="store_true", help="Espacio de búsqueda agresivo (riesgo alto)")
    parser.add_argument("--label-smoothing", action="store_true", help="Usa label smoothing con prior Elo")
    args = parser.parse_args()

    try:
        result = run_auto_loop(
            n_trials=args.trials,
            experiment_name=args.name,
            tune_only=args.tune_only,
            backtest=args.backtest,
            walk_forward=not args.no_walk_forward,
            aggressive=args.aggressive,
            label_smoothing=args.label_smoothing,
        )
        _complete_monitor(result)
        print(f"\n{C['green']}[OK] Auto Loop Engineering completado: {result['experiment_name']}{C['reset']}")
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
    sys.exit(main())
