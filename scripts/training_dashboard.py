"""
Training dashboard for the terminal.

Usage:
    python scripts/training_dashboard.py
    python scripts/training_dashboard.py --top 10
    python scripts/training_dashboard.py --evolution
    python scripts/training_dashboard.py --csv data/models/loop_engineering_partial.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "red": "\033[91m",
    "dim": "\033[2m",
}

RESULTS_JSON = Path(__file__).parent.parent / "data" / "models" / "loop_engineering.json"
PARTIAL_CSV = Path(__file__).parent.parent / "data" / "models" / "loop_engineering_partial.csv"


def color_value(value: float, threshold: float, higher_is_better: bool = True) -> str:
    good = (value >= threshold) if higher_is_better else (value <= threshold)
    color = "green" if good else "yellow"
    return f"{C[color]}{value}{C['reset']}"


def ascii_histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    if not values:
        return "Sin datos"
    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        return "[" + "#" * width + "]"
    counts = [0] * bins
    for v in values:
        idx = int((v - min_v) / (max_v - min_v) * (bins - 1))
        counts[idx] += 1
    max_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        lo = min_v + (max_v - min_v) * i / bins
        hi = min_v + (max_v - min_v) * (i + 1) / bins
        bar = "#" * int(c / max_count * width)
        lines.append(f"{lo:6.2f}-{hi:6.2f} |{bar:<{width}s}| {c}")
    return "\n".join(lines)


def ascii_evolution(batches: list[dict]) -> str:
    if not batches:
        return "Sin batches"
    best_accs = [b["best"]["test_accuracy"] for b in batches]
    min_v, max_v = min(best_accs), max(best_accs)
    if min_v == max_v:
        return "[" + "#" * 40 + "]"
    lines = []
    for b, acc in zip(batches, best_accs, strict=False):
        bar_len = int((acc - min_v) / (max_v - min_v) * 40)
        bar = "#" * bar_len
        lines.append(f"Batch {b['batch']:2d} |{bar:<40s}| {acc:.2f}%")
    return "\n".join(lines)


def load_runs(source: str | None = None) -> list[dict]:
    runs = []
    # Default: prefer JSON full results; fallback to CSV partial.
    if source == "csv" and PARTIAL_CSV.exists():
        df = pd.read_csv(PARTIAL_CSV)
        for _, row in df.iterrows():
            runs.append(row.to_dict())
    elif source == "json" and RESULTS_JSON.exists():
        data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        runs.extend(data.get("all_runs", []))
    elif source is None:
        if RESULTS_JSON.exists():
            data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
            runs.extend(data.get("all_runs", []))
        elif PARTIAL_CSV.exists():
            df = pd.read_csv(PARTIAL_CSV)
            for _, row in df.iterrows():
                runs.append(row.to_dict())
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Training dashboard for Mondial-Xboost")
    parser.add_argument("--top", type=int, default=5, help="Show top N configurations")
    parser.add_argument("--evolution", action="store_true", help="Show ASCII evolution chart")
    parser.add_argument("--source", choices=["json", "csv"], help="Source: json results or csv partial")
    args = parser.parse_args()

    if not RESULTS_JSON.exists() and not PARTIAL_CSV.exists():
        print(f"{C['red']}No se encontraron datos de entrenamiento.{C['reset']}")
        print("Corré primero: python scripts/train.py --loop --trials 10")
        return 1

    runs = load_runs(args.source)
    if not runs:
        print(f"{C['yellow']}No hay runs cargadas todavía.{C['reset']}")
        return 0

    df = pd.DataFrame(runs)
    accs = df["test_accuracy"].tolist()
    gaps = df["overfit_gap"].tolist()

    print(f"\n{C['bold']}{C['cyan']}{'=' * 70}{C['reset']}")
    print(f"{C['bold']}{C['cyan']}  TRAINING DASHBOARD — Mondial-Xboost{C['reset']}")
    print(f"{C['bold']}{C['cyan']}{'=' * 70}{C['reset']}")

    print(f"\n{C['bold']}Resumen general{C['reset']}")
    print(f"  Total trials       : {len(runs)}")
    batches = df["batch"].nunique() if "batch" in df.columns else 0
    print(f"  Batches            : {batches}")
    print(f"  Mejor test accuracy: {color_value(max(accs), 55.0)}%")
    print(f"  Peor test accuracy : {min(accs):.2f}%")
    print(f"  Promedio           : {statistics.mean(accs):.2f}%")
    print(f"  Mediana            : {statistics.median(accs):.2f}%")
    print(f"  Desv. std          : {statistics.stdev(accs) if len(accs) > 1 else 0:.2f}%")

    print(f"\n{C['bold']}Distribución de test accuracy{C['reset']}")
    print(ascii_histogram(accs))

    print(f"\n{C['bold']}Distribución de overfit gap{C['reset']}")
    print(ascii_histogram(gaps, bins=8))

    high_gap = sum(1 for g in gaps if g > 15)
    if high_gap:
        print(f"\n{C['yellow']}! ATENCION: {high_gap} configs tienen overfit gap > 15% (riesgo de sobreajuste){C['reset']}")

    print(f"\n{C['bold']}Top {args.top} configuraciones (por test accuracy){C['reset']}")
    top = df.nlargest(args.top, "test_accuracy")
    print(f"{'Batch':>6s} {'Trial':>6s} {'Test%':>8s} {'Val%':>8s} {'LogLoss':>9s} {'Gap%':>7s} {'Params'}")
    print("-" * 120)
    for _, row in top.iterrows():
        params = row.get("params", {})
        if isinstance(params, str):
            params = json.loads(params.replace("'", '"'))
        params_str = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in list(params.items())[:4])
        print(
            f"{int(row.get('batch', 0)):6d} {int(row.get('trial', 0)):6d} "
            f"{row['test_accuracy']:8.2f} {row.get('val_accuracy', 0):8.2f} "
            f"{row['log_loss']:9.4f} {row['overfit_gap']:7.2f} {params_str}"
        )

    if args.evolution and RESULTS_JSON.exists():
        data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        batches = data.get("batches", [])
        if batches:
            print(f"\n{C['bold']}Evolución por batch (mejor test accuracy){C['reset']}")
            print(ascii_evolution(batches))

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
