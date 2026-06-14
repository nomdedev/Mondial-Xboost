#!/usr/bin/env python3
"""
run-backtest-gate.py — Mondial-Xboost Backtest Gate
====================================================

Ejecuta el backtest de Mundiales y compara métricas contra los thresholds del
loop-engineering. Devuelve PASS / WARNING / BLOCK.

Salida: `backtest/results/world_cup_backtest_summary.json`
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.world_cup_backtest import run_all_backtests  # noqa: E402

RESULTS_DIR = ROOT / "backtest" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "world_cup_backtest_summary.json"
GATES_PATH = ROOT / ".agents" / "skills" / "loop-engineering" / "gates.json"


def load_thresholds() -> dict[str, float]:
    try:
        with open(GATES_PATH, encoding="utf-8") as f:
            return json.load(f).get("thresholds", {})
    except Exception:
        return {}


def main() -> int:
    thresholds = load_thresholds()

    results = run_all_backtests(years=[2014, 2018, 2022])
    rows = [r.to_dict() for r in results]

    avg_log_loss = sum(r["log_loss"] for r in rows) / len(rows)
    avg_brier = sum(r["brier_score"] for r in rows) / len(rows)
    avg_accuracy = sum(r["top_pick_accuracy"] for r in rows) / len(rows)
    avg_roi = sum(r["roi_simulated"] for r in rows) / len(rows)

    checks = {
        "log_loss": (avg_log_loss, thresholds.get("backtest_log_loss", 1.05), "<"),
        "brier": (avg_brier, thresholds.get("backtest_brier", 0.22), "<"),
        "accuracy": (avg_accuracy, thresholds.get("backtest_accuracy", 0.45), ">"),
        "roi": (avg_roi, thresholds.get("backtest_roi", -0.05), ">"),
    }

    failures = []
    warnings = []
    for metric, (value, limit, op) in checks.items():
        if op == "<" and value > limit:
            failures.append(f"{metric}: {value:.4f} > {limit}")
        elif op == ">" and value < limit:
            failures.append(f"{metric}: {value:.4f} < {limit}")

    # A single tournament being catastrophic is a warning
    for r in rows:
        if r["log_loss"] > thresholds.get("backtest_log_loss", 1.05) * 1.1:
            warnings.append(f"WC {r['wc_year']} log_loss {r['log_loss']:.4f} is elevated")

    if failures:
        verdict = "BLOCK"
    elif warnings:
        verdict = "WARNING"
    else:
        verdict = "PASS"

    summary: dict[str, Any] = {
        "run_at": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "tournaments": rows,
        "average": {
            "log_loss": round(avg_log_loss, 4),
            "brier_score": round(avg_brier, 4),
            "top_pick_accuracy": round(avg_accuracy, 4),
            "roi_simulated": round(avg_roi, 4),
        },
        "thresholds": {
            "log_loss": thresholds.get("backtest_log_loss", 1.05),
            "brier": thresholds.get("backtest_brier", 0.22),
            "accuracy": thresholds.get("backtest_accuracy", 0.45),
            "roi": thresholds.get("backtest_roi", -0.05),
        },
        "failures": failures,
        "warnings": warnings,
    }

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print("World Cup Backtest Gate")
    print("=" * 60)
    for r in rows:
        print(f"WC {r['wc_year']}: log_loss={r['log_loss']:.4f} brier={r['brier_score']:.4f} "
              f"acc={r['top_pick_accuracy']:.2%} roi={r['roi_simulated']:.2%}")
    print("-" * 60)
    print(f"Average: log_loss={avg_log_loss:.4f} brier={avg_brier:.4f} "
          f"acc={avg_accuracy:.2%} roi={avg_roi:.2%}")
    print(f"Verdict: {verdict}")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
    if warnings:
        for w in warnings:
            print(f"  WARN: {w}")
    print(f"Report saved to {OUTPUT_PATH.relative_to(ROOT)}")

    return 0 if verdict in ("PASS", "WARNING") else 1


if __name__ == "__main__":
    sys.exit(main())
