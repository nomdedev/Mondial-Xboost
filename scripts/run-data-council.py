#!/usr/bin/env python3
"""
run-data-council.py — Mondial-Xboost Data Council Runner
=========================================================

Ejecuta las verificaciones del Data Council y genera un reporte consolidado:
  - Elo vs World Football Elo
  - Validación de pesos de jugadores
  - Audit de temporal leakage
  - Chequeos de schema / NaN / duplicados

Salida: `.agents/logs/data-council-report.json`
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
REPORT_PATH = ROOT / ".agents" / "logs" / "data-council-report.json"
RESULTS_DIR = ROOT / "backtest" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_command(cmd: str, cwd: Path = ROOT, timeout: int = 300) -> tuple[int, str, str]:
    cmd = cmd.strip()
    if cmd.startswith("python "):
        cmd = f"{sys.executable} {cmd[7:]}"
    elif cmd == "python":
        cmd = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    try:
        proc = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Timeout after {timeout}s"
    except FileNotFoundError as exc:
        return 127, "", f"Command not found: {exc}"


def check_elo() -> dict[str, Any]:
    code, stdout, stderr = run_command("python scripts/compare_elo_worldfootball.py")
    combined = (stdout + "\n" + stderr).strip()
    report_file = RESULTS_DIR / "elo_comparison.json"
    details: dict[str, Any] = {}
    if report_file.exists():
        try:
            with open(report_file, encoding="utf-8") as f:
                details = json.load(f)
        except Exception as exc:
            details = {"parse_error": str(exc)}

    top100 = details.get("top100_teams_stats", {})
    mean_diff = top100.get("mean_absolute_difference", 9999)
    corr = details.get("correlation", 0.0)
    status = "PASS" if (mean_diff < 100 and corr > 0.90) else "FAIL"
    return {
        "gate": "elo_calibration",
        "status": status,
        "mean_abs_diff_top100": mean_diff,
        "correlation_top100": corr,
        "output": combined,
        "evidence": str(report_file.relative_to(ROOT)),
    }


def check_player_weights() -> dict[str, Any]:
    try:
        sys.path.insert(0, str(ROOT))
        from predictors.player_weights import PlayerWeightingEngine

        weights = PlayerWeightingEngine.DEFAULT_WEIGHTS
        total = sum(weights.values())
        valid = abs(total - 1.0) < 1e-6
        status = "PASS" if valid else "FAIL"
        return {
            "gate": "player_weights_ok",
            "status": status,
            "weight_sum": round(total, 6),
            "n_weights": len(weights),
            "output": f"Weight sum = {total:.6f}",
            "evidence": "predictors/player_weights.py",
        }
    except Exception as exc:
        return {"gate": "player_weights_ok", "status": "FAIL", "output": str(exc), "evidence": ""}


def check_leakage() -> dict[str, Any]:
    audit_script = ROOT / "scripts" / "audit_leakage.py"
    if not audit_script.exists():
        return {"gate": "no_leakage", "status": "SKIP", "output": "scripts/audit_leakage.py not found", "evidence": ""}
    code, stdout, stderr = run_command("python scripts/audit_leakage.py")
    combined = (stdout + "\n" + stderr).strip()
    # Parse summary line: "FAIL:    0" should not trigger a failure.
    fail_match = __import__("re").search(r"FAIL:\s*(\d+)", combined)
    fail_count = int(fail_match.group(1)) if fail_match else 0
    status = "PASS" if code == 0 and fail_count == 0 else "FAIL"
    return {"gate": "no_leakage", "status": status, "output": combined, "evidence": "scripts/audit_leakage.py"}


def check_schema_and_duplicates() -> dict[str, Any]:
    try:
        sys.path.insert(0, str(ROOT))
        from predictors.feature_engineering import FEATURE_COLS, load_historical_results

        df = load_historical_results()
        duplicate_matches = df.duplicated(subset=["date", "home_team", "away_team"]).sum()
        issues = []
        if duplicate_matches > 0:
            issues.append(f"{duplicate_matches} duplicate matches")
        return {
            "gate": "schema_and_duplicates",
            "status": "PASS" if not issues else "WARNING",
            "feature_cols_count": len(FEATURE_COLS),
            "duplicate_matches": int(duplicate_matches),
            "output": "; ".join(issues) if issues else "No duplicate matches found",
            "evidence": "MondialXboost.Web/Data/historical_results.csv",
        }
    except Exception as exc:
        return {"gate": "schema_and_duplicates", "status": "FAIL", "output": str(exc), "evidence": ""}


def main() -> int:
    checks = [
        check_elo(),
        check_player_weights(),
        check_leakage(),
        check_schema_and_duplicates(),
    ]

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "project": "MondialXboost xBoost",
        "verdict": "BLOCK" if any(c["status"] == "FAIL" for c in checks) else "PASS",
        "checks": checks,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("Data Council Report")
    print("=" * 60)
    for c in checks:
        print(f"{c['gate']:<22} {c['status']:<6} {c.get('output', '')[:60]}")
    print("-" * 60)
    print(f"Verdict: {report['verdict']}")
    print(f"Report saved to {REPORT_PATH.relative_to(ROOT)}")

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
