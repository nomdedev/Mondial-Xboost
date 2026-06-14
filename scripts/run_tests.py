"""
Run tests from the terminal with a friendly summary.

Usage:
    python scripts/run_tests.py
    python scripts/run_tests.py --fast
    python scripts/run_tests.py --gate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

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


def run_command(cmd: list[str], description: str) -> tuple[int, str, str]:
    print(f"\n{C['dim']}Ejecutando: {description}...{C['reset']}")
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def print_summary(checks: list[dict]) -> None:
    print(f"\n{C['bold']}{'=' * 70}{C['reset']}")
    print(f"{C['bold']}RESUMEN DE TESTS{C['reset']}")
    print(f"{C['bold']}{'=' * 70}{C['reset']}")
    for check in checks:
        status = check["status"]
        color = "green" if status == "PASS" else "red" if status == "FAIL" else "yellow"
        icon = "[OK]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[SKIP]"
        print(f"  {icon} {check['name']:40s} {C[color]}{status}{C['reset']} ({check['time']:.1f}s)")
    total = len(checks)
    passed = sum(1 for c in checks if c["status"] == "PASS")
    failed = sum(1 for c in checks if c["status"] == "FAIL")
    print(f"\n  Total: {total} | {C['green']}PASS: {passed}{C['reset']} | {C['red']}FAIL: {failed}{C['reset']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run project tests and gates")
    parser.add_argument("--fast", action="store_true", help="Skip slow gates (backtest, bridge)")
    parser.add_argument("--gate", action="store_true", help="Also run verify_gates")
    args = parser.parse_args()

    checks = []

    # 1. Python lint
    code, out, err, elapsed = run_command(
        [sys.executable, "-m", "ruff", "check", "scripts/", "predictors/", "backtest/", "tests/"],
        "Python lint (ruff)"
    )
    checks.append({"name": "Python lint", "status": "PASS" if code == 0 else "FAIL", "time": elapsed})
    if code != 0:
        print(err or out)

    # 2. Python tests
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    code, out, err, elapsed = run_command(cmd, "Python unit tests")
    status = "PASS" if code == 0 else "FAIL"
    checks.append({"name": "Python unit tests", "status": status, "time": elapsed})
    if status == "PASS":
        # extract count from output
        for line in (out + err).splitlines():
            if "passed" in line:
                print(f"  {C['green']}{line.strip()}{C['reset']}")
                break
    else:
        print(out)
        print(err)

    # 3. Data council (fast)
    code, out, err, elapsed = run_command(
        [sys.executable, "scripts/run_data_council.py"],
        "Data council"
    )
    checks.append({"name": "Data council", "status": "PASS" if code == 0 else "FAIL", "time": elapsed})

    # 4. Backtest gate (slow)
    if not args.fast:
        code, out, err, elapsed = run_command(
            [sys.executable, "scripts/run_backtest_gate.py"],
            "World Cup backtest gate"
        )
        checks.append({"name": "Backtest gate", "status": "PASS" if code == 0 else "FAIL", "time": elapsed})
        if code == 0:
            for line in out.splitlines():
                if "Verdict" in line:
                    print(f"  {C['green']}{line.strip()}{C['reset']}")
    else:
        checks.append({"name": "Backtest gate", "status": "SKIP", "time": 0})

    # 5. Bridge smoke test (slow)
    if not args.fast:
        code, out, err, elapsed = run_command(
            [sys.executable, "scripts/run_bridge_smoke_test.py"],
            "Bridge smoke test"
        )
        checks.append({"name": "Bridge smoke test", "status": "PASS" if code == 0 else "FAIL", "time": elapsed})
    else:
        checks.append({"name": "Bridge smoke test", "status": "SKIP", "time": 0})

    # 6. Full verify_gates
    if args.gate:
        code, out, err, elapsed = run_command(
            [sys.executable, "scripts/verify_gates.py", "--skip", "dotnet_build,dotnet_format,dotnet_test"],
            "verify_gates"
        )
        checks.append({"name": "verify_gates", "status": "PASS" if code == 0 else "FAIL", "time": elapsed})

    print_summary(checks)

    if any(c["status"] == "FAIL" for c in checks):
        print(f"\n{C['red']}Algunos checks fallaron. Revisá la salida arriba.{C['reset']}")
        return 1

    print(f"\n{C['green']}[OK] Todos los checks pasaron.{C['reset']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
