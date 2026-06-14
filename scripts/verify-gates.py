#!/usr/bin/env python3
"""
verify-gates.py — Mondial-Xboost Loop Engineering Gate Runner
==============================================================

Ejecuta los gates definidos en `.agents/skills/loop-engineering/gates.json`,
registra resultados en `.agents/logs/pipeline-state.json` y devuelve código
de salida 0 si todos los gates requeridos pasan.

Uso:
    python scripts/verify-gates.py
    python scripts/verify-gates.py --phase 6
    python scripts/verify-gates.py --gate secret_scan
    python scripts/verify-gates.py --skip dotnet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
GATES_PATH = ROOT / ".agents" / "skills" / "loop-engineering" / "gates.json"
STATE_PATH = ROOT / ".agents" / "logs" / "pipeline-state.json"
RESULTS_DIR = ROOT / "backtest" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Patterns considered sensitive. These are heuristic; they flag candidates for
# human review, not certainties.
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"][a-z0-9_-]{16,}['\"]"),
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
    re.compile(r"(?i)(secret|token)\s*[:=]\s*['\"][a-z0-9_-]{16,}['\"]"),
    re.compile(r"(?i)openrouter[_-]?api[_-]?key\s*[:=]\s*['\"][^'\"]+['\"]"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI-style key
    re.compile(r"[a-zA-Z0-9_-]{40}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}"),  # JWT-ish
]


def load_gates() -> dict[str, Any]:
    with open(GATES_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "project": "MondialXboost xBoost",
        "version": "2.0.0",
        "updatedAt": datetime.now(UTC).isoformat(),
        "activeFeatures": [],
        "completedFeatures": [],
        "blockedFeatures": [],
        "gates": [],
    }


def save_state(state: dict[str, Any]) -> None:
    state["updatedAt"] = datetime.now(UTC).isoformat()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def record_gate(state: dict[str, Any], gate_id: str, phase: int, status: str, output: str, evidence: str | None = None) -> None:
    state["gates"] = [g for g in state["gates"] if not (g["gateId"] == gate_id and g["phase"] == phase)]
    state["gates"].append({
        "featureId": "LOOP-ENGINEERING",
        "phase": phase,
        "gate": gate_id,
        "gateId": gate_id,
        "status": status,
        "verifiedBy": "verify-gates.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "evidence": evidence or "",
        "output": output[:2000] if output else "",
    })


def prepare_command(cmd: str) -> str:
    """Ensure the command uses the same Python interpreter as this script."""
    cmd = cmd.strip()
    if cmd.startswith("python "):
        cmd = f"{sys.executable} {cmd[7:]}"
    elif cmd == "python":
        cmd = sys.executable
    elif cmd.startswith("pytest "):
        cmd = f"{sys.executable} -m pytest {cmd[7:]}"
    elif cmd == "pytest":
        cmd = f"{sys.executable} -m pytest"
    elif cmd.startswith("ruff "):
        cmd = f"{sys.executable} -m ruff {cmd[5:]}"
    elif cmd == "ruff":
        cmd = f"{sys.executable} -m ruff"
    return cmd


def run_command(cmd: str, cwd: Path = ROOT, timeout: int = 300, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    """Run a shell command and return (exit_code, stdout, stderr)."""
    cmd = prepare_command(cmd)
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(ROOT)
    if env:
        run_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out after {timeout}s"
    except FileNotFoundError as exc:
        return 127, "", f"Command not found: {exc}"


def command_available(name: str) -> bool:
    code, _, _ = run_command(f"which {name}" if os.name != "nt" else f"where {name}")
    return code == 0


def check_secret_scan() -> tuple[str, str, str]:
    """Scan repository for likely secrets. Returns (status, evidence, output)."""
    hits: list[str] = []
    scanned = 0
    # Scan source files (skip binaries, venv/node_modules, tests with fake keys)
    skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", "bin", "obj", "models", "data", "tests", "scripts/archive"}
    skip_filename_patterns = ["test", "tests", "fixture", "mock"]
    extensions = {".py", ".cs", ".cshtml", ".razor", ".js", ".ts", ".md", ".json", ".yml", ".yaml", ".txt"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        lower_name = path.name.lower()
        if any(p in lower_name for p in skip_filename_patterns):
            continue
        if path.suffix.lower() not in extensions:
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    hits.append(f"{path.relative_to(ROOT)}:{line_no}: {pattern.pattern[:40]}...")
                    break

    output = f"Scanned {scanned} files. {len(hits)} candidate secret matches."
    if hits:
        output += "\n" + "\n".join(hits[:20])
        return "FAIL", "secret scan output", output
    return "PASS", "secret scan output", output


def evaluate_gate(gate: dict[str, Any], phase: int, skip_patterns: list[str]) -> dict[str, Any]:
    gate_id = gate["id"]
    name = gate["name"]
    required = gate.get("required", True)

    if any(sp in gate_id or sp in gate.get("command", "") for sp in skip_patterns):
        return {"id": gate_id, "name": name, "status": "SKIP", "output": "Skipped by user", "required": required}

    # Secret scan is implemented inline
    if gate_id == "secret_scan":
        status, evidence, output = check_secret_scan()
        return {"id": gate_id, "name": name, "status": status, "output": output, "evidence": evidence, "required": required}

    cmd = gate.get("command", "")
    if not cmd:
        return {"id": gate_id, "name": name, "status": "SKIP", "output": "No command defined", "required": required}

    # If the command depends on dotnet and dotnet is unavailable, mark SKIP with warning
    if "dotnet" in cmd and not command_available("dotnet"):
        return {
            "id": gate_id,
            "name": name,
            "status": "SKIP",
            "output": "dotnet SDK not available in this environment; run in environment with .NET 9 SDK",
            "evidence": "",
            "required": required,
        }

    code, stdout, stderr = run_command(cmd, timeout=gate.get("timeout", 300))
    combined = (stdout + "\n" + stderr).strip()

    status = "PASS" if code == 0 else "FAIL"

    # For Elo gate, parse the JSON report to apply thresholds
    if gate_id == "elo_calibration" and code == 0:
        try:
            report_path = ROOT / "backtest" / "results" / "elo_comparison.json"
            if report_path.exists():
                with open(report_path, encoding="utf-8") as f:
                    report = json.load(f)
                top100_diff = report.get("top100_teams_stats", {}).get("mean_absolute_difference", 9999)
                corr = report.get("correlation", 0.0)
                config = load_gates()
                thresholds = config.get("thresholds", {})
                if top100_diff > thresholds.get("elo_top100_mean_abs_diff", 100):
                    status = "FAIL"
                    combined += f"\nFAIL: top100 mean abs diff {top100_diff:.2f} > {thresholds['elo_top100_mean_abs_diff']}"
                if corr < thresholds.get("elo_top100_correlation", 0.90):
                    status = "FAIL"
                    combined += f"\nFAIL: top100 correlation {corr:.3f} < {thresholds['elo_top100_correlation']}"
        except Exception as exc:
            status = "FAIL"
            combined += f"\nCould not parse elo_comparison.json: {exc}"

    return {
        "id": gate_id,
        "name": name,
        "status": status,
        "output": combined,
        "evidence": gate.get("evidence", ""),
        "required": required,
        "exit_code": code,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mondial-Xboost loop engineering gate runner")
    parser.add_argument("--phase", type=int, help="Run only gates for this phase")
    parser.add_argument("--gate", type=str, help="Run only this gate")
    parser.add_argument("--skip", type=str, default="", help="Comma-separated list of substrings; matching gates are skipped")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON only")
    args = parser.parse_args()

    skip_patterns = [s.strip() for s in args.skip.split(",") if s.strip()]

    try:
        config = load_gates()
    except Exception as exc:
        print(f"ERROR: Could not load gates config: {exc}", file=sys.stderr)
        return 2

    state = load_state()
    results: list[dict[str, Any]] = []

    for phase_def in config.get("phases", []):
        phase = phase_def["phase"]
        if args.phase and phase != args.phase:
            continue
        for gate in phase_def.get("gates", []):
            if args.gate and gate["id"] != args.gate:
                continue
            result = evaluate_gate(gate, phase, skip_patterns)
            results.append(result)
            record_gate(state, gate["id"], phase, result["status"], result.get("output", ""), result.get("evidence"))

    save_state(state)

    required_failures = [r for r in results if r["status"] == "FAIL" and r.get("required", True)]
    warnings = [r for r in results if r["status"] == "WARNING"]
    skipped = [r for r in results if r["status"] == "SKIP"]

    if args.json:
        print(json.dumps({"results": results, "summary": {
            "total": len(results),
            "pass": sum(1 for r in results if r["status"] == "PASS"),
            "fail": sum(1 for r in results if r["status"] == "FAIL"),
            "warning": len(warnings),
            "skip": len(skipped),
        }}, indent=2))
    else:
        print("=" * 60)
        print("Mondial-Xboost Loop Engineering Gate Report")
        print("=" * 60)
        for r in results:
            req_flag = "[R]" if r.get("required", True) else "[O]"
            print(f"{req_flag} Phase {r.get('phase', '-'):<3} {r['id']:<22} {r['status']:<6} {r['name']}")
            if r["status"] in ("FAIL", "WARNING") and r.get("output"):
                for line in r["output"].splitlines()[:5]:
                    print(f"     {line}")
        print("-" * 60)
        print(f"Total: {len(results)} | PASS: {sum(1 for r in results if r['status'] == 'PASS')} | "
              f"FAIL: {sum(1 for r in results if r['status'] == 'FAIL')} | "
              f"WARNING: {len(warnings)} | SKIP: {len(skipped)}")
        if required_failures:
            print("\nRequired gates FAILED — pipeline blocked.")

    return 1 if required_failures else 0


if __name__ == "__main__":
    sys.exit(main())
