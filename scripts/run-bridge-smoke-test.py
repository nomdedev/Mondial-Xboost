#!/usr/bin/env python3
"""
run-bridge-smoke-test.py — Mondial-Xboost Bridge Smoke Test
============================================================

Levanta el bridge FastAPI en un puerto efímero, envía fixtures de prueba y
verifica que las respuestas sean coherentes.

Salida: `backtest/results/bridge_smoke.json`
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "backtest" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "bridge_smoke.json"


def find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_server(url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                return resp.status == 200
        except Exception:
            time.sleep(0.2)
    return False


def main() -> int:
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "predictors.api:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        health_url = f"{base_url}/health"
        if not wait_for_server(health_url, timeout=60):
            stdout, stderr = proc.communicate(timeout=5)
            result = {
                "status": "FAIL",
                "error": "Server did not become healthy",
                "stdout": stdout,
                "stderr": stderr,
            }
            OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return 1

        with urllib.request.urlopen(health_url, timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))

        if not health.get("model_loaded"):
            print("No model loaded; triggering /train ...")
            train_req = urllib.request.Request(
                f"{base_url}/train",
                data=b"",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(train_req, timeout=300) as resp:
                train_resp = json.loads(resp.read().decode("utf-8"))
                print(f"Train response: {train_resp.get('status')} — {train_resp.get('metrics', {})}")

            if not wait_for_server(health_url, timeout=60):
                stdout, stderr = proc.communicate(timeout=5)
                result = {
                    "status": "FAIL",
                    "error": "Server did not become healthy after /train",
                    "stdout": stdout,
                    "stderr": stderr,
                }
                OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
                return 1

        fixtures = [
            {"date": "2026-06-15", "home_team": "Argentina", "away_team": "Brazil", "neutral": True},
            {"date": "2026-06-16", "home_team": "Germany", "away_team": "France", "neutral": True},
            {"date": "2026-06-17", "home_team": "Spain", "away_team": "Italy", "neutral": True},
        ]

        req = urllib.request.Request(
            f"{base_url}/predict",
            data=json.dumps({"fixtures": fixtures}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        predictions = data.get("predictions", [])
        errors: list[str] = []

        if len(predictions) != len(fixtures):
            errors.append(f"Expected {len(fixtures)} predictions, got {len(predictions)}")

        for i, p in enumerate(predictions):
            probs = [p.get("prob_away_win", 0), p.get("prob_draw", 0), p.get("prob_home_win", 0)]
            total = sum(probs)
            if abs(total - 1.0) > 0.01:
                errors.append(f"Prediction {i}: probabilities sum to {total:.4f}")
            if p.get("top_pick") not in ("Home", "Draw", "Away"):
                errors.append(f"Prediction {i}: invalid top_pick {p.get('top_pick')}")
            if p.get("expected_home_goals", 0) < 0 or p.get("expected_away_goals", 0) < 0:
                errors.append(f"Prediction {i}: negative expected goals")

        result: dict[str, Any] = {
            "tested_at": datetime.now(UTC).isoformat(),
            "base_url": base_url,
            "fixtures_sent": len(fixtures),
            "predictions_received": len(predictions),
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "sample_prediction": predictions[0] if predictions else None,
        }

        OUTPUT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

        print("=" * 60)
        print("Bridge Smoke Test")
        print("=" * 60)
        print(f"Server:      {base_url}")
        print(f"Fixtures:    {result['fixtures_sent']}")
        print(f"Predictions: {result['predictions_received']}")
        print(f"Status:      {result['status']}")
        if errors:
            for e in errors:
                print(f"  ERROR: {e}")
        print(f"Report saved to {OUTPUT_PATH.relative_to(ROOT)}")

        return 0 if not errors else 1

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
