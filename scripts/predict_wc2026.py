#!/usr/bin/env python3
"""Predict every 2026 World Cup group-stage match with the trained model.

Usage:
    python scripts/predict_wc2026.py [--model xgboost_football] [--blend]

Outputs:
    data/wc2026_predictions.json   # machine-readable predictions
    data/wc2026_predictions.md     # markdown table
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
FIXTURES_JSON = ROOT / "data" / "wc2026_fixtures.json"
PREDICTIONS_JSON = ROOT / "data" / "wc2026_predictions.json"
PREDICTIONS_MD = ROOT / "data" / "wc2026_predictions.md"


def generate_fixtures() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_wc2026_fixtures.py")],
        cwd=ROOT,
        check=True,
    )


def run_predictions(model: str, blend: bool) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "predict.py"),
        "--fixtures",
        str(FIXTURES_JSON),
        "--model",
        model,
        "--output",
        str(PREDICTIONS_JSON),
    ]
    if blend:
        cmd.append("--blend")

    subprocess.run(cmd, cwd=ROOT, check=True)


def render_markdown(predictions: list[dict]) -> str:
    lines = [
        "# Predicciones Mundial 2026 — Fase de Grupos",
        "",
        f"_Modelo: xgboost_football | Total partidos: {len(predictions)}_",
        "",
        "| Fecha | Grupo | Local | Visitante | H | D | A | xG L | xG V | Pick |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for p in predictions:
        group = p.get("group", "")
        lines.append(
            f"| {p['date']} | {group} | {p['home_team']} | {p['away_team']} | "
            f"{p['prob_home_win']*100:.1f}% | {p['prob_draw']*100:.1f}% | {p['prob_away_win']*100:.1f}% | "
            f"{p['expected_home_goals']:.2f} | {p['expected_away_goals']:.2f} | {p['top_pick']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict all WC 2026 group-stage matches")
    parser.add_argument("--model", default="xgboost_football", help="Model name to load")
    parser.add_argument("--blend", action="store_true", help="Use blended predictor")
    args = parser.parse_args()

    generate_fixtures()
    run_predictions(args.model, args.blend)

    with PREDICTIONS_JSON.open(encoding="utf-8") as f:
        predictions = json.load(f)

    markdown = render_markdown(predictions)
    with PREDICTIONS_MD.open("w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Guardado Markdown: {PREDICTIONS_MD}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
