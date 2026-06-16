"""Build a static snapshot of the full 2026 World Cup simulation.

Run before deploying to Vercel:
    python scripts/build_wc2026_tournament.py

The resulting data/wc2026_tournament.json is served instantly by the dashboard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.wc2026_tournament import simulate_tournament  # noqa: E402

OUTPUT_PATH = ROOT / "data" / "wc2026_tournament.json"


def main() -> None:
    print("Simulating full 2026 World Cup tournament...")
    result = simulate_tournament("xgboost_football")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved tournament snapshot to {OUTPUT_PATH}")
    print(f"Predicted champion: {result['champion']}")


if __name__ == "__main__":
    main()
