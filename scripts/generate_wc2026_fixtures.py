#!/usr/bin/env python3
"""Generate a fixtures JSON for the 2026 World Cup group stage.

Usage:
    python scripts/generate_wc2026_fixtures.py
    python scripts/predict.py --fixtures data/wc2026_fixtures.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
GROUPS_CSV = ROOT / "MondialXboost.Web" / "Data" / "wc2026_groups.csv"
OUTPUT = ROOT / "data" / "wc2026_fixtures.json"

# Realistic 2026 World Cup group-stage matchday base dates.
# Groups A-D kick off 11-14 Jun (MD1), 17-20 Jun (MD2), 23-25 Jun (MD3).
# Groups E-H kick off 12-15 Jun (MD1), 17-20 Jun (MD2), 25-27 Jun (MD3).
# Groups I-L kick off 15-17 Jun (MD1), 21-23 Jun (MD2), 27-28 Jun (MD3).
MATCHDAY_DATES = {
    "A": ["2026-06-12", "2026-06-17", "2026-06-23"],
    "B": ["2026-06-13", "2026-06-18", "2026-06-24"],
    "C": ["2026-06-13", "2026-06-18", "2026-06-24"],
    "D": ["2026-06-14", "2026-06-19", "2026-06-25"],
    "E": ["2026-06-14", "2026-06-19", "2026-06-25"],
    "F": ["2026-06-15", "2026-06-20", "2026-06-26"],
    "G": ["2026-06-15", "2026-06-20", "2026-06-26"],
    "H": ["2026-06-16", "2026-06-21", "2026-06-27"],
    "I": ["2026-06-16", "2026-06-22", "2026-06-27"],
    "J": ["2026-06-17", "2026-06-22", "2026-06-28"],
    "K": ["2026-06-17", "2026-06-23", "2026-06-28"],
    "L": ["2026-06-17", "2026-06-23", "2026-06-28"],
}


def load_groups(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["group"], []).append(row["team"])
    return groups


def distribute_pairings(teams: list[str]) -> list[tuple[str, str]]:
    """Return the 6 round-robin pairings ordered by matchday.

    For 4 teams [0,1,2,3] we produce:
      MD1: (0,1), (2,3)
      MD2: (0,2), (1,3)
      MD3: (0,3), (1,2)
    """
    return [
        (teams[0], teams[1]),
        (teams[2], teams[3]),
        (teams[0], teams[2]),
        (teams[1], teams[3]),
        (teams[0], teams[3]),
        (teams[1], teams[2]),
    ]


def generate_fixtures(groups: dict[str, list[str]]) -> list[dict]:
    fixtures: list[dict] = []
    for group_name in sorted(groups):
        teams = groups[group_name]
        dates = MATCHDAY_DATES[group_name]
        pairings = distribute_pairings(teams)
        for idx, (home, away) in enumerate(pairings):
            matchday = idx // 2
            fixtures.append(
                {
                    "date": dates[matchday],
                    "home_team": home,
                    "away_team": away,
                    "neutral": True,
                    "group": group_name,
                }
            )
    return fixtures


def main() -> int:
    if not GROUPS_CSV.exists():
        print(f"[ERROR] No se encontró {GROUPS_CSV}", file=sys.stderr)
        return 1

    groups = load_groups(GROUPS_CSV)
    fixtures = generate_fixtures(groups)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2, ensure_ascii=False)

    print(f"Generados {len(fixtures)} partidos de fase de grupos en {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
