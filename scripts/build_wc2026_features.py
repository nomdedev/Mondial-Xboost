"""Pre-compute historical features used by wc2026_engine.py for fast inference.

Run locally before deploying to Vercel:
    python scripts/build_wc2026_features.py

This writes three compressed CSVs to data/prepared/ that are used in production
instead of recomputing Elo, rolling stats and H2H on every cold start.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from predictors.feature_engineering import (  # noqa: E402
    _build_team_history,
    _compute_h2h,
    _compute_team_rolling,
    compute_elo_ratings,
    load_historical_results,
)

PREPARED_DIR = ROOT / "data" / "prepared"


def main() -> None:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading historical results...")
    historical = load_historical_results()
    historical["date"] = pd.to_datetime(historical["date"])

    print("Computing Elo ratings...")
    historical = compute_elo_ratings(historical)

    print("Building team history...")
    long = _build_team_history(historical)

    print("Computing rolling team stats...")
    long = _compute_team_rolling(long)

    print("Computing H2H stats...")
    historical_h2h = _compute_h2h(historical)

    # Persist only the columns required by wc2026_engine.build_features_for_fixtures.
    historical_cols = [
        "date", "home_team", "away_team",
        "home_elo_before", "away_elo_before",
        "home_elo_recent", "away_elo_recent",
    ]
    historical[historical_cols].to_csv(
        PREPARED_DIR / "historical.csv.gz", index=False, compression="gzip"
    )

    long_cols = [
        "date", "team", "points_avg_5", "points_avg_10",
        "goals_scored_avg_10", "goals_conceded_avg_10",
        "win_rate_10", "draw_rate_10", "loss_rate_10", "matches_played",
    ]
    # Keep only rows where the team has at least one previous match so the file is smaller.
    long[long["matches_played"] > 0][long_cols].to_csv(
        PREPARED_DIR / "long.csv.gz", index=False, compression="gzip"
    )

    h2h_cols = [
        "date", "home_team", "away_team",
        "h2h_last_result", "h2h_goals_avg", "h2h_wins_diff", "h2h_years_since",
    ]
    historical_h2h[h2h_cols].to_csv(
        PREPARED_DIR / "h2h.csv.gz", index=False, compression="gzip"
    )

    print(f"Saved prepared features to {PREPARED_DIR}")
    for path in sorted(PREPARED_DIR.glob("*.csv.gz")):
        print(f"  {path.name}: {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
