"""
Make predictions from the terminal.

Usage:
    python scripts/predict.py --home Brazil --away Morocco --date 2026-06-20
    python scripts/predict.py --fixtures fixtures.json
    python scripts/predict.py --last-n 5
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Force UTF-8 on Windows terminals so accented chars render without crashing.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from predictors.blended_predictor import BlendedFootballPredictor
from predictors.cold_start_model import ColdStartPredictor
from predictors.feature_engineering import (
    _build_team_history,
    _compute_h2h,
    _compute_team_rolling,
    _recent_matches_count,
    compute_elo_ratings,
    load_historical_results,
)
from predictors.random_forest_engine import MODELS_DIR as RF_MODELS_DIR
from predictors.random_forest_engine import RandomForestFootballPredictor
from predictors.xgboost_engine import MODELS_DIR, XGBoostFootballPredictor

C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "red": "\033[91m",
    "dim": "\033[2m",
}


def _get_latest_team_stats(long: pd.DataFrame, team: str, date: pd.Timestamp) -> dict:
    team_long = long[(long["team"] == team) & (long["date"] < date)].sort_values("date")
    if team_long.empty:
        return {
            "points_avg_5": 0.5, "points_avg_10": 0.5,
            "goals_scored_avg_10": 1.3, "goals_conceded_avg_10": 1.3,
            "win_rate_10": 0.0, "draw_rate_10": 0.0, "loss_rate_10": 0.0,
            "matches_played": 0,
        }
    last = team_long.iloc[-1]
    return {
        "points_avg_5": last.get("points_avg_5", 0.5),
        "points_avg_10": last.get("points_avg_10", 0.5),
        "goals_scored_avg_10": last.get("goals_scored_avg_10", 1.3),
        "goals_conceded_avg_10": last.get("goals_conceded_avg_10", 1.3),
        "win_rate_10": last.get("win_rate_10", 0.0),
        "draw_rate_10": last.get("draw_rate_10", 0.0),
        "loss_rate_10": last.get("loss_rate_10", 0.0),
        "matches_played": last.get("matches_played", 0),
    }


def build_features_for_prediction(historical: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Build FEATURE_COLS for fixtures not present in historical data."""
    historical = historical.copy()
    fixtures = fixtures.copy()
    historical["date"] = pd.to_datetime(historical["date"])
    fixtures["date"] = pd.to_datetime(fixtures["date"])

    # Elo ratings from historical data
    historical = compute_elo_ratings(historical)

    elo_rows = []
    for _, fx in fixtures.iterrows():
        date = fx["date"]
        home, away = fx["home_team"], fx["away_team"]
        before = historical[historical["date"] < date]

        def last_elo(team, col):
            home_elos = before[before["home_team"] == team][["date", col]].rename(
                columns={col: "elo"}
            )
            away_elos = before[before["away_team"] == team][["date", col]].rename(
                columns={col: "elo"}
            )
            all_elos = pd.concat([home_elos, away_elos]).sort_values("date")
            return float(all_elos.iloc[-1]["elo"]) if not all_elos.empty else 1500.0

        elo_rows.append({
            "date": date, "home_team": home, "away_team": away,
            "home_elo_before": last_elo(home, "home_elo_before"),
            "away_elo_before": last_elo(away, "away_elo_before"),
            "home_elo_recent": last_elo(home, "home_elo_recent"),
            "away_elo_recent": last_elo(away, "away_elo_recent"),
        })

    elo_df = pd.DataFrame(elo_rows)
    fixtures = fixtures.merge(elo_df, on=["date", "home_team", "away_team"], how="left")
    fixtures["elo_diff"] = fixtures["home_elo_before"] - fixtures["away_elo_before"]
    fixtures["elo_diff_recent"] = fixtures["home_elo_recent"] - fixtures["away_elo_recent"]

    # Recent match counts for cold-start detection
    recent_counts = _recent_matches_count(historical, fixtures, years=8.0)
    fixtures = fixtures.merge(
        recent_counts,
        on=["date", "home_team", "away_team"],
        how="left",
    )

    # Team rolling stats
    long = _build_team_history(historical)
    long = _compute_team_rolling(long)

    stat_rows = []
    for _, fx in fixtures.iterrows():
        date = fx["date"]
        home, away = fx["home_team"], fx["away_team"]
        home_stats = _get_latest_team_stats(long, home, date)
        away_stats = _get_latest_team_stats(long, away, date)
        row = {"date": date, "home_team": home, "away_team": away}
        for k, v in home_stats.items():
            row[f"home_{k}"] = v
        for k, v in away_stats.items():
            row[f"away_{k}"] = v
        stat_rows.append(row)

    stats_df = pd.DataFrame(stat_rows)
    fixtures = fixtures.merge(stats_df, on=["date", "home_team", "away_team"], how="left")

    # H2H stats
    historical_h2h = _compute_h2h(historical)
    h2h_rows = []
    for _, fx in fixtures.iterrows():
        date = fx["date"]
        home, away = fx["home_team"], fx["away_team"]
        before = historical_h2h[historical_h2h["date"] < date]
        mask = before.apply(
            lambda r: sorted([r["home_team"], r["away_team"]]) == sorted([home, away]), axis=1
        )
        pair = before[mask]

        if not pair.empty:
            last = pair.iloc[-1]
            h2h_last_result = float(last["h2h_last_result"])
            h2h_goals_avg = float(last["h2h_goals_avg"])
            h2h_wins_diff = float(last["h2h_wins_diff"])
            h2h_years_since = (date - last["date"]).days / 365.25
        else:
            h2h_last_result, h2h_goals_avg, h2h_wins_diff, h2h_years_since = 0.0, 1.3, 0.0, 20.0

        h2h_rows.append({
            "date": date, "home_team": home, "away_team": away,
            "h2h_last_result": h2h_last_result,
            "h2h_goals_avg": h2h_goals_avg,
            "h2h_wins_diff": h2h_wins_diff,
            "h2h_years_since": h2h_years_since,
        })

    h2h_df = pd.DataFrame(h2h_rows)
    fixtures = fixtures.merge(h2h_df, on=["date", "home_team", "away_team"], how="left")

    # Fill defaults
    fixtures["elo_diff"] = fixtures["elo_diff"].fillna(0.0)
    fixtures["elo_diff_recent"] = fixtures["elo_diff_recent"].fillna(0.0)
    for col in ["home_points_avg_5", "home_points_avg_10", "away_points_avg_5", "away_points_avg_10"]:
        fixtures[col] = fixtures[col].fillna(0.5)
    for col in ["home_win_rate_10", "home_draw_rate_10", "home_loss_rate_10",
                "away_win_rate_10", "away_draw_rate_10", "away_loss_rate_10", "h2h_last_result"]:
        fixtures[col] = fixtures[col].fillna(0.0)
    for col in ["home_goals_scored_avg_10", "home_goals_conceded_avg_10",
                "away_goals_scored_avg_10", "away_goals_conceded_avg_10", "h2h_goals_avg"]:
        fixtures[col] = fixtures[col].fillna(1.3)
    fixtures["home_matches_played"] = fixtures["home_matches_played"].fillna(0).astype(int)
    fixtures["away_matches_played"] = fixtures["away_matches_played"].fillna(0).astype(int)
    fixtures["home_recent_matches"] = fixtures["home_recent_matches"].fillna(0).astype(int)
    fixtures["away_recent_matches"] = fixtures["away_recent_matches"].fillna(0).astype(int)
    fixtures["h2h_wins_diff"] = fixtures["h2h_wins_diff"].fillna(0.0)
    fixtures["h2h_years_since"] = fixtures["h2h_years_since"].fillna(20.0)

    return fixtures


def load_model(name: str = "xgboost_football", blend: bool = False, cold_start_only: bool = False, engine: str = "xgboost"):
    if cold_start_only:
        path = MODELS_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el cold-start model '{name}'. Entrenalo primero.")
        return ColdStartPredictor.load(name)
    if blend:
        canonical_name = "xgboost_football"
        cold_name = name if name != "xgboost_football" else "cold_start"
        return BlendedFootballPredictor.load(canonical_name=canonical_name, cold_start_name=cold_name)
    if engine == "random_forest":
        path = RF_MODELS_DIR / f"{name}_meta.json"
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el modelo RF '{name}'. Entrenalo primero con: python scripts/train.py --engine random_forest")
        return RandomForestFootballPredictor.load(name)
    path = MODELS_DIR / f"{name}_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo '{name}'. Entrenalo primero con: python scripts/train.py")
    return XGBoostFootballPredictor.load(name)


def print_predictions(predictions: list[dict]) -> None:
    print(f"\n{C['bold']}{'Fecha':12s} {'Local':16s} {'Visitante':16s} {'1X2':10s} {'Local%':>8s} {'Empate%':>8s} {'Visita%':>8s} {'xG_L':>5s} {'xG_V':>5s} {'Pick':>6s}{C['reset']}")
    print("-" * 100)
    for p in predictions:
        date = p["date"][:10] if isinstance(p["date"], str) else str(p["date"])[:10]
        neutral = "Neutral" if p.get("neutral") else "Localía"
        home_pct = p["prob_home_win"] * 100
        draw_pct = p["prob_draw"] * 100
        away_pct = p["prob_away_win"] * 100
        pick = p["top_pick"]
        pick_color = "green" if pick == "Home" else "yellow" if pick == "Draw" else "red"
        print(
            f"{date:12s} {p['home_team']:16s} {p['away_team']:16s} {neutral:10s} "
            f"{home_pct:7.1f}% {draw_pct:7.1f}% {away_pct:7.1f}% "
            f"{p['expected_home_goals']:5.2f} {p['expected_away_goals']:5.2f} "
            f"{C[pick_color]}{pick:>6s}{C['reset']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict football match outcomes")
    parser.add_argument("--home", type=str, help="Home team name")
    parser.add_argument("--away", type=str, help="Away team name")
    parser.add_argument("--date", type=str, default=pd.Timestamp.now().strftime("%Y-%m-%d"), help="Fixture date")
    parser.add_argument("--fixtures", type=str, help="JSON file with fixtures list")
    parser.add_argument("--last-n", type=int, help="Predict last N matches from historical data")
    parser.add_argument("--model", type=str, default="xgboost_football", help="Model name to load")
    parser.add_argument("--blend", action="store_true", help="Usar blended predictor (canónico + cold-start)")
    parser.add_argument("--cold-start-only", action="store_true", help="Usar solo el cold-start model")
    args = parser.parse_args()

    try:
        print(f"{C['dim']}Cargando modelo '{args.model}'...{C['reset']}")
        predictor = load_model(args.model, blend=args.blend, cold_start_only=args.cold_start_only, engine=args.engine)

        print(f"{C['dim']}Cargando datos históricos...{C['reset']}")
        historical = load_historical_results()

        if args.fixtures:
            with open(args.fixtures, encoding="utf-8") as f:
                fixtures_data = json.load(f)
            fixtures = pd.DataFrame(fixtures_data)
        elif args.last_n:
            fixtures = historical[["date", "home_team", "away_team", "neutral"]].tail(args.last_n)
        elif args.home and args.away:
            fixtures = pd.DataFrame([{
                "date": args.date,
                "home_team": args.home,
                "away_team": args.away,
                "neutral": False,
            }])
        else:
            print(f"{C['red']}Error: especificá --home/--away, --fixtures o --last-n{C['reset']}")
            return 1

        features = build_features_for_prediction(historical, fixtures)
        predictions = predictor.predict(features)

        print(f"\n{C['bold']}{C['cyan']}Predicciones — {args.engine}/{args.model}{C['reset']}")
        print_predictions(predictions)
        return 0

    except FileNotFoundError as exc:
        print(f"{C['red']}[ERROR] {exc}{C['reset']}")
        return 1
    except Exception as exc:
        print(f"{C['red']}[ERROR] {exc}{C['reset']}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
