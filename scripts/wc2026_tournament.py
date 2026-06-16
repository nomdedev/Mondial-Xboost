"""Full 2026 World Cup simulation: group stage + knockout bracket to champion.

This is a simplified simulation intended for the dashboard:
- Group stage uses the canonical model predictions for the 72 group fixtures.
- Standings are computed from predicted results (3 pts win, 1 draw).
- The top two from each group plus the eight best third-placed teams advance.
- Knockout bracket is seeded by group performance (1 vs 32, 2 vs 31, ...).
- Each knockout match is predicted with the canonical model and a winner is
  advanced until a champion is crowned.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).parent.parent

from scripts.wc2026_engine import load_fixtures, predict_fixture_list  # noqa: E402


def _result_from_prediction(pred: dict[str, Any]) -> tuple[str, int, int]:
    """Return (outcome, home_points, away_points) from a prediction dict."""
    top = pred.get("top_pick")
    if top == "Home":
        return "home", 3, 0
    if top == "Away":
        return "away", 0, 3
    return "draw", 1, 1


def _team_seed_key(team: dict[str, Any]) -> tuple[float, float, float]:
    return (-team["points"], -team["goal_diff"], -team["goals_for"])


def _group_standings(fixtures: list[dict], predictions: list[dict]) -> dict[str, list[dict]]:
    """Compute group tables from predicted group-stage fixtures."""
    pred_index = {
        (p["home_team"], p["away_team"], p["date"]): p
        for p in predictions
    }

    tables: dict[str, dict[str, dict]] = {}
    for fx in fixtures:
        group = fx.get("group")
        if not group:
            continue
        pred = pred_index.get((fx["home_team"], fx["away_team"], fx["date"]))
        if pred is None:
            continue

        home = fx["home_team"]
        away = fx["away_team"]
        hg = float(pred.get("expected_home_goals", 0) or 0)
        ag = float(pred.get("expected_away_goals", 0) or 0)
        outcome, hp, ap = _result_from_prediction(pred)

        for team, pts, gf, ga in ((home, hp, hg, ag), (away, ap, ag, hg)):
            tables.setdefault(group, {}).setdefault(
                team,
                {
                    "team": team,
                    "played": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "points": 0,
                    "goals_for": 0.0,
                    "goals_against": 0.0,
                    "goal_diff": 0.0,
                },
            )
            rec = tables[group][team]
            rec["played"] += 1
            rec["points"] += pts
            rec["goals_for"] += gf
            rec["goals_against"] += ga
            rec["goal_diff"] = rec["goals_for"] - rec["goals_against"]
            if outcome == "home" and team == home:
                rec["wins"] += 1
            elif outcome == "away" and team == away:
                rec["wins"] += 1
            elif outcome == "draw":
                rec["draws"] += 1
            else:
                rec["losses"] += 1

    sorted_tables = {}
    for group, teams in tables.items():
        sorted_tables[group] = sorted(
            teams.values(),
            key=lambda t: (-t["points"], -t["goal_diff"], -t["goals_for"]),
        )
    return sorted_tables


def _qualified_teams(standings: dict[str, list[dict]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return winners, runners-up and best third-placed teams."""
    group_order = sorted(standings)
    winners = [standings[g][0] for g in group_order if len(standings[g]) >= 1]
    runners = [standings[g][1] for g in group_order if len(standings[g]) >= 2]
    thirds = [standings[g][2] for g in group_order if len(standings[g]) >= 3]
    best_thirds = sorted(thirds, key=_team_seed_key)[:8]
    return winners, runners, best_thirds


def _knockout_bracket(qualified: list[dict]) -> list[dict]:
    """Build a 32-team bracket seeded by group-stage performance."""
    teams = sorted(qualified, key=_team_seed_key)
    bracket = []
    n = len(teams)
    round_date = datetime(2026, 7, 1)
    for i in range(n // 2):
        high, low = teams[i], teams[n - 1 - i]
        bracket.append({
            "home_team": high["team"],
            "away_team": low["team"],
            "date": round_date.strftime("%Y-%m-%d"),
            "neutral": True,
            "round": "round_of_32",
            "match": f"R32-{i + 1}",
            "home_seed": i + 1,
            "away_seed": n - i,
        })
    return bracket


def _winner_from_prediction(pred: dict[str, Any]) -> str:
    top = pred.get("top_pick")
    if top == "Home":
        return pred["home_team"]
    if top == "Away":
        return pred["away_team"]
    # Draw: advance the team with higher win probability.
    if pred.get("prob_home_win", 0) >= pred.get("prob_away_win", 0):
        return pred["home_team"]
    return pred["away_team"]


def _simulate_knockout_round(fixtures: list[dict], model_name: str) -> tuple[list[dict], list[dict]]:
    """Predict a knockout round and return winners plus detailed results."""
    predictions = predict_fixture_list(fixtures, model_name)
    results = []
    winners = []
    for fx, pred in zip(fixtures, predictions):
        winner = _winner_from_prediction(pred)
        winners.append({
            "team": winner,
            "group_stage": {"points": 0, "goal_diff": 0, "goals_for": 0},
        })
        results.append({
            **fx,
            "prediction": pred,
            "winner": winner,
        })
    return winners, results


def _build_next_round(winners: list[dict], round_name: str, base_date: datetime) -> list[dict]:
    """Pair winners for the next round (1 vs 2, 3 vs 4, ...)."""
    fixtures = []
    for i in range(0, len(winners), 2):
        if i + 1 < len(winners):
            fixtures.append({
                "home_team": winners[i]["team"],
                "away_team": winners[i + 1]["team"],
                "date": base_date.strftime("%Y-%m-%d"),
                "neutral": True,
                "round": round_name,
                "match": f"{round_name.upper()}-{i // 2 + 1}",
            })
    return fixtures


def simulate_tournament(model_name: str = "xgboost_football") -> dict[str, Any]:
    """Simulate the full 2026 World Cup and return a structured result."""
    fixtures = load_fixtures()
    if not fixtures:
        raise RuntimeError("No WC 2026 fixtures available")

    # Group stage
    group_predictions = predict_fixture_list(fixtures, model_name)
    standings = _group_standings(fixtures, group_predictions)

    # Attach group-stage stats to qualified teams so seeding is meaningful.
    winners, runners, best_thirds = _qualified_teams(standings)
    qualified = []
    for team in winners + runners + best_thirds:
        qualified.append({
            "team": team["team"],
            "points": team["points"],
            "goal_diff": team["goal_diff"],
            "goals_for": team["goals_for"],
        })

    # Knockout
    bracket = _knockout_bracket(qualified)
    rounds = {
        "round_of_32": [],
        "round_of_16": [],
        "quarter_finals": [],
        "semi_finals": [],
        "final": [],
        "third_place": [],
    }

    # Round of 32
    r32_winners, r32_results = _simulate_knockout_round(bracket, model_name)
    rounds["round_of_32"] = r32_results

    # Round of 16
    r16_fixtures = _build_next_round(r32_winners, "round_of_16", datetime(2026, 7, 4))
    r16_winners, r16_results = _simulate_knockout_round(r16_fixtures, model_name)
    rounds["round_of_16"] = r16_results

    # Quarter finals
    qf_fixtures = _build_next_round(r16_winners, "quarter_finals", datetime(2026, 7, 8))
    qf_winners, qf_results = _simulate_knockout_round(qf_fixtures, model_name)
    rounds["quarter_finals"] = qf_results

    # Semi finals
    sf_fixtures = _build_next_round(qf_winners, "semi_finals", datetime(2026, 7, 12))
    sf_winners, sf_results = _simulate_knockout_round(sf_fixtures, model_name)
    rounds["semi_finals"] = sf_results
    sf_losers = []
    for fx, pred in zip(sf_fixtures, sf_results):
        loser = fx["home_team"] if fx["home_team"] != pred["winner"] else fx["away_team"]
        sf_losers.append({"team": loser})

    # Third place
    third_fixture = [{
        "home_team": sf_losers[0]["team"],
        "away_team": sf_losers[1]["team"],
        "date": (datetime(2026, 7, 15)).strftime("%Y-%m-%d"),
        "neutral": True,
        "round": "third_place",
        "match": "3RD-1",
    }]
    _, third_results = _simulate_knockout_round(third_fixture, model_name)
    rounds["third_place"] = third_results

    # Final
    final_fixture = [{
        "home_team": sf_winners[0]["team"],
        "away_team": sf_winners[1]["team"],
        "date": (datetime(2026, 7, 16)).strftime("%Y-%m-%d"),
        "neutral": True,
        "round": "final",
        "match": "FINAL-1",
    }]
    _, final_results = _simulate_knockout_round(final_fixture, model_name)
    rounds["final"] = final_results

    champion = final_results[0]["winner"]

    return {
        "champion": champion,
        "group_stage": {
            "fixtures": group_predictions,
            "standings": standings,
        },
        "knockout": rounds,
    }


if __name__ == "__main__":
    import json
    result = simulate_tournament()
    print(json.dumps(result, indent=2, default=str))
