"""
Player-level weighting algorithm for national teams.

Given player statistics (goals, assists, minutes, xG, etc.), computes a
composite player score and aggregates it to a team-level strength modifier.

The modifier can be multiplied into the expected-goals estimate produced by
team-level models.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PlayerStats:
    name: str
    team: str
    position: str
    minutes: float = 0.0
    goals: float = 0.0
    assists: float = 0.0
    xg: float = 0.0
    xa: float = 0.0  # expected assists
    shots: float = 0.0
    key_passes: float = 0.0
    tackles: float = 0.0
    interceptions: float = 0.0
    passes: float = 0.0
    pass_accuracy: float = 0.0
    caps: float = 0.0
    age: float = 25.0
    market_value: float = 0.0
    recent_form_score: float = 0.0  # 0-10
    injury_status: str = "available"  # available, doubt, injured, suspended


class PlayerWeightingEngine:
    """Compute player and team strength scores."""

    # Default feature weights; autoresearch can optimize these.
    # Normalized so they sum to exactly 1.0.
    DEFAULT_WEIGHTS = {
        "goals_per_90": 0.176991,
        "assists_per_90": 0.088496,
        "xg_per_90": 0.132743,
        "xa_per_90": 0.070796,
        "shots_per_90": 0.044248,
        "key_passes_per_90": 0.044248,
        "defensive_actions_per_90": 0.070796,
        "pass_accuracy": 0.044248,
        "minutes_share": 0.088496,
        "caps": 0.070796,
        "age_peak": 0.035398,
        "recent_form": 0.088496,
        "market_value": 0.044248,
    }

    INJURY_MULTIPLIER = {
        "available": 1.0,
        "doubt": 0.70,
        "injured": 0.0,
        "suspended": 0.0,
    }

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

    def _validate_weights(self):
        total = sum(self.weights.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def _per_90(self, value: float, minutes: float) -> float:
        return (value / minutes * 90.0) if minutes > 0 else 0.0

    def _age_peak_factor(self, age: float) -> float:
        """Peak around 27 years old."""
        return max(0.5, 1.0 - abs(age - 27.0) / 15.0)

    def compute_player_score(self, player: PlayerStats) -> float:
        """Composite score for a single player."""
        minutes = max(player.minutes, 1.0)
        features = {
            "goals_per_90": self._per_90(player.goals, minutes),
            "assists_per_90": self._per_90(player.assists, minutes),
            "xg_per_90": self._per_90(player.xg, minutes),
            "xa_per_90": self._per_90(player.xa, minutes),
            "shots_per_90": self._per_90(player.shots, minutes),
            "key_passes_per_90": self._per_90(player.key_passes, minutes),
            "defensive_actions_per_90": self._per_90(player.tackles + player.interceptions, minutes),
            "pass_accuracy": player.pass_accuracy / 100.0,
            "minutes_share": min(player.minutes / (90.0 * 10.0), 1.0),
            "caps": min(player.caps / 50.0, 1.0),
            "age_peak": self._age_peak_factor(player.age),
            "recent_form": player.recent_form_score / 10.0,
            "market_value": min(np.log1p(player.market_value) / np.log1p(100_000_000), 1.0),
        }

        raw_score = sum(features[k] * self.weights[k] for k in self.weights)
        injury_mult = self.INJURY_MULTIPLIER.get(player.injury_status, 1.0)
        return raw_score * injury_mult

    def compute_team_strength(self, players: Iterable[PlayerStats], top_n: int = 14) -> dict[str, float]:
        """
        Aggregate player scores to team-level modifiers.

        Returns attack_modifier, defense_modifier, squad_depth_score.
        """
        player_list = list(players)
        if not player_list:
            return {"attack_modifier": 1.0, "defense_modifier": 1.0, "squad_depth_score": 0.0}

        scores = [(p, self.compute_player_score(p)) for p in player_list]
        scores.sort(key=lambda x: x[1], reverse=True)

        top_players = scores[:top_n]
        top_attack = [p for p, _ in top_players if p.position in {"FW", "MF", "AM"}]
        top_defense = [p for p, _ in top_players if p.position in {"DF", "DM", "GK"}]

        attack_modifier = 1.0 + np.mean([s for p, s in top_attack]) if top_attack else 1.0
        defense_modifier = 1.0 - 0.5 * np.mean([s for p, s in top_defense]) if top_defense else 1.0
        squad_depth_score = np.mean([s for _, s in scores[:18]]) if len(scores) >= 18 else np.mean([s for _, s in scores])

        return {
            "attack_modifier": round(float(attack_modifier), 4),
            "defense_modifier": round(float(defense_modifier), 4),
            "squad_depth_score": round(float(squad_depth_score), 4),
            "top_player_score": round(float(top_players[0][1]), 4) if top_players else 0.0,
        }


def compute_team_strength_from_dataframe(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Given a dataframe of player stats, return team-level modifiers.

    Expected columns: name, team, position, minutes, goals, assists, xg, xa,
    shots, key_passes, tackles, interceptions, passes, pass_accuracy, caps,
    age, market_value, recent_form_score, injury_status.
    """
    engine = PlayerWeightingEngine(weights)
    teams = df["team"].unique()
    rows = []
    for team in teams:
        team_df = df[df["team"] == team]
        players = [PlayerStats(**row) for row in team_df.to_dict(orient="records")]
        result = engine.compute_team_strength(players)
        result["team"] = team
        rows.append(result)
    return pd.DataFrame(rows)


def main():
    # Example with synthetic data
    players = [
        PlayerStats("Messi", "Argentina", "FW", minutes=900, goals=5, assists=3, xg=4.5, recent_form_score=9.0, caps=180, age=37, market_value=30_000_000),
        PlayerStats("Alvarez", "Argentina", "FW", minutes=800, goals=4, assists=1, xg=3.8, recent_form_score=8.0, caps=35, age=24, market_value=80_000_000),
        PlayerStats("Otamendi", "Argentina", "DF", minutes=850, tackles=25, interceptions=15, recent_form_score=7.0, caps=110, age=36, market_value=8_000_000),
    ]
    engine = PlayerWeightingEngine()
    for p in players:
        print(p.name, engine.compute_player_score(p))
    print(engine.compute_team_strength(players))


if __name__ == "__main__":
    main()
