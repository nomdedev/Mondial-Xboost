"""
Feature engineering for football match outcome prediction.

Reads historical results and computes team-level, head-to-head and context
features for training an XGBoost predictor.

All features are computed using only information available BEFORE the match
(date of the fixture). Temporal leakage is guarded by `as_of_date`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

DEFAULT_ELO = 1500.0
ELO_K = 30.0

# Canonical feature columns produced by this module and expected by the
# XGBoost predictor. This is the single source of truth for the ML pipeline.
FEATURE_COLS = [
    "elo_diff",
    "elo_diff_recent",
    "home_points_avg_5",
    "home_points_avg_10",
    "home_goals_scored_avg_10",
    "home_goals_conceded_avg_10",
    "home_win_rate_10",
    "home_draw_rate_10",
    "home_loss_rate_10",
    "home_matches_played",
    "away_points_avg_5",
    "away_points_avg_10",
    "away_goals_scored_avg_10",
    "away_goals_conceded_avg_10",
    "away_win_rate_10",
    "away_draw_rate_10",
    "away_loss_rate_10",
    "away_matches_played",
    "h2h_wins_diff",
    "h2h_goals_avg",
    "h2h_last_result",
    "h2h_years_since",
    "neutral",
    # New features
    "home_momentum_3",
    "away_momentum_3",
    "home_sos_5",
    "away_sos_5",
    "home_points_weighted_10",
    "away_points_weighted_10",
    "tournament_importance",
]


def load_historical_results(path: Path | str | None = None) -> pd.DataFrame:
    """Load match data from CSV (default) or a provided Parquet/CSV path."""
    if path is None:
        root = Path(__file__).parent.parent
        candidates = [
            root / "MondialXboost.Web" / "Data" / "historical_results.csv",
            root / "data" / "raw" / "historical_results.csv",
            root / "data" / "historical_results.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            # Fallback to the canonical location so the error message is clear.
            path = candidates[0]

    if str(path).endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["date"])

    df = df.sort_values("date").reset_index(drop=True)

    # Detect duplicate fixtures with conflicting scores
    dup_keys = ["date", "home_team", "away_team"]
    if df.duplicated(subset=dup_keys).any():
        dups = df[df.duplicated(subset=dup_keys, keep=False)].sort_values(dup_keys)
        # Check for conflicting scores
        grouped = dups.groupby(dup_keys, sort=False)
        conflicting = []
        for key, group in grouped:
            scores = set(zip(group["home_score"], group["away_score"]))
            if len(scores) > 1:
                conflicting.append(key)
        if conflicting:
            examples = conflicting[:3]
            raise ValueError(
                f"Duplicate fixtures with conflicting scores found: {examples}. "
                "Clean the dataset before training."
            )
        # Exact duplicates: keep first
        df = df.drop_duplicates(subset=dup_keys, keep="first").reset_index(drop=True)

    # Normalize column names (our dataset uses home_team_name/away_team_name, not home_team/away_team)
    if "home_team_name" in df.columns and "home_team" not in df.columns:
        df = df.rename(columns={"home_team_name": "home_team", "away_team_name": "away_team"})

    # Normalize column names (our dataset uses home_goals/away_goals, not home_score/away_score)
    if "home_goals" in df.columns and "home_score" not in df.columns:
        df = df.rename(columns={"home_goals": "home_score", "away_goals": "away_score"})

    # Ensure neutral column exists
    if "neutral" not in df.columns:
        df["neutral"] = False
    else:
        df["neutral"] = df["neutral"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})

    return df


def compute_elo_ratings(
    df: pd.DataFrame,
    initial: float = DEFAULT_ELO,
    k: float = ELO_K,
    decay_halflife_years: float | None = None,
    recent_years: float | None = 8.0,
) -> pd.DataFrame:
    """Add Elo rating columns for home and away teams before each match.

    Uses eloratings.net methodology:
    - K varies by match importance (WC=60, Continental=50, Qualifier=40, League=30, Friendly=20)
    - K is NOT adjusted by goal difference (that would be leakage)
    - Home advantage: +100 points in rating difference
    - Optional temporal decay: inactive teams regress toward ``initial``.
    - Optional recent Elo: second rating computed from the last ``recent_years``.
    """
    decay_halflife_days = decay_halflife_years * 365.25 if decay_halflife_years else None
    recent_cutoff_days = recent_years * 365.25 if recent_years else None

    ratings: dict[str, float] = {}
    ratings_recent: dict[str, float] = {}
    last_match_date: dict[str, pd.Timestamp] = {}
    matches_played: dict[str, int] = {}
    home_elo_before: list[float] = []
    away_elo_before: list[float] = []
    home_elo_recent_list: list[float] = []
    away_elo_recent_list: list[float] = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        home_elo = ratings.get(home, initial)
        away_elo = ratings.get(away, initial)
        home_elo_rec = ratings_recent.get(home, initial)
        away_elo_rec = ratings_recent.get(away, initial)

        # Apply temporal decay based on days since last match
        if decay_halflife_days is not None:
            for team, current_rating in ((home, home_elo), (away, away_elo)):
                last_date = last_match_date.get(team)
                if last_date is not None:
                    days = (date - last_date).days
                    factor = np.exp(-days / decay_halflife_days)
                    ratings[team] = initial + (current_rating - initial) * factor
            home_elo = ratings.get(home, initial)
            away_elo = ratings.get(away, initial)

        home_elo_before.append(home_elo)
        away_elo_before.append(away_elo)
        home_elo_recent_list.append(home_elo_rec)
        away_elo_recent_list.append(away_elo_rec)

        # Determine K by match importance only (score margin is NOT used — that would be leakage)
        match_type = str(row.get("tournament", ""))
        league = str(row.get("league_code", ""))

        if "World Cup" in match_type or "WC" in league:
            k_adj = 60
        elif any(x in match_type for x in ["Euro", "Copa America", "Asian Cup", "Africa Cup"]):
            k_adj = 50
        elif "qualifier" in match_type.lower():
            k_adj = 40
        elif any(x in league for x in ["E0", "SP1", "I1", "D1", "F1", "N1", "B1", "P1", "T1", "G1"]):
            k_adj = 30  # Major leagues
        else:
            k_adj = 20  # Friendly / minor league

        # Expected scores with home advantage (+100 points) unless match is neutral
        home_score = row["home_score"]
        away_score = row["away_score"]
        is_neutral = bool(row.get("neutral", False))
        home_advantage = 0 if is_neutral else 100

        def _update(rh: float, ra: float) -> tuple[float, float]:
            dr = (rh - ra) + home_advantage
            expected_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
            result_home = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
            delta = k_adj * (result_home - expected_home)
            return rh + delta, ra - delta

        home_elo, away_elo = _update(home_elo, away_elo)
        ratings[home] = home_elo
        ratings[away] = away_elo

        # Recent Elo only considers matches within recent_years
        if recent_cutoff_days is None:
            home_elo_rec, away_elo_rec = _update(home_elo_rec, away_elo_rec)
            ratings_recent[home] = home_elo_rec
            ratings_recent[away] = away_elo_rec
        else:
            # Add this match and drop contributions older than the cutoff.
            # Approximation: maintain a recent rating and regress it toward initial
            # for the inactive period before applying the new delta.
            for team, current_rating in ((home, home_elo_rec), (away, away_elo_rec)):
                last_date = last_match_date.get(team)
                if last_date is not None:
                    days = (date - last_match_date.get(team)).days
                    if days > recent_cutoff_days:
                        # Full regression to initial if inactive longer than the window
                        ratings_recent[team] = initial
                    else:
                        factor = np.exp(-days / (recent_cutoff_days / np.log(2)))
                        ratings_recent[team] = initial + (current_rating - initial) * factor
            home_elo_rec = ratings_recent.get(home, initial)
            away_elo_rec = ratings_recent.get(away, initial)
            home_elo_rec, away_elo_rec = _update(home_elo_rec, away_elo_rec)
            ratings_recent[home] = home_elo_rec
            ratings_recent[away] = away_elo_rec

        last_match_date[home] = date
        last_match_date[away] = date
        matches_played[home] = matches_played.get(home, 0) + 1
        matches_played[away] = matches_played.get(away, 0) + 1

    df = df.copy()
    df["home_elo_before"] = home_elo_before
    df["away_elo_before"] = away_elo_before
    df["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]
    df["home_elo_recent"] = home_elo_recent_list
    df["away_elo_recent"] = away_elo_recent_list
    df["elo_diff_recent"] = df["home_elo_recent"] - df["away_elo_recent"]
    df["home_elo_provisional"] = df["home_team"].map(lambda x: matches_played.get(x, 0) < 30)
    df["away_elo_provisional"] = df["away_team"].map(lambda x: matches_played.get(x, 0) < 30)
    return df


def _build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert results into a long-format team history where each row is one team's
    perspective of a match. This makes rolling computations vectorized.
    """
    home_cols = ["date", "home_team", "away_team", "home_score", "away_score", "neutral"]
    elo_cols = ["home_elo_before", "away_elo_before"]
    if all(c in df.columns for c in elo_cols):
        home_cols.extend(elo_cols)

    home = df[home_cols].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["goals_scored"] = home["home_score"]
    home["goals_conceded"] = home["away_score"]
    home["is_home"] = True
    home["points"] = np.where(home["home_score"] > home["away_score"], 3,
                              np.where(home["home_score"] == home["away_score"], 1, 0))
    if "home_elo_before" in home.columns:
        home["team_elo_before"] = home["home_elo_before"]
        home["opponent_elo_before"] = home["away_elo_before"]

    away = df[home_cols].copy() if all(c in df.columns for c in elo_cols) else df[["date", "home_team", "away_team", "home_score", "away_score", "neutral"]].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["goals_scored"] = away["away_score"]
    away["goals_conceded"] = away["home_score"]
    away["is_home"] = False
    away["points"] = np.where(away["away_score"] > away["home_score"], 3,
                              np.where(away["away_score"] == away["home_score"], 1, 0))
    if "home_elo_before" in away.columns:
        away["team_elo_before"] = away["away_elo_before"]
        away["opponent_elo_before"] = away["home_elo_before"]

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)
    return long


def _compute_team_rolling(long: pd.DataFrame, windows: list[int] = (5, 10)) -> pd.DataFrame:
    """Compute rolling team stats per team before each date."""
    long = long.copy()
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    for w in windows:
        # Shift 1 to avoid including the current match
        long[f"points_avg_{w}"] = long.groupby("team")["points"].transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=1).mean() / 3.0
        )
        long[f"goals_scored_avg_{w}"] = long.groupby("team")["goals_scored"].transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=1).mean()
        )
        long[f"goals_conceded_avg_{w}"] = long.groupby("team")["goals_conceded"].transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=1).mean()
        )
        long[f"win_rate_{w}"] = long.groupby("team")["points"].transform(
            lambda s: (s.shift(1).rolling(window=w, min_periods=1).apply(lambda x: (x == 3).sum()) / s.shift(1).rolling(window=w, min_periods=1).count())
        )
        long[f"draw_rate_{w}"] = long.groupby("team")["points"].transform(
            lambda s: (s.shift(1).rolling(window=w, min_periods=1).apply(lambda x: (x == 1).sum()) / s.shift(1).rolling(window=w, min_periods=1).count())
        )
        long[f"loss_rate_{w}"] = long.groupby("team")["points"].transform(
            lambda s: (s.shift(1).rolling(window=w, min_periods=1).apply(lambda x: (x == 0).sum()) / s.shift(1).rolling(window=w, min_periods=1).count())
        )

    long["matches_played"] = long.groupby("team").cumcount()
    return long


def _add_entropy_weighted_points(long: pd.DataFrame, windows: list[int] = (10,)) -> pd.DataFrame:
    """
    Añade rolling averages ponderadas por entropía del resultado esperado.
    
    Resultados más sorprendentes (un underdog ganando al favorito) pesan MÁS
    porque señalizan cambios reales en la calidad del equipo.
    El peso se calcula como: 1 / p_elo(resultado_observado)
    """
    if "team_elo_before" not in long.columns:
        return long

    long = long.copy()
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    # Calcular probabilidad Elo de cada resultado para cada fila
    team_elo = long["team_elo_before"].values
    opponent_elo = long["opponent_elo_before"].values
    is_home = long["is_home"].values
    outcome = long["points"].values.copy()
    outcome[outcome == 3] = 2

    elo_probs = np.zeros((len(long), 3))
    dr = team_elo - opponent_elo
    home_adj = np.where(is_home, 100, 0)
    dr_adj = dr + home_adj
    for i in range(len(long)):
        home_exp = 1.0 / (1.0 + 10.0 ** (-dr_adj[i] / 400.0))
        elo_probs[i] = [1 - home_exp, 0.0, home_exp]
        elo_probs[i, 1] = 0.15 * (1 - abs(2 * home_exp - 1))
        elo_probs[i, 0] -= elo_probs[i, 1] / 2
        elo_probs[i, 2] -= elo_probs[i, 1] / 2
        elo_probs[i] = np.maximum(elo_probs[i], 0.01)
        elo_probs[i] /= elo_probs[i].sum()

    long["_surprise_weight"] = 1.0 / (elo_probs[np.arange(len(long)), outcome.astype(int)] + 0.05)
    long["_surprise_weight"] = long["_surprise_weight"].clip(0.5, 5.0)

    for w in windows:
        def _weighted_rolling(series, weights, ww):
            result = np.full(len(series), np.nan)
            for i in range(len(series)):
                start = max(0, i - ww)
                vals = series.values[max(0, i - ww):i]
                wts = weights.values[max(0, i - ww):i]
                if len(vals) == 0:
                    result[i] = np.nan
                else:
                    result[i] = np.average(vals, weights=wts) / 3.0
            return result

        long[f"points_weighted_{w}"] = long.groupby("team").apply(
            lambda g: pd.Series(_weighted_rolling(g["points"], g["_surprise_weight"], w), index=g.index)
        ).reset_index(level=0, drop=True)

    long = long.drop(columns=["_surprise_weight"])
    return long


def _add_sos_features(long: pd.DataFrame, windows: list[int] = (5,)) -> pd.DataFrame:
    """
    Añade Strength of Schedule (SOS): el Elo promedio de los oponentes recientes.
    Captura si un equipo viene de jugar contra rivales fuertes o débiles.
    """
    if "opponent_elo_before" not in long.columns:
        return long

    long = long.copy()
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    for w in windows:
        long[f"sos_{w}"] = long.groupby("team")["opponent_elo_before"].transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=1).mean()
        )

    return long


def _add_momentum_features(long: pd.DataFrame) -> pd.DataFrame:
    """
    Añade momentum: diferencia entre forma reciente (últimos 3) y forma histórica (últimos 20).
    Un valor positivo indica que el equipo está en racha.
    """
    long = long.copy()
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    short_window = 3
    long_window = 15

    def _rolling_mean(series, w):
        return series.shift(1).rolling(window=w, min_periods=1).mean()

    points_norm = long["points"] / 3.0
    long["momentum"] = _rolling_mean(points_norm, short_window) - _rolling_mean(points_norm, long_window)
    long["momentum"] = long["momentum"].fillna(0.0)

    return long


def _compute_h2h(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Compute head-to-head stats per ordered pair before each date."""
    df = df.copy()
    # Create a canonical pair key so A-vs-B and B-vs-A group together
    df["pair_key"] = df.apply(lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1)
    df = df.sort_values(["pair_key", "date"]).reset_index(drop=True)

    # Home-team perspective result for each match
    df["home_result"] = np.where(
        df["home_score"] > df["away_score"], 1.0,
        np.where(df["home_score"] == df["away_score"], 0.5, 0.0)
    )

    # Last result for the pair before current match, from current home_team perspective
    def _last_result_from_home(g):
        out = []
        prev_home_team = None
        prev_home_result = None
        for _, row in g.iterrows():
            if prev_home_result is None:
                out.append(np.nan)
            else:
                if row["home_team"] == prev_home_team:
                    out.append(prev_home_result)
                else:
                    # Current home was away in previous match
                    out.append(1.0 - prev_home_result if prev_home_result != 0.5 else 0.5)
            prev_home_team = row["home_team"]
            prev_home_result = row["home_result"]
        return pd.Series(out, index=g.index)

    df["h2h_last_result"] = df.groupby("pair_key", group_keys=False).apply(_last_result_from_home)

    # Rolling h2h goals and wins diff
    df["h2h_goals"] = df["home_score"] + df["away_score"]
    df["h2h_goals_avg"] = df.groupby("pair_key")["h2h_goals"].transform(
        lambda s: s.shift(1).rolling(window=n, min_periods=1).mean()
    )

    # Home-team-centric wins diff in last n h2h matches
    def _wins_diff(g):
        home_score = g["home_score"].values
        away_score = g["away_score"].values
        wins = np.where(home_score > away_score, 1, np.where(home_score == away_score, 0, -1))
        return pd.Series(wins, index=g.index).shift(1).rolling(window=n, min_periods=1).mean()

    df["h2h_wins_diff"] = df.groupby("pair_key", group_keys=False).apply(_wins_diff)

    # Years since last h2h
    df["h2h_years_since"] = df.groupby("pair_key")["date"].diff().dt.days / 365.25
    return df


def _recent_matches_count(
    historical: pd.DataFrame,
    fixtures: pd.DataFrame,
    years: float = 8.0,
) -> pd.DataFrame:
    """Return recent match counts per team as of each fixture date."""
    from bisect import bisect_left

    cutoff = pd.Timedelta(days=years * 365.25)

    # Precompute sorted match dates per team.
    team_dates: dict[str, list[pd.Timestamp]] = {}
    for _, row in historical.iterrows():
        date = row["date"]
        for team in (row["home_team"], row["away_team"]):
            team_dates.setdefault(team, []).append(date)
    for team in team_dates:
        team_dates[team].sort()

    rows = []
    for _, fx in fixtures.iterrows():
        date = fx["date"]
        home, away = fx["home_team"], fx["away_team"]
        window_start = date - cutoff

        def _count(team: str) -> int:
            dates = team_dates.get(team, [])
            left = bisect_left(dates, window_start)
            right = bisect_left(dates, date)
            return right - left

        rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_recent_matches": _count(home),
            "away_recent_matches": _count(away),
        })
    return pd.DataFrame(rows)


def build_features(
    historical: pd.DataFrame,
    fixtures: pd.DataFrame,
    min_date: str | None = None,
    elo_decay_halflife_years: float | None = None,
    elo_recent_years: float | None = 8.0,
) -> pd.DataFrame:
    """
    Build a feature matrix for the supplied fixtures using only historical data
    available before each fixture's date.

    Args:
        historical: Full match history used to compute look-back features
            (Elo, rolling stats, H2H). This should NOT be pre-filtered by date.
        fixtures: Matches to featurize. For training these are usually a subset
            of historical limited by min_date; for prediction they are future fixtures.
        min_date: If provided, drop training rows earlier than this date from
            the returned feature matrix. Prediction callers should leave this as None.
        elo_decay_halflife_years: If set, regress inactive Elo ratings toward
            the default (1500) with this half-life.
        elo_recent_years: Window in years for the recent Elo feature.
    """
    historical = historical.copy()
    fixtures = fixtures.copy()

    for df in (historical, fixtures):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Elo ratings
    historical = compute_elo_ratings(
        historical,
        decay_halflife_years=elo_decay_halflife_years,
        recent_years=elo_recent_years,
    )

    # Team rolling stats from long-format history
    long = _build_team_history(historical)
    long = _compute_team_rolling(long)
    long = _add_entropy_weighted_points(long)
    long = _add_sos_features(long)
    long = _add_momentum_features(long)

    # H2H stats
    historical = _compute_h2h(historical)

    # Merge team rolling stats to fixtures
    home_stats = long[long["is_home"]].copy()
    away_stats = long[~long["is_home"]].copy()

    # We need stats as of the fixture date. Since historical includes the fixture itself,
    # we can use the long rows that correspond to each fixture.
    # For fixtures that are NOT in historical, we approximate with most recent known stats.
    # We'll build a lookup of latest stats per team as of each historical date.

    # Simplification: for training, fixtures are historical rows, so we can merge directly.
    # For prediction, we use the latest known stats from historical.

    fixtures = fixtures.merge(
        historical[["date", "home_team", "away_team", "home_elo_before", "away_elo_before",
                    "home_elo_recent", "away_elo_recent",
                    "h2h_last_result", "h2h_goals_avg", "h2h_wins_diff", "h2h_years_since"]],
        on=["date", "home_team", "away_team"],
        how="left",
    )

    # Recent match counts (used by the cold-start model)
    recent_counts = _recent_matches_count(historical, fixtures, years=elo_recent_years or 8.0)
    fixtures = fixtures.merge(
        recent_counts,
        on=["date", "home_team", "away_team"],
        how="left",
    )

    # Preserve the fixture's own neutral flag (historical may also carry a neutral column)
    if "neutral_x" in fixtures.columns:
        fixtures["neutral"] = fixtures["neutral_x"].fillna(False).astype(bool)
        fixtures = fixtures.drop(columns=["neutral_x", "neutral_y"], errors="ignore")

    # Merge home team rolling stats
    home_cols = ["date", "team", "points_avg_5", "points_avg_10",
                 "goals_scored_avg_10", "goals_conceded_avg_10",
                 "win_rate_10", "draw_rate_10", "loss_rate_10", "matches_played",
                 "momentum", "points_weighted_10", "sos_5"]
    available_home = [c for c in home_cols if c in home_stats.columns]
    fixtures = fixtures.merge(
        home_stats[available_home].rename(columns={"team": "home_team"}),
        on=["date", "home_team"],
        how="left",
    )
    # Merge away team rolling stats
    available_away = [c + "_away" if c not in ("date", "team") else c for c in available_home]
    fixtures = fixtures.merge(
        away_stats[list(set(available_home) | {"date", "team"})].rename(columns={"team": "away_team"}),
        on=["date", "away_team"],
        how="left",
        suffixes=("", "_away"),
    )

    # Rename home and away columns
    fixtures = fixtures.rename(columns={
        "points_avg_5": "home_points_avg_5",
        "points_avg_10": "home_points_avg_10",
        "goals_scored_avg_10": "home_goals_scored_avg_10",
        "goals_conceded_avg_10": "home_goals_conceded_avg_10",
        "win_rate_10": "home_win_rate_10",
        "draw_rate_10": "home_draw_rate_10",
        "loss_rate_10": "home_loss_rate_10",
        "matches_played": "home_matches_played",
        "momentum": "home_momentum_3",
        "points_weighted_10": "home_points_weighted_10",
        "sos_5": "home_sos_5",
        "points_avg_5_away": "away_points_avg_5",
        "points_avg_10_away": "away_points_avg_10",
        "goals_scored_avg_10_away": "away_goals_scored_avg_10",
        "goals_conceded_avg_10_away": "away_goals_conceded_avg_10",
        "win_rate_10_away": "away_win_rate_10",
        "draw_rate_10_away": "away_draw_rate_10",
        "loss_rate_10_away": "away_loss_rate_10",
        "matches_played_away": "away_matches_played",
        "momentum_away": "away_momentum_3",
        "points_weighted_10_away": "away_points_weighted_10",
        "sos_5_away": "away_sos_5",
    })

    # Compute elo_diff from before-match ratings
    fixtures["elo_diff"] = fixtures["home_elo_before"] - fixtures["away_elo_before"]
    fixtures["elo_diff_recent"] = fixtures["home_elo_recent"] - fixtures["away_elo_recent"]

    # Fill missing values for teams without history
    fixtures["elo_diff"] = fixtures["elo_diff"].fillna(0.0)
    fixtures["elo_diff_recent"] = fixtures["elo_diff_recent"].fillna(0.0)
    for col in ["home_points_avg_5", "home_points_avg_10", "away_points_avg_5", "away_points_avg_10"]:
        fixtures[col] = fixtures[col].fillna(0.5)
    for col in ["home_win_rate_10", "home_draw_rate_10", "home_loss_rate_10",
                "away_win_rate_10", "away_draw_rate_10", "away_loss_rate_10",
                "h2h_last_result"]:
        fixtures[col] = fixtures[col].fillna(0.0)
    for col in ["home_goals_scored_avg_10", "home_goals_conceded_avg_10",
                "away_goals_scored_avg_10", "away_goals_conceded_avg_10",
                "h2h_goals_avg"]:
        fixtures[col] = fixtures[col].fillna(1.3)
    fixtures["home_matches_played"] = fixtures["home_matches_played"].fillna(0).astype(int)
    fixtures["away_matches_played"] = fixtures["away_matches_played"].fillna(0).astype(int)
    fixtures["home_recent_matches"] = fixtures["home_recent_matches"].fillna(0).astype(int)
    fixtures["away_recent_matches"] = fixtures["away_recent_matches"].fillna(0).astype(int)
    fixtures["h2h_wins_diff"] = fixtures["h2h_wins_diff"].fillna(0.0)
    fixtures["h2h_years_since"] = fixtures["h2h_years_since"].fillna(20.0)

    # Fill new features
    for col in ["home_momentum_3", "away_momentum_3"]:
        if col in fixtures.columns:
            fixtures[col] = fixtures[col].fillna(0.0)
    for col in ["home_sos_5", "away_sos_5"]:
        if col in fixtures.columns:
            fixtures[col] = fixtures[col].fillna(1500.0)
    for col in ["home_points_weighted_10", "away_points_weighted_10"]:
        if col in fixtures.columns:
            fixtures[col] = fixtures[col].fillna(0.5)

    # Tournament importance feature
    IMPORTANCE_MAP = {
        "FIFA World Cup": 5.0, "UEFA Euro": 4.5, "Copa America": 4.5,
        "African Cup of Nations": 4.0, "AFC Asian Cup": 4.0,
        "FIFA World Cup qualification": 3.5, "UEFA Euro qualification": 3.0,
        "Copa America qualification": 3.0,
        "UEFA Nations League": 2.5, "CONCACAF Nations League": 2.5,
        "African Cup of Nations qualification": 2.5,
        "Gold Cup": 2.5, "AFC Asian Cup qualification": 2.0,
        "Friendly": 0.5,
    }
    if "tournament" in fixtures.columns:
        fixtures["tournament_importance"] = fixtures["tournament"].map(
            lambda t: next((v for k, v in IMPORTANCE_MAP.items() if k.lower() in str(t).lower()), 1.0)
        )
    else:
        fixtures["tournament_importance"] = 1.0
    fixtures["tournament_importance"] = fixtures["tournament_importance"].fillna(1.0)

    # Targets
    if "home_score" in fixtures and "away_score" in fixtures:
        fixtures["outcome"] = np.where(
            fixtures["home_score"] > fixtures["away_score"], 2,
            np.where(fixtures["home_score"] == fixtures["away_score"], 1, 0)
        )
        fixtures["btts"] = ((fixtures["home_score"] > 0) & (fixtures["away_score"] > 0)).astype(int)
        fixtures["over_2_5"] = ((fixtures["home_score"] + fixtures["away_score"]) > 2.5).astype(int)

    if min_date is not None:
        fixtures = fixtures[fixtures["date"] >= pd.to_datetime(min_date)].copy()

    return fixtures


def build_training_dataset(
    historical: pd.DataFrame | None = None,
    min_date: str | None = None,
    elo_decay_halflife_years: float | None = None,
    elo_recent_years: float | None = 8.0,
) -> pd.DataFrame:
    """Build a training dataset without temporal leakage.

    The full historical record is used to compute look-back features (Elo,
    rolling stats, H2H) so that teams with sparse recent history still benefit
    from their older matches. The returned matrix is then optionally limited by
    min_date.
    """
    if historical is None:
        historical = load_historical_results()

    # Use the full history for feature computation; min_date filters the rows later.
    cols = ["date", "home_team", "away_team", "home_score", "away_score", "neutral"]
    if "tournament" in historical.columns:
        cols.append("tournament")
    fixtures = historical[cols].copy()

    return build_features(
        historical,
        fixtures,
        min_date=min_date,
        elo_decay_halflife_years=elo_decay_halflife_years,
        elo_recent_years=elo_recent_years,
    )


def save_features(df: pd.DataFrame, name: str = "features") -> Path:
    """Save feature matrix to Parquet and a JSON sample for inspection."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = FEATURES_DIR / f"{name}.parquet"
    df.to_parquet(parquet_path, index=False)

    sample_path = PROCESSED_DIR / f"{name}_sample.json"
    sample = df.head(5).fillna("null").to_dict(orient="records")
    sample_path.write_text(json.dumps(sample, indent=2, default=str), encoding="utf-8")

    return parquet_path


if __name__ == "__main__":
    print("Loading historical results...")
    historical = load_historical_results()
    print(f"Historical matches: {len(historical)}")

    print("Building training dataset (no temporal leakage)...")
    train = build_training_dataset(historical, min_date="2010-01-01")
    print(f"Training rows: {len(train)}")

    path = save_features(train, "train_historical")
    print(f"Saved to {path}")
    print(train.head())
