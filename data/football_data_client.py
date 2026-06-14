"""
Client for football-data.co.uk datasets.

Provides historical match results and betting odds for top European leagues
and international tournaments. Useful as an additional data source for
features and market-implied probabilities.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


RAW_DIR = Path(__file__).parent / "raw" / "football_data"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Mapping of league codes to football-data.co.uk filenames
LEAGUE_CODES = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "T1": "Super Lig",
    "G1": "Super League Greece",
}

# Seasons as strings: 2324 means 2023-2024
SEASONS = [f"{y:02d}{(y+1):02d}" for y in range(10, 25)]


def _season_to_years(season: str) -> tuple[int, int]:
    """Convert '2324' to (2023, 2024)."""
    return 2000 + int(season[:2]), 2000 + int(season[2:])


def fetch_league_season(league: str, season: str, timeout: int = 60) -> pd.DataFrame | None:
    """Fetch a single CSV from football-data.co.uk."""
    url = BASE_URL.format(season=season, league=league)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), parse_dates=["Date"], dayfirst=True)
        df["league_code"] = league
        df["season"] = season
        df["season_start_year"], df["season_end_year"] = _season_to_years(season)
        return df
    except Exception as ex:
        print(f"Failed to fetch {url}: {ex}")
        return None


def fetch_all(leagues: Iterable[str] | None = None, seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """Fetch multiple leagues and seasons, returning a combined dataframe."""
    leagues = leagues or ["E0", "SP1", "D1", "I1", "F1"]
    seasons = seasons or SEASONS

    frames = []
    for league in leagues:
        for season in seasons:
            df = fetch_league_season(league, season)
            if df is not None and not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined


def standardize_to_mondial-xboost_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert football-data columns into Mondial-Xboost historical_results format.
    Only includes matches where FTR (full-time result) is present.
    """
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"]),
        "home_team": df["HomeTeam"],
        "away_team": df["AwayTeam"],
        "home_score": pd.to_numeric(df["FTHG"], errors="coerce"),
        "away_score": pd.to_numeric(df["FTAG"], errors="coerce"),
        "tournament": df.get("league_code", "football-data"),
        "city": "",
        "country": "",
        "neutral": False,
    })
    out = out.dropna(subset=["home_score", "away_score"]).copy()
    return out


def save_raw(df: pd.DataFrame, name: str = "football_data_raw") -> Path:
    """Save raw combined data to Parquet."""
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract market-implied probabilities from betting odds columns."""
    odds_cols = [c for c in df.columns if c in {"B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "PSH", "PSD", "PSA"}]
    if not odds_cols:
        return pd.DataFrame()

    features = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True),
        "home_team": df["HomeTeam"],
        "away_team": df["AwayTeam"],
    })

    for outcome, suffix in [("home", "H"), ("draw", "D"), ("away", "A")]:
        cols = [c for c in odds_cols if c.endswith(suffix)]
        if cols:
            features[f"odds_{outcome}"] = df[cols].mean(axis=1)
            features[f"odds_implied_prob_{outcome}"] = 1.0 / features[f"odds_{outcome}"]

    # Normalize implied probs so they sum to 1
    prob_cols = [c for c in features.columns if c.startswith("odds_implied_prob_")]
    total = features[prob_cols].sum(axis=1)
    for col in prob_cols:
        features[col] = features[col] / total

    return features


def main():
    print("Fetching football-data.co.uk datasets...")
    raw = fetch_all(leagues=["E0", "SP1", "D1", "I1", "F1"], seasons=SEASONS[-5:])
    if raw.empty:
        print("No data fetched.")
        return

    path = save_raw(raw)
    print(f"Saved {len(raw)} rows to {path}")

    standardized = standardize_to_mondial-xboost_format(raw)
    print(f"Standardized {len(standardized)} matches")

    odds = load_odds_features(raw)
    print(f"Odds features: {len(odds)} rows, columns: {list(odds.columns)}")


if __name__ == "__main__":
    main()
