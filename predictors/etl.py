"""
ETL Pipeline — Raw to Processed to Features
============================================
Transforms raw data from multiple free sources into ML-ready features.

Sources:
    - football-data.co.uk (CSVs históricos de ligas europeas + internacional)
    - Wikipedia (resultados históricos de mundiales)
    - football-data.co.uk WC (World Cup data)
    - football-data.co.uk EC (Euro Championship data)

Pipeline:
    raw/football-data/  →  processed/matches.parquet  →  features/X_train.parquet
    raw/wikipedia/        →  processed/matches.parquet
    raw/news/             →  processed/news_sentiment.json

Usage:
    from etl import ETLPipeline
    etl = ETLPipeline()
    etl.run_all()
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add scrapers to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scrapers"))
from football_data_scraper import FootballDataScraper
from wikipedia_scraper import WikipediaScraper


class ETLPipeline:
    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")
    FEATURES_DIR = Path("data/features")

    def __init__(self):
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        self.football_data = FootballDataScraper()
        self.wikipedia = WikipediaScraper()

    # ===== Step 1: Process football-data.co.uk =====

    def process_football_data(self, leagues: list[str] = None,
                               start_season: int = 2018,
                               end_season: int = 2024) -> pd.DataFrame:
        """Extract match data from football-data.co.uk CSVs."""
        print(f"Downloading football-data for leagues: {leagues or 'default'}")

        df = self.football_data.build_training_dataset(
            leagues=leagues,
            start_season=start_season,
            end_season=end_season
        )

        if df.empty:
            print("WARNING: No data from football-data.co.uk")
            return pd.DataFrame()

        # Standardize column names
        df = df.rename(columns={
            "home_team": "home_team_name",
            "away_team": "away_team_name",
            "home_goals": "home_goals",
            "away_goals": "away_goals",
            "date": "date",
        })

        # Add source marker
        df["data_source"] = "football-data"

        print(f"Loaded {len(df)} matches from football-data.co.uk")
        return df

    # ===== Step 2: Process Wikipedia Data =====

    def process_wikipedia(self, years: list[int] = None) -> pd.DataFrame:
        """Extract World Cup match data from Wikipedia."""
        if years is None:
            years = [2022, 2018, 2014, 2010, 2006, 2002, 1998, 1994, 1990, 1986, 1982, 1978, 1974, 1970, 1966, 1962, 1958, 1954, 1950, 1938, 1934, 1930]

        all_matches = []

        for year in years:
            print(f"Fetching World Cup {year} from Wikipedia...")
            df = self.wikipedia.get_world_cup_results(year)
            if not df.empty:
                # Convert to standard format
                standard = self.wikipedia.convert_to_standard_format(df)
                all_matches.append(standard)

            import time
            time.sleep(1)  # Rate limiting

        if not all_matches:
            print("WARNING: No data from Wikipedia")
            return pd.DataFrame()

        combined = pd.concat(all_matches, ignore_index=True)
        combined["data_source"] = "wikipedia"

        # Standardize column names
        combined = combined.rename(columns={
            "home_team": "home_team_name",
            "away_team": "away_team_name",
        })

        print(f"Loaded {len(combined)} matches from Wikipedia")
        return combined

    # ===== Step 3: Process API-Football (legacy, if available) =====

    def process_api_football(self, db_path: str = "data/raw/api-football/cache.db") -> pd.DataFrame:
        """Extract match data from API-Football cache (if exists)."""
        db = Path(db_path)
        if not db.exists():
            print(f"INFO: API-Football cache not found at {db_path}, skipping")
            return pd.DataFrame()

        conn = sqlite3.connect(str(db))

        # Get all cached fixture data
        cursor = conn.execute("SELECT key, data FROM cache WHERE key LIKE 'fixtures:%'")
        rows = cursor.fetchall()
        conn.close()

        matches = []
        for key, data_json in rows:
            data = json.loads(data_json)
            for fixture in data.get("response", []):
                fixture_info = fixture.get("fixture", {})
                league = fixture.get("league", {})
                teams = fixture.get("teams", {})
                goals = fixture.get("goals", {})
                score = fixture.get("score", {})

                home_team = teams.get("home", {})
                away_team = teams.get("away", {})

                match = {
                    "fixture_id": fixture_info.get("id"),
                    "date": fixture_info.get("date"),
                    "timestamp": fixture_info.get("timestamp"),
                    "league_id": league.get("id"),
                    "league_name": league.get("name"),
                    "season": league.get("season"),
                    "round": league.get("round"),
                    "home_team_id": home_team.get("id"),
                    "home_team_name": home_team.get("name"),
                    "away_team_id": away_team.get("id"),
                    "away_team_name": away_team.get("name"),
                    "home_goals": goals.get("home"),
                    "away_goals": goals.get("away"),
                    "home_goals_ht": score.get("halftime", {}).get("home"),
                    "away_goals_ht": score.get("halftime", {}).get("away"),
                    "status": fixture_info.get("status", {}).get("short"),
                    "venue": fixture_info.get("venue", {}).get("name"),
                    "city": fixture_info.get("venue", {}).get("city"),
                    "referee": fixture_info.get("referee"),
                    "data_source": "api-football",
                }
                matches.append(match)

        df = pd.DataFrame(matches)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

        print(f"Loaded {len(df)} matches from API-Football cache")
        return df

    def process_team_stats(self, db_path: str = "data/raw/api-football/cache.db") -> pd.DataFrame:
        """Extract team statistics from cache."""
        db = Path(db_path)
        if not db.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(str(db))
        cursor = conn.execute("SELECT key, data FROM cache WHERE key LIKE 'teams/statistics:%'")
        rows = cursor.fetchall()
        conn.close()

        stats = []
        for key, data_json in rows:
            data = json.loads(data_json)
            for response in data.get("response", []):
                team = response.get("team", {})
                league = response.get("league", {})
                fixtures = response.get("fixtures", {})
                goals = response.get("goals", {})

                stat = {
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "league_id": league.get("id"),
                    "season": league.get("season"),
                    "form": response.get("form", ""),
                    "played_home": fixtures.get("played", {}).get("home"),
                    "played_away": fixtures.get("played", {}).get("away"),
                    "played_total": fixtures.get("played", {}).get("total"),
                    "wins_home": fixtures.get("wins", {}).get("home"),
                    "wins_away": fixtures.get("wins", {}).get("away"),
                    "wins_total": fixtures.get("wins", {}).get("total"),
                    "draws_home": fixtures.get("draws", {}).get("home"),
                    "draws_away": fixtures.get("draws", {}).get("away"),
                    "draws_total": fixtures.get("draws", {}).get("total"),
                    "loses_home": fixtures.get("loses", {}).get("home"),
                    "loses_away": fixtures.get("loses", {}).get("away"),
                    "loses_total": fixtures.get("loses", {}).get("total"),
                    "goals_for_home": goals.get("for", {}).get("total", {}).get("home"),
                    "goals_for_away": goals.get("for", {}).get("total", {}).get("away"),
                    "goals_for_total": goals.get("for", {}).get("total", {}).get("total"),
                    "goals_against_home": goals.get("against", {}).get("total", {}).get("home"),
                    "goals_against_away": goals.get("against", {}).get("total", {}).get("away"),
                    "goals_against_total": goals.get("against", {}).get("total", {}).get("total"),
                }
                stats.append(stat)

        return pd.DataFrame(stats)

    def process_player_stats(self, db_path: str = "data/raw/api-football/cache.db") -> pd.DataFrame:
        """Extract player statistics from cache."""
        db = Path(db_path)
        if not db.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(str(db))
        cursor = conn.execute("SELECT id, team_id, league_id, season, stats FROM players")
        rows = cursor.fetchall()
        conn.close()

        players = []
        for player_id, team_id, league_id, season, stats_json in rows:
            stats = json.loads(stats_json)
            players.append(stats)

        return pd.DataFrame(players)

    # ===== Step 2: Process News Data =====

    def process_news(self, db_path: str = "data/raw/news/cache.db") -> pd.DataFrame:
        """Extract news articles from cache."""
        db = Path(db_path)
        if not db.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(str(db))
        cursor = conn.execute("""
            SELECT source, url, title, content, published_at, mentions_teams,
                   injury_keywords, relevance_score, scraped_at
            FROM articles
            ORDER BY scraped_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        articles = []
        for row in rows:
            articles.append({
                "source": row[0],
                "url": row[1],
                "title": row[2],
                "content": row[3],
                "published_at": row[4],
                "mentions_teams": json.loads(row[5]) if row[5] else [],
                "injury_keywords": json.loads(row[6]) if row[6] else [],
                "relevance_score": row[7],
                "scraped_at": row[8]
            })

        df = pd.DataFrame(articles)
        if not df.empty:
            df["scraped_at"] = pd.to_datetime(df["scraped_at"])

        return df

    # ===== Step 3: Save Processed Data =====

    def save_processed(self, df: pd.DataFrame, name: str):
        """Save processed dataframe to parquet."""
        path = self.PROCESSED_DIR / f"{name}.parquet"

        # Ensure all columns are proper types before saving
        df = df.copy()

        # Convert object columns that might have mixed types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric, fallback to string
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except Exception:
                    pass

            # Ensure datetime columns are proper datetime
            if 'date' in col.lower() and df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Ensure season column is string (handles "2012/2013" format from American leagues)
        if 'season' in df.columns:
            df['season'] = df['season'].astype(str)

        df.to_parquet(path, index=False)
        print(f"Saved {name}: {len(df)} rows to {path}")
        return path

    # ===== Step 4: Generate Features =====

    def generate_match_features(self, matches_df: pd.DataFrame,
                                 team_stats_df: pd.DataFrame,
                                 lookback_matches: int = 5) -> pd.DataFrame:
        """Generate features for each match based on historical data."""
        if matches_df.empty or team_stats_df.empty:
            return pd.DataFrame()

        features = []

        for _, match in matches_df.iterrows():
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]
            match_date = match["date"]

            if pd.isna(match_date):
                continue

            # Get historical matches for both teams before this match
            home_history = matches_df[
                ((matches_df["home_team_id"] == home_id) | (matches_df["away_team_id"] == home_id)) &
                (matches_df["date"] < match_date)
            ].sort_values("date", ascending=False).head(lookback_matches)

            away_history = matches_df[
                ((matches_df["home_team_id"] == away_id) | (matches_df["away_team_id"] == away_id)) &
                (matches_df["date"] < match_date)
            ].sort_values("date", ascending=False).head(lookback_matches)

            # Calculate form features
            home_form = self._calculate_form(home_history, home_id)
            away_form = self._calculate_form(away_history, away_id)

            # Get team stats
            home_stats = team_stats_df[team_stats_df["team_id"] == home_id]
            away_stats = team_stats_df[team_stats_df["team_id"] == away_id]

            feature = {
                "fixture_id": match["fixture_id"],
                "date": match_date,
                "home_team_id": home_id,
                "home_team_name": match["home_team_name"],
                "away_team_id": away_id,
                "away_team_name": match["away_team_name"],
                "league_id": match["league_id"],
                "season": match["season"],

                # Form features
                "home_form_points": home_form["points"],
                "home_form_goals_scored": home_form["goals_scored"],
                "home_form_goals_conceded": home_form["goals_conceded"],
                "home_form_win_rate": home_form["win_rate"],
                "away_form_points": away_form["points"],
                "away_form_goals_scored": away_form["goals_scored"],
                "away_form_goals_conceded": away_form["goals_conceded"],
                "away_form_win_rate": away_form["win_rate"],

                # Team stats (if available)
                "home_goals_for_total": home_stats["goals_for_total"].iloc[0] if not home_stats.empty else np.nan,
                "home_goals_against_total": home_stats["goals_against_total"].iloc[0] if not home_stats.empty else np.nan,
                "away_goals_for_total": away_stats["goals_for_total"].iloc[0] if not away_stats.empty else np.nan,
                "away_goals_against_total": away_stats["goals_against_total"].iloc[0] if not away_stats.empty else np.nan,

                # Target
                "home_goals": match["home_goals"],
                "away_goals": match["away_goals"],
                "outcome": self._calculate_outcome(match["home_goals"], match["away_goals"]),
            }

            features.append(feature)

        return pd.DataFrame(features)

    def _calculate_form(self, history: pd.DataFrame, team_id: int) -> dict[str, float]:
        """Calculate form metrics from recent matches."""
        if history.empty:
            return {"points": 0, "goals_scored": 0, "goals_conceded": 0, "win_rate": 0}

        points = 0
        goals_scored = 0
        goals_conceded = 0
        wins = 0

        for _, match in history.iterrows():
            if match["home_team_id"] == team_id:
                team_goals = match["home_goals"] if not pd.isna(match["home_goals"]) else 0
                opp_goals = match["away_goals"] if not pd.isna(match["away_goals"]) else 0
            else:
                team_goals = match["away_goals"] if not pd.isna(match["away_goals"]) else 0
                opp_goals = match["home_goals"] if not pd.isna(match["home_goals"]) else 0

            goals_scored += team_goals
            goals_conceded += opp_goals

            if team_goals > opp_goals:
                points += 3
                wins += 1
            elif team_goals == opp_goals:
                points += 1

        n = len(history)
        return {
            "points": points / max(n * 3, 1) * 9,  # Scale to 0-9
            "goals_scored": goals_scored / n,
            "goals_conceded": goals_conceded / n,
            "win_rate": wins / n
        }

    def _calculate_outcome(self, home_goals, away_goals) -> int | None:
        """Calculate outcome: 0=away win, 1=draw, 2=home win."""
        if pd.isna(home_goals) or pd.isna(away_goals):
            return None
        if home_goals > away_goals:
            return 2
        elif home_goals < away_goals:
            return 0
        else:
            return 1

    # ===== Step 5: Add External Features =====

    def add_elo_features(self, features_df: pd.DataFrame,
                         elo_csv: str = "MondialXboost.Web/elo_snapshot.csv") -> pd.DataFrame:
        """Add Elo rating features from existing CSV."""
        elo_path = Path(elo_csv)
        if not elo_path.exists():
            print(f"WARNING: {elo_csv} not found")
            return features_df

        # Map team names to IDs (this would need a mapping table)
        # For now, just add placeholder columns
        features_df["home_elo"] = np.nan
        features_df["away_elo"] = np.nan
        features_df["elo_diff"] = np.nan

        return features_df

    def add_news_features(self, features_df: pd.DataFrame,
                          news_df: pd.DataFrame) -> pd.DataFrame:
        """Add news sentiment and injury features."""
        if news_df.empty:
            features_df["news_sentiment"] = 0.0
            features_df["injury_severity_home"] = 0.0
            features_df["injury_severity_away"] = 0.0
            return features_df

        # Aggregate news by team and date
        features_df["news_sentiment"] = 0.0
        features_df["injury_severity_home"] = 0.0
        features_df["injury_severity_away"] = 0.0

        return features_df

    # ===== Main Pipeline =====

    def run_all(
        self,
        use_football_data: bool = True,
        use_wikipedia: bool = True,
        use_api_football: bool = False,  # Legacy, disabled by default
        football_data_leagues: list[str] = None,
        football_data_start: int = 2018,
        football_data_end: int = 2024,
        wikipedia_years: list[int] = None,
    ):
        """Run complete ETL pipeline using free data sources.

        Args:
            use_football_data: Enable football-data.co.uk (CSVs gratis)
            use_wikipedia: Enable Wikipedia (resultados históricos mundiales)
            use_api_football: Enable API-Football (legacy, requires paid API key)
            football_data_leagues: List of league codes (e.g., ["E0", "SP1", "WC"])
            football_data_start: Start season year
            football_data_end: End season year
            wikipedia_years: List of World Cup years to fetch
        """
        print("=== ETL Pipeline Start (Free Data Sources) ===")

        all_matches = []

        # Step 1: Process football-data.co.uk
        if use_football_data:
            print("\n[1/5] Processing football-data.co.uk...")
            fd_df = self.process_football_data(
                leagues=football_data_leagues,
                start_season=football_data_start,
                end_season=football_data_end
            )
            if not fd_df.empty:
                all_matches.append(fd_df)
                self.save_processed(fd_df, "matches_football_data")

        # Step 2: Process Wikipedia
        if use_wikipedia:
            print("\n[2/5] Processing Wikipedia...")
            wiki_df = self.process_wikipedia(years=wikipedia_years)
            if not wiki_df.empty:
                all_matches.append(wiki_df)
                self.save_processed(wiki_df, "matches_wikipedia")

        # Step 3: Process API-Football (legacy, if available)
        if use_api_football:
            print("\n[3/5] Processing API-Football (legacy)...")
            api_df = self.process_api_football()
            if not api_df.empty:
                all_matches.append(api_df)
                self.save_processed(api_df, "matches_api_football")

        # Combine all match sources
        print("\n[4/5] Combining match data...")
        if all_matches:
            combined_matches = pd.concat(all_matches, ignore_index=True)
            # Remove duplicates (same date, home, away)
            combined_matches = combined_matches.drop_duplicates(
                subset=["date", "home_team_name", "away_team_name"],
                keep="first"
            )
            self.save_processed(combined_matches, "matches_all")
            print(f"Combined dataset: {len(combined_matches)} unique matches")
        else:
            combined_matches = pd.DataFrame()
            print("WARNING: No match data from any source")

        # Step 4: Process news
        print("\n[5/5] Processing news data...")
        news_df = self.process_news()
        self.save_processed(news_df, "news")

        print("\n=== ETL Pipeline Complete ===")
        print(f"Sources used: football-data={use_football_data}, wikipedia={use_wikipedia}, api-football={use_api_football}")
        print(f"Total matches: {len(combined_matches) if not combined_matches.empty else 0}")

        return combined_matches


# ===== CLI =====

if __name__ == "__main__":
    etl = ETLPipeline()
    features = etl.run_all()

    if not features.empty:
        print(f"\nFeature columns: {list(features.columns)}")
        print("\nSample features:")
        print(features.head(3).to_string())

    print("\nETL OK")
