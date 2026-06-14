"""
FootballDataScraper — football-data.co.uk CSV Downloader
=========================================================
Downloads free historical football data CSVs from football-data.co.uk.
This site provides match results and betting odds for free download.

Data available:
- Match results (home/away goals, date, referee, etc.)
- Betting odds (various bookmakers)
- Half-time results
- Cards, corners (some leagues)

Usage:
    from football_data_scraper import FootballDataScraper
    scraper = FootballDataScraper()
    
    # Download World Cup 2022 data
    df = scraper.download_world_cup_data(season=2022)
    
    # Download all international data
    scraper.download_all_international()
"""
import pandas as pd
import requests
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import time


class FootballDataScraper:
    """
    Scraper for football-data.co.uk
    
    URL format: https://www.football-data.co.uk/mmz4281/YYYY/LEAGUE.csv
    Where:
    - YYYY = season (e.g., 2223 for 2022-23 season)
    - LEAGUE = league code (e.g., E0 for Premier League, SP1 for La Liga, I1 for Serie A)
    
    International tournaments:
    - World Cup: WC (but limited historical data)
    - Euro: EC
    
    Note: This site focuses on European leagues. For international tournaments,
    we may need to supplement with other sources.
    """
    
    BASE_URL = "https://www.football-data.co.uk/mmz4281"
    NEW_URL = "https://www.football-data.co.uk/new"
    
    # League codes for major leagues (European - use mmz4281/season/code.csv)
    LEAGUE_CODES = {
        # England
        "E0": "Premier League",
        "E1": "Championship",
        "E2": "League 1",
        "E3": "League 2",
        "EC": "Conference",
        
        # Scotland
        "SC0": "Scottish Premiership",
        "SC1": "Scottish Division 1",
        "SC2": "Scottish Division 2",
        "SC3": "Scottish Division 3",
        
        # Germany
        "D1": "Bundesliga",
        "D2": "Bundesliga 2",
        
        # Italy
        "I1": "Serie A",
        "I2": "Serie B",
        
        # Spain
        "SP1": "La Liga",
        "SP2": "Segunda Division",
        
        # France
        "F1": "Ligue 1",
        "F2": "Ligue 2",
        
        # Netherlands
        "N1": "Eredivisie",
        
        # Belgium
        "B1": "Jupiler League",
        
        # Portugal
        "P1": "Primeira Liga",
        
        # Turkey
        "T1": "Super Lig",
        
        # Greece
        "G1": "Super League Greece",
        
        # Argentina
        "ARG": "Primera Division",
        
        # Austria
        "AUS": "Bundesliga",
        
        # Brazil
        "BRA": "Serie A",
        
        # China
        "CHN": "Super League",
        
        # Denmark
        "DNK": "Superliga",
        
        # Finland
        "FIN": "Veikkausliiga",
        
        # Ireland
        "IRL": "Premier Division",
        
        # Japan
        "JPN": "J-League",
        
        # Mexico
        "MEX": "Liga MX",
        
        # Norway
        "NOR": "Eliteserien",
        
        # Poland
        "POL": "Ekstraklasa",
        
        # Romania
        "ROU": "Liga 1",
        
        # Russia
        "RUS": "Premier League",
        
        # Sweden
        "SWE": "Allsvenskan",
        
        # Switzerland
        "SWZ": "Super League",
        
        # USA
        "USA": "MLS",
        
        # International
        "WC": "World Cup",
        "EC": "European Championship",
    }
    
    # International tournament codes (not always available)
    INTERNATIONAL_CODES = {
        "WC": "World Cup",
        "EC": "Euro Championship",
    }
    
    # American and other leagues (use new/CODE.csv - all seasons in one file)
    NEW_LEAGUE_CODES = {
        "ARG": "Argentina Primera Division",
        "AUS": "Austria Bundesliga",
        "BRA": "Brazil Serie A",
        "CHN": "China Super League",
        "DNK": "Denmark Superliga",
        "FIN": "Finland Veikkausliiga",
        "IRL": "Ireland Premier Division",
        "JPN": "Japan J-League",
        "MEX": "Mexico Liga MX",
        "NOR": "Norway Eliteserien",
        "POL": "Poland Ekstraklasa",
        "ROU": "Romania Liga 1",
        "RUS": "Russia Premier League",
        "SWE": "Sweden Allsvenskan",
        "SWZ": "Switzerland Super League",
        "USA": "USA MLS",
    }
    
    def __init__(self, cache_dir: str = "data/raw/football-data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _season_to_code(self, season: int) -> str:
        """Convert season year to football-data code.
        
        2022-23 season = "2223"
        2023-24 season = "2324"
        """
        start = str(season)[2:]
        end = str(season + 1)[2:]
        return f"{start}{end}"
    
    def download_csv(self, league_code: str, season: int) -> Optional[pd.DataFrame]:
        """Download CSV for a specific league and season.
        
        For European leagues: uses mmz4281/season/code.csv
        For American/other leagues: uses new/code.csv (all seasons in one file)
        """
        # Check if it's a "new" league (American, etc.)
        if league_code in self.NEW_LEAGUE_CODES:
            return self._download_new_league(league_code)
        
        # European league format
        season_code = self._season_to_code(season)
        url = f"{self.BASE_URL}/{season_code}/{league_code}.csv"
        
        cache_file = self.cache_dir / f"{league_code}_{season_code}.csv"
        
        # Check cache
        if cache_file.exists():
            print(f"Using cached: {cache_file}")
            return pd.read_csv(cache_file)
        
        print(f"Downloading: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save raw CSV
            with open(cache_file, "wb") as f:
                f.write(response.content)
            
            # Parse CSV
            df = pd.read_csv(cache_file)
            
            print(f"Downloaded {len(df)} rows for {league_code} {season_code}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return None
    
    def _download_new_league(self, league_code: str) -> Optional[pd.DataFrame]:
        """Download CSV for American/other leagues (all seasons in one file)."""
        url = f"{self.NEW_URL}/{league_code}.csv"
        
        cache_file = self.cache_dir / f"{league_code}_all.csv"
        
        # Check cache
        if cache_file.exists():
            print(f"Using cached: {cache_file}")
            try:
                return pd.read_csv(cache_file)
            except pd.errors.ParserError:
                # Cache corrupted, re-download
                print(f"Cache corrupted for {league_code}, re-downloading...")
                cache_file.unlink()
        
        print(f"Downloading: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save raw CSV
            with open(cache_file, "wb") as f:
                f.write(response.content)
            
            # Parse CSV with error handling for encoding issues
            try:
                df = pd.read_csv(cache_file)
            except pd.errors.ParserError:
                # Try with different encoding
                df = pd.read_csv(cache_file, encoding='latin-1', on_bad_lines='skip')
            
            print(f"Downloaded {len(df)} rows for {league_code} (all seasons)")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return None
    
    def download_league_range(self, league_code: str, 
                               start_season: int, 
                               end_season: int) -> pd.DataFrame:
        """Download multiple seasons for a league."""
        # For "new" leagues, just download once (all seasons in one file)
        if league_code in self.NEW_LEAGUE_CODES:
            df = self.download_csv(league_code, 0)  # season ignored for new leagues
            if df is not None:
                df["league_code"] = league_code
            return df if df is not None else pd.DataFrame()
        
        # European leagues: download each season
        all_data = []
        
        for season in range(start_season, end_season + 1):
            df = self.download_csv(league_code, season)
            if df is not None:
                df["season"] = season
                df["league_code"] = league_code
                all_data.append(df)
            
            time.sleep(0.5)  # Rate limiting
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def download_all_leagues(self, season: int) -> Dict[str, pd.DataFrame]:
        """Download all available European leagues for a season."""
        results = {}
        
        for code in self.LEAGUE_CODES.keys():
            df = self.download_csv(code, season)
            if df is not None:
                results[code] = df
        
        return results
    
    def download_new_leagues(self) -> Dict[str, pd.DataFrame]:
        """Download all American/other leagues (all seasons in one file each)."""
        results = {}
        
        for code in self.NEW_LEAGUE_CODES.keys():
            df = self.download_csv(code, 0)  # season ignored
            if df is not None:
                results[code] = df
            time.sleep(0.5)  # Rate limiting
        
        return results
    
    def download_world_cup_data(self, season: int = 2022) -> Optional[pd.DataFrame]:
        """Download World Cup data if available."""
        return self.download_csv("WC", season)
    
    def download_euro_data(self, season: int = 2021) -> Optional[pd.DataFrame]:
        """Download Euro Championship data if available."""
        return self.download_csv("EC", season)
    
    def get_available_columns(self, league_code: str = "E0", season: int = 2023) -> List[str]:
        """Get list of columns available in the CSV."""
        df = self.download_csv(league_code, season)
        if df is not None:
            return list(df.columns)
        return []
    
    def extract_key_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only the key columns we need for prediction."""
        # Core columns that are always present
        key_cols = [
            "Date",       # Match date
            "HomeTeam",   # Home team name
            "AwayTeam",   # Away team name
            "FTHG",       # Full-time home goals
            "FTAG",       # Full-time away goals
            "FTR",        # Full-time result (H=home, D=draw, A=away)
            "HTHG",       # Half-time home goals
            "HTAG",       # Half-time away goals
            "HTR",        # Half-time result
            "Referee",    # Referee name
            "HS",         # Home shots
            "AS",         # Away shots
            "HST",        # Home shots on target
            "AST",        # Away shots on target
            "HC",         # Home corners
            "AC",         # Away corners
            "HF",         # Home fouls
            "AF",         # Away fouls
            "HY",         # Home yellow cards
            "AY",         # Away yellow cards
            "HR",         # Home red cards
            "AR",         # Away red cards
        ]
        
        # Only keep columns that exist in the dataframe
        available_cols = [col for col in key_cols if col in df.columns]
        
        result = df[available_cols].copy()
        
        # Add season info if available
        if "season" in df.columns:
            result["season"] = df["season"]
        if "league_code" in df.columns:
            result["league_code"] = df["league_code"]
        
        return result
    
    def get_betting_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract betting odds columns if available."""
        odds_cols = [
            "B365H", "B365D", "B365A",  # Bet365
            "BWH", "BWD", "BWA",        # Bet&Win
            "IWH", "IWD", "IWA",        # Interwetten
            "PSH", "PSD", "PSA",        # Pinnacle
            "WHH", "WHD", "WHA",        # William Hill
            "VCH", "VCD", "VCA",        # VC Bet
        ]
        
        available_odds = [col for col in odds_cols if col in df.columns]
        
        if not available_odds:
            return pd.DataFrame()
        
        return df[available_odds].copy()
    
    def convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert football-data format to our standard match format.
        
        Handles both European format (Date, HomeTeam, AwayTeam, FTHG, FTAG)
        and American format (Date, Home, Away, HG, AG, Res).
        """
        result = []
        
        # Detect format: European uses "HomeTeam", American uses "Home"
        is_american_format = "Home" in df.columns and "HomeTeam" not in df.columns
        
        for _, row in df.iterrows():
            if is_american_format:
                # American format (new/ARG.csv, etc.)
                match = {
                    "date": row.get("Date"),
                    "home_team": row.get("Home"),
                    "away_team": row.get("Away"),
                    "home_goals": row.get("HG"),
                    "away_goals": row.get("AG"),
                    "result": row.get("Res"),  # H, D, A
                    "ht_home_goals": None,  # Not available in American format
                    "ht_away_goals": None,
                    "ht_result": None,
                    "referee": None,
                    "home_shots": None,
                    "away_shots": None,
                    "home_shots_on_target": None,
                    "away_shots_on_target": None,
                    "home_corners": None,
                    "away_corners": None,
                    "home_fouls": None,
                    "away_fouls": None,
                    "home_yellow": None,
                    "away_yellow": None,
                    "home_red": None,
                    "away_red": None,
                }
            else:
                # European format
                match = {
                    "date": row.get("Date"),
                    "home_team": row.get("HomeTeam"),
                    "away_team": row.get("AwayTeam"),
                    "home_goals": row.get("FTHG"),
                    "away_goals": row.get("FTAG"),
                    "result": row.get("FTR"),  # H, D, A
                    "ht_home_goals": row.get("HTHG"),
                    "ht_away_goals": row.get("HTAG"),
                    "ht_result": row.get("HTR"),
                    "referee": row.get("Referee"),
                    "home_shots": row.get("HS"),
                    "away_shots": row.get("AS"),
                    "home_shots_on_target": row.get("HST"),
                    "away_shots_on_target": row.get("AST"),
                    "home_corners": row.get("HC"),
                    "away_corners": row.get("AC"),
                    "home_fouls": row.get("HF"),
                    "away_fouls": row.get("AF"),
                    "home_yellow": row.get("HY"),
                    "away_yellow": row.get("AY"),
                    "home_red": row.get("HR"),
                    "away_red": row.get("AR"),
                }
            
            # Add season info
            if "season" in row:
                match["season"] = row["season"]
            if "league_code" in row:
                match["league_code"] = row["league_code"]
            if "Season" in row:
                match["season"] = row["Season"]
            
            result.append(match)
        
        return pd.DataFrame(result)
    
    def download_all_international(self) -> pd.DataFrame:
        """Download all available international tournament data."""
        all_data = []
        
        # Try World Cup data for various years
        wc_years = [2022, 2018, 2014, 2010, 2006]
        for year in wc_years:
            df = self.download_world_cup_data(year)
            if df is not None:
                df["season"] = year
                df["tournament"] = "World Cup"
                all_data.append(df)
            time.sleep(0.5)
        
        # Try Euro data
        euro_years = [2021, 2016, 2012, 2008]
        for year in euro_years:
            df = self.download_euro_data(year)
            if df is not None:
                df["season"] = year
                df["tournament"] = "Euro"
                all_data.append(df)
            time.sleep(0.5)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def build_training_dataset(self, 
                                leagues: List[str] = None,
                                start_season: int = 2018,
                                end_season: int = 2023) -> pd.DataFrame:
        """Build a comprehensive training dataset from multiple leagues.
        
        Includes European leagues (seasonal CSVs) and American leagues (all seasons in one CSV).
        """
        if leagues is None:
            # All available leagues (European + American + International)
            leagues = [
                # European top 5
                "E0", "SP1", "I1", "D1", "F1",
                # Other European
                "N1", "B1", "P1", "T1", "G1",
                "SC0", "AUS", "DNK", "FIN", "NOR", "SWE", "SWZ", "POL", "ROU", "RUS",
                # American
                "ARG", "BRA", "MEX", "USA", "CHN", "JPN", "IRL",
                # International
                "WC", "EC",
            ]
        
        all_data = []
        
        for league in leagues:
            df = self.download_league_range(league, start_season, end_season)
            if not df.empty:
                standard_df = self.convert_to_standard_format(df)
                if not standard_df.empty:
                    all_data.append(standard_df)
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Remove rows with missing goals
        combined = combined.dropna(subset=["home_goals", "away_goals"])
        
        # Convert date - handle different formats
        if "date" in combined.columns:
            # Try multiple date formats
            combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        
        # Convert goals to numeric
        combined["home_goals"] = pd.to_numeric(combined["home_goals"], errors="coerce")
        combined["away_goals"] = pd.to_numeric(combined["away_goals"], errors="coerce")
        
        return combined


# ===== CLI =====

if __name__ == "__main__":
    scraper = FootballDataScraper()
    
    print("=== FootballDataScraper Test ===")
    
    # Check available columns
    print("\n1. Checking available columns in Premier League 2023-24:")
    cols = scraper.get_available_columns("E0", 2023)
    print(f"Columns: {cols}")
    
    # Download single season
    print("\n2. Downloading Premier League 2023-24:")
    df = scraper.download_csv("E0", 2023)
    if df is not None:
        print(f"Downloaded {len(df)} matches")
        
        # Extract key columns
        key_df = scraper.extract_key_columns(df)
        print(f"\nKey columns sample:")
        print(key_df.head(3))
        
        # Convert to standard format
        standard = scraper.convert_to_standard_format(df)
        print(f"\nStandard format sample:")
        print(standard.head(3))
    
    # Try World Cup
    print("\n3. Trying World Cup 2022:")
    wc_df = scraper.download_world_cup_data(2022)
    if wc_df is not None:
        print(f"World Cup 2022: {len(wc_df)} matches")
    else:
        print("World Cup 2022 data not available")
    
    # Build training dataset
    print("\n4. Building training dataset:")
    training = scraper.build_training_dataset(
        leagues=["E0", "SP1", "I1"],
        start_season=2022,
        end_season=2023
    )
    print(f"Training dataset: {len(training)} matches")
    if not training.empty:
        print(training.head(3))
    
    print("\nFootballDataScraper OK")
