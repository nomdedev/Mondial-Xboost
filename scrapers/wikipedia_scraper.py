"""
WikipediaScraper — World Cup Historical Data
============================================
Scrapes World Cup historical data from Wikipedia.
Provides: results, squads, match details, group tables.

Usage:
    from wikipedia_scraper import WikipediaScraper
    scraper = WikipediaScraper()
    
    # Get World Cup 2022 results
    results = scraper.get_world_cup_results(2022)
    
    # Get all historical World Cup results
    all_results = scraper.get_all_world_cup_results()
"""
import re
import json
import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup


class WikipediaScraper:
    """
    Scraper for Wikipedia football data.
    
    Wikipedia has comprehensive World Cup data:
    - Match results (with scorers, attendance, referee)
    - Group tables
    - Knockout brackets
    - Squads (players, ages, caps, clubs)
    - Statistics (top scorers, assists, etc.)
    
    URLs:
    - 2022 WC: https://en.wikipedia.org/wiki/2022_FIFA_World_Cup
    - 2018 WC: https://en.wikipedia.org/wiki/2018_FIFA_World_Cup
    - etc.
    """
    
    BASE_URL = "https://en.wikipedia.org/wiki"
    CACHE_DIR = Path("data/raw/wikipedia")
    
    WORLD_CUP_YEARS = [2022, 2018, 2014, 2010, 2006, 2002, 1998, 1994, 1990, 1986, 1982, 1978, 1974, 1970, 1966, 1962, 1958, 1954, 1950, 1938, 1934, 1930]
    
    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Oloraculo xBoost/1.0 (Research Project)"
        })
    
    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse Wikipedia page."""
        cache_file = self.CACHE_DIR / f"{url.split('/')[-1]}.html"
        
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return BeautifulSoup(f.read(), "lxml")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Save cache
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            return BeautifulSoup(response.text, "lxml")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def get_world_cup_results(self, year: int) -> pd.DataFrame:
        """Get match results for a specific World Cup."""
        url = f"{self.BASE_URL}/{year}_FIFA_World_Cup"
        soup = self._get_page(url)
        
        if not soup:
            return pd.DataFrame()
        
        matches = []
        
        # Find match result tables
        # Wikipedia uses specific table classes for match results
        tables = soup.find_all("table", {"class": "wikitable"})
        
        for table in tables:
            # Check if this is a match results table
            caption = table.find("caption")
            if caption and "match" in caption.get_text().lower():
                rows = table.find_all("tr")
                
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 4:
                        try:
                            match = self._parse_match_row(cells, year)
                            if match:
                                matches.append(match)
                        except Exception as e:
                            continue
        
        # Also try to find match summaries in the page structure
        # Wikipedia sometimes uses different formats
        match_sections = soup.find_all("div", {"class": "footballbox"})
        
        for section in match_sections:
            try:
                match = self._parse_footballbox(section, year)
                if match:
                    matches.append(match)
            except Exception:
                continue
        
        if not matches:
            print(f"No matches found for World Cup {year}")
            return pd.DataFrame()
        
        df = pd.DataFrame(matches)
        df["tournament"] = "World Cup"
        df["year"] = year
        
        return df
    
    def _parse_match_row(self, cells, year: int) -> Optional[Dict]:
        """Parse a match row from Wikipedia table."""
        texts = [cell.get_text(strip=True) for cell in cells]
        
        if len(texts) < 4:
            return None
        
        # Try to extract date, teams, and score
        # Format varies, but usually: Date | Team1 | Score | Team2
        
        date_str = texts[0] if texts[0] else None
        home_team = texts[1] if len(texts) > 1 else None
        score = texts[2] if len(texts) > 2 else None
        away_team = texts[3] if len(texts) > 3 else None
        
        if not all([home_team, away_team, score]):
            return None
        
        # Parse score
        score_match = re.match(r"(\d+)\s*[-–]\s*(\d+)", score)
        if score_match:
            home_goals = int(score_match.group(1))
            away_goals = int(score_match.group(2))
        else:
            return None
        
        return {
            "date": date_str,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "score": score,
        }
    
    def _parse_footballbox(self, section, year: int) -> Optional[Dict]:
        """Parse a footballbox div from Wikipedia."""
        # footballbox is Wikipedia's modern match display format
        
        # Get teams
        teams = section.find_all("th", {"class": "fhome"})
        if not teams:
            teams = section.find_all("th", {"class": "faway"})
        
        # Get score
        score_elem = section.find("th", {"class": "fscore"})
        
        # Get date
        date_elem = section.find("div", {"class": "fdate"})
        
        if not score_elem:
            return None
        
        score_text = score_elem.get_text(strip=True)
        score_match = re.match(r"(\d+)\s*[-–]\s*(\d+)", score_text)
        
        if not score_match:
            return None
        
        home_team = ""
        away_team = ""
        
        # Try to find team names
        team_headers = section.find_all("th")
        for th in team_headers:
            if "fhome" in th.get("class", []):
                home_team = th.get_text(strip=True)
            elif "faway" in th.get("class", []):
                away_team = th.get_text(strip=True)
        
        return {
            "date": date_elem.get_text(strip=True) if date_elem else None,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": int(score_match.group(1)),
            "away_goals": int(score_match.group(2)),
            "score": score_text,
        }
    
    def get_world_cup_squads(self, year: int) -> Dict[str, List[Dict]]:
        """Get squad information for all teams in a World Cup."""
        url = f"{self.BASE_URL}/{year}_FIFA_World_Cup_squads"
        soup = self._get_page(url)
        
        if not soup:
            return {}
        
        squads = {}
        
        # Find squad sections
        sections = soup.find_all("div", {"class": "mw-parser-output"})
        
        for section in sections:
            # Find team headers (usually h3 or h4)
            headers = section.find_all(["h3", "h4"])
            
            for header in headers:
                team_name = header.get_text(strip=True)
                
                # Find table after header
                table = header.find_next("table", {"class": "wikitable"})
                if table:
                    players = []
                    rows = table.find_all("tr")[1:]  # Skip header
                    
                    for row in rows:
                        cells = row.find_all(["td", "th"])
                        if len(cells) >= 3:
                            player = {
                                "number": cells[0].get_text(strip=True) if len(cells) > 0 else "",
                                "position": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                                "name": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                                "age": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                                "caps": cells[4].get_text(strip=True) if len(cells) > 4 else "",
                                "club": cells[5].get_text(strip=True) if len(cells) > 5 else "",
                            }
                            players.append(player)
                    
                    squads[team_name] = players
        
        return squads
    
    def get_world_cup_statistics(self, year: int) -> Dict:
        """Get tournament statistics (top scorers, assists, etc.)."""
        url = f"{self.BASE_URL}/{year}_FIFA_World_Cup_statistics"
        soup = self._get_page(url)
        
        if not soup:
            return {}
        
        stats = {
            "top_scorers": [],
            "top_assists": [],
            "best_players": [],
        }
        
        # Find statistics tables
        tables = soup.find_all("table", {"class": "wikitable"})
        
        for table in tables:
            caption = table.find("caption")
            if caption:
                caption_text = caption.get_text().lower()
                
                if "goal" in caption_text or "scorer" in caption_text:
                    rows = table.find_all("tr")[1:]
                    for row in rows[:10]:  # Top 10
                        cells = row.find_all("td")
                        if len(cells) >= 2:
                            stats["top_scorers"].append({
                                "player": cells[0].get_text(strip=True),
                                "goals": cells[1].get_text(strip=True),
                            })
                
                elif "assist" in caption_text:
                    rows = table.find_all("tr")[1:]
                    for row in rows[:10]:
                        cells = row.find_all("td")
                        if len(cells) >= 2:
                            stats["top_assists"].append({
                                "player": cells[0].get_text(strip=True),
                                "assists": cells[1].get_text(strip=True),
                            })
        
        return stats
    
    def get_all_world_cup_results(self) -> pd.DataFrame:
        """Get all historical World Cup results."""
        all_matches = []
        
        for year in self.WORLD_CUP_YEARS:
            print(f"Fetching World Cup {year}...")
            df = self.get_world_cup_results(year)
            if not df.empty:
                all_matches.append(df)
            
            # Rate limiting
            import time
            time.sleep(1)
        
        if not all_matches:
            return pd.DataFrame()
        
        return pd.concat(all_matches, ignore_index=True)
    
    def get_fifa_rankings(self) -> pd.DataFrame:
        """Get current FIFA rankings from Wikipedia."""
        url = f"{self.BASE_URL}/FIFA_Men%27s_World_Rankings"
        soup = self._get_page(url)
        
        if not soup:
            return pd.DataFrame()
        
        # Find rankings table
        tables = soup.find_all("table", {"class": "wikitable"})
        
        for table in tables:
            caption = table.find("caption")
            if caption and "ranking" in caption.get_text().lower():
                rankings = []
                rows = table.find_all("tr")[1:]
                
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 3:
                        rankings.append({
                            "rank": cells[0].get_text(strip=True),
                            "team": cells[1].get_text(strip=True),
                            "points": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                        })
                
                return pd.DataFrame(rankings)
        
        return pd.DataFrame()
    
    def convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Wikipedia format to standard match format."""
        if df.empty:
            return df
        
        result = []
        
        for _, row in df.iterrows():
            match = {
                "date": row.get("date"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "home_goals": row.get("home_goals"),
                "away_goals": row.get("away_goals"),
                "tournament": row.get("tournament", "World Cup"),
                "year": row.get("year"),
                "venue": "neutral",  # World Cup matches are neutral
            }
            result.append(match)
        
        return pd.DataFrame(result)


# ===== CLI =====

if __name__ == "__main__":
    scraper = WikipediaScraper()
    
    print("=== WikipediaScraper Test ===")
    
    # Test World Cup 2022
    print("\n1. World Cup 2022 results:")
    wc_2022 = scraper.get_world_cup_results(2022)
    if not wc_2022.empty:
        print(f"Found {len(wc_2022)} matches")
        print(wc_2022.head(5).to_string())
    else:
        print("No matches found (Wikipedia format may have changed)")
    
    # Test World Cup 2018
    print("\n2. World Cup 2018 results:")
    wc_2018 = scraper.get_world_cup_results(2018)
    if not wc_2018.empty:
        print(f"Found {len(wc_2018)} matches")
        print(wc_2018.head(3).to_string())
    
    # Test squads
    print("\n3. World Cup 2022 squads (Argentina):")
    squads = scraper.get_world_cup_squads(2022)
    if squads:
        argentina_squad = squads.get("Argentina", [])
        print(f"Argentina squad: {len(argentina_squad)} players")
        for player in argentina_squad[:5]:
            print(f"  {player}")
    
    # Test statistics
    print("\n4. World Cup 2022 statistics:")
    stats = scraper.get_world_cup_statistics(2022)
    if stats:
        print(f"Top scorers: {stats.get('top_scorers', [])[:3]}")
    
    print("\nWikipediaScraper OK")
