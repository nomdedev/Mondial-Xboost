"""
FlashscoreScraper — Recent Results & Team Form (ESPN Fix)
=========================================================
ESPN returns 400/403. Use alternative sources:
- FIFA.com for official match data
- National-Football-Teams.com for international matches
- WorldFootball.net for comprehensive results
"""
import re
import json
import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time


class FlashscoreScraper:
    """
    Scraper using multiple sources for recent form and H2H data.
    
    Sources:
    1. WorldFootball.net - comprehensive match database
    2. National-Football-Teams.com - international matches
    3. FIFA.com - official FIFA data
    """
    
    WORLDFOOTBALL_URL = "https://www.worldfootball.net"
    NFT_URL = "https://www.national-football-teams.com"
    FIFA_URL = "https://www.fifa.com"
    
    # Team slug mappings
    TEAM_SLUGS = {
        "Argentina": "argentina",
        "Brazil": "brazil",
        "Germany": "germany",
        "France": "france",
        "Spain": "spain",
        "England": "england",
        "Italy": "italy",
        "Portugal": "portugal",
        "Netherlands": "netherlands",
        "Belgium": "belgium",
        "Uruguay": "uruguay",
        "Croatia": "croatia",
        "Mexico": "mexico",
        "USA": "united-states",
        "Japan": "japan",
        "South Korea": "south-korea",
        "Senegal": "senegal",
        "Morocco": "morocco",
        "Switzerland": "switzerland",
        "Poland": "poland",
        "Wales": "wales",
        "Australia": "australia",
        "Canada": "canada",
        "Ecuador": "ecuador",
        "Cameroon": "cameroon",
        "Ghana": "ghana",
        "Saudi Arabia": "saudi-arabia",
        "Iran": "iran",
        "Tunisia": "tunisia",
        "Qatar": "qatar",
        "Costa Rica": "costa-rica",
    }
    
    def __init__(self, cache_dir: str = "data/raw/flashscore"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
    
    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse page with caching."""
        cache_file = self.cache_dir / f"{url.replace('/', '_').replace(':', '_')[:100]}.html"
        
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return BeautifulSoup(f.read(), "lxml")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            return BeautifulSoup(response.text, "lxml")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def get_worldfootball_team_form(self, team_name: str, n: int = 5) -> List[Dict]:
        """Get team form from WorldFootball.net."""
        slug = self.TEAM_SLUGS.get(team_name)
        if not slug:
            print(f"Slug not found for: {team_name}")
            return []
        
        url = f"{self.WORLDFOOTBALL_URL}/teams/{slug}/10/"
        soup = self._get_page(url)
        
        if not soup:
            return []
        
        matches = []
        
        # WorldFootball uses table with class "standard_tabelle"
        tables = soup.find_all("table", {"class": "standard_tabelle"})
        
        for table in tables:
            rows = table.find_all("tr")
            
            for row in rows[1:]:  # Skip header
                cells = row.find_all(["td", "th"])
                if len(cells) >= 5:
                    try:
                        date = cells[0].get_text(strip=True)
                        competition = cells[1].get_text(strip=True)
                        home_team = cells[2].get_text(strip=True)
                        score = cells[3].get_text(strip=True)
                        away_team = cells[4].get_text(strip=True)
                        
                        # Parse score
                        score_match = re.match(r"(\d+):(\d+)", score)
                        if score_match:
                            home_goals = int(score_match.group(1))
                            away_goals = int(score_match.group(2))
                            
                            # Determine result for queried team
                            if team_name.lower() in home_team.lower():
                                venue = "home"
                                team_goals = home_goals
                                opponent_goals = away_goals
                            else:
                                venue = "away"
                                team_goals = away_goals
                                opponent_goals = home_goals
                            
                            if team_goals > opponent_goals:
                                result = "W"
                            elif team_goals < opponent_goals:
                                result = "L"
                            else:
                                result = "D"
                            
                            matches.append({
                                "date": date,
                                "competition": competition,
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_goals": home_goals,
                                "away_goals": away_goals,
                                "score": score,
                                "result": result,
                                "venue": venue,
                                "team_goals": team_goals,
                                "opponent_goals": opponent_goals,
                            })
                            
                            if len(matches) >= n:
                                return matches
                    except Exception:
                        continue
        
        return matches
    
    def get_nft_team_form(self, team_name: str, n: int = 5) -> List[Dict]:
        """Get team form from National-Football-Teams.com."""
        slug = self.TEAM_SLUGS.get(team_name)
        if not slug:
            return []
        
        url = f"{self.NFT_URL}/country/{slug}/"
        soup = self._get_page(url)
        
        if not soup:
            return []
        
        matches = []
        
        # Find match history tables
        tables = soup.find_all("table", {"class": "table"})
        
        for table in tables:
            caption = table.find("caption")
            if caption and "matches" in caption.get_text().lower():
                rows = table.find_all("tr")
                
                for row in rows[1:]:
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 5:
                        try:
                            match = {
                                "date": cells[0].get_text(strip=True),
                                "venue": cells[1].get_text(strip=True),
                                "opponent": cells[2].get_text(strip=True),
                                "score": cells[3].get_text(strip=True),
                                "competition": cells[4].get_text(strip=True),
                            }
                            
                            # Parse score
                            score_match = re.match(r"(\d+):(\d+)", match["score"])
                            if score_match:
                                match["goals_for"] = int(score_match.group(1))
                                match["goals_against"] = int(score_match.group(2))
                                
                                if match["goals_for"] > match["goals_against"]:
                                    match["result"] = "W"
                                elif match["goals_for"] < match["goals_against"]:
                                    match["result"] = "L"
                                else:
                                    match["result"] = "D"
                            
                            matches.append(match)
                            
                            if len(matches) >= n:
                                return matches
                        except Exception:
                            continue
        
        return matches
    
    def get_worldfootball_h2h(self, team1: str, team2: str, n: int = 5) -> List[Dict]:
        """Get head-to-head from WorldFootball.net."""
        slug1 = self.TEAM_SLUGS.get(team1)
        slug2 = self.TEAM_SLUGS.get(team2)
        
        if not slug1 or not slug2:
            return []
        
        url = f"{self.WORLDFOOTBALL_URL}/teams/{slug1}/vs/{slug2}/"
        soup = self._get_page(url)
        
        if not soup:
            return []
        
        matches = []
        
        tables = soup.find_all("table", {"class": "standard_tabelle"})
        
        for table in tables:
            rows = table.find_all("tr")
            
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 5:
                    try:
                        match = {
                            "date": cells[0].get_text(strip=True),
                            "competition": cells[1].get_text(strip=True),
                            "home_team": cells[2].get_text(strip=True),
                            "score": cells[3].get_text(strip=True),
                            "away_team": cells[4].get_text(strip=True),
                        }
                        
                        # Parse score
                        score_match = re.match(r"(\d+):(\d+)", match["score"])
                        if score_match:
                            match["home_goals"] = int(score_match.group(1))
                            match["away_goals"] = int(score_match.group(2))
                        
                        matches.append(match)
                        
                        if len(matches) >= n:
                            return matches
                    except Exception:
                        continue
        
        return matches
    
    def calculate_team_form_score(self, team_name: str, n: int = 5) -> Dict:
        """Calculate form metrics for a team."""
        # Try WorldFootball first, then NFT
        matches = self.get_worldfootball_team_form(team_name, n)
        
        if not matches:
            matches = self.get_nft_team_form(team_name, n)
        
        if not matches:
            return {
                "matches_played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "form_score": 0.0,
                "form_string": "",
                "win_rate": 0.0,
                "points_per_game": 0.0,
            }
        
        wins = sum(1 for m in matches if m.get("result") == "W")
        draws = sum(1 for m in matches if m.get("result") == "D")
        losses = sum(1 for m in matches if m.get("result") == "L")
        
        goals_for = sum(m.get("team_goals", m.get("goals_for", 0)) for m in matches)
        goals_against = sum(m.get("opponent_goals", m.get("goals_against", 0)) for m in matches)
        
        # Weighted form score (recent matches weighted more)
        weights = [1.0, 0.9, 0.8, 0.7, 0.6][:len(matches)]
        form_score = sum(
            weights[i] * (1 if m.get("result") == "W" else 0.5 if m.get("result") == "D" else 0)
            for i, m in enumerate(matches)
        ) / sum(weights)
        
        form_string = "".join(m.get("result", "?") for m in matches)
        
        return {
            "matches_played": len(matches),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_difference": goals_for - goals_against,
            "form_score": round(form_score, 3),
            "form_string": form_string,
            "win_rate": round(wins / len(matches), 3) if matches else 0,
            "points_per_game": round((wins * 3 + draws) / len(matches), 3) if matches else 0,
        }
    
    def get_all_team_forms(self, teams: List[str], n: int = 5) -> Dict[str, Dict]:
        """Get form for multiple teams."""
        results = {}
        
        for team in teams:
            print(f"Getting form for {team}...")
            results[team] = self.calculate_team_form_score(team, n)
            time.sleep(1)  # Rate limiting
        
        return results


# ===== CLI =====

if __name__ == "__main__":
    scraper = FlashscoreScraper()
    
    print("=== FlashscoreScraper Test (WorldFootball + NFT) ===")
    
    # Test WorldFootball team form
    print("\n1. Argentina form from WorldFootball:")
    form = scraper.get_worldfootball_team_form("Argentina", n=5)
    print(f"Found {len(form)} matches")
    for match in form[:3]:
        print(f"  {match}")
    
    # Calculate form score
    if form:
        print("\n2. Argentina form score:")
        score = scraper.calculate_team_form_score("Argentina", n=5)
        print(f"Form score: {score}")
    
    # Test WorldFootball H2H
    print("\n3. Argentina vs Brazil H2H from WorldFootball:")
    h2h = scraper.get_worldfootball_h2h("Argentina", "Brazil", n=5)
    print(f"Found {len(h2h)} matches")
    for match in h2h[:3]:
        print(f"  {match}")
    
    # Test NFT
    print("\n4. Argentina form from National-Football-Teams:")
    nft_form = scraper.get_nft_team_form("Argentina", n=5)
    print(f"Found {len(nft_form)} matches")
    for match in nft_form[:3]:
        print(f"  {match}")
    
    print("\nFlashscoreScraper OK")
