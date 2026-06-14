"""
PlayerStatsService — API-Football v3 Connector
==============================================
Fetches player statistics, team squads, fixtures, and head-to-head data
from API-Football v3. Caches results to SQLite/JSON.

Usage:
    from player_stats_service import PlayerStatsService
    service = PlayerStatsService(api_key="YOUR_KEY")
    players = service.get_team_squad(team_id=33)  # Manchester United
    stats = service.get_player_stats(player_id=909, league=39, season=2023)
"""
import os
import json
import time
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import requests


@dataclass
class PlayerInfo:
    id: int
    name: str
    firstname: str
    lastname: str
    age: Optional[int]
    nationality: str
    height: Optional[str]
    weight: Optional[str]
    injured: bool
    photo: Optional[str]


@dataclass
class PlayerStats:
    player_id: int
    team_id: int
    league_id: int
    season: int
    appearances: int
    lineups: int
    minutes: int
    goals: int
    assists: int
    shots: Optional[int]
    shots_on_target: Optional[int]
    passes: Optional[int]
    key_passes: Optional[int]
    tackles: Optional[int]
    duels_won: Optional[int]
    duels_total: Optional[int]
    dribbles_attempts: Optional[int]
    dribbles_success: Optional[int]
    fouls_drawn: Optional[int]
    fouls_committed: Optional[int]
    yellow_cards: int
    red_cards: int
    penalty_scored: Optional[int]
    penalty_missed: Optional[int]
    rating: Optional[float]
    xG: Optional[float] = None
    xA: Optional[float] = None


class PlayerStatsService:
    BASE_URL = "https://v3.football.api-sports.io"
    CACHE_DIR = Path("data/raw/api-football")
    DB_PATH = Path("data/raw/api-football/cache.db")
    RATE_LIMIT_DELAY = 0.15  # 6 requests per second max (free tier: 10/sec)

    def __init__(self, api_key: Optional[str] = None, season: int = 2026):
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY", "")
        self.season = season
        self.session = requests.Session()
        self.session.headers.update({
            "x-apisports-key": self.api_key,
            "Accept": "application/json"
        })
        self._ensure_cache()

    def _ensure_cache(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                data TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY,
                name TEXT,
                team_id INTEGER,
                league_id INTEGER,
                season INTEGER,
                stats TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        sorted_params = sorted(params.items())
        return f"{endpoint}:{json.dumps(sorted_params, separators=(',', ':'))}"

    def _get_cached(self, key: str, max_age_hours: int = 24) -> Optional[Dict]:
        conn = sqlite3.connect(str(self.DB_PATH))
        cursor = conn.execute(
            "SELECT data, fetched_at FROM cache WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return None
        data, fetched_at = row
        fetched = datetime.fromisoformat(fetched_at)
        if datetime.now() - fetched > timedelta(hours=max_age_hours):
            return None
        return json.loads(data)

    def _set_cached(self, key: str, data: Dict):
        conn = sqlite3.connect(str(self.DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, data, fetched_at) VALUES (?, ?, ?)",
            (key, json.dumps(data), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    def _request(self, endpoint: str, params: Optional[Dict] = None, use_cache: bool = True) -> Dict:
        params = params or {}
        cache_key = self._cache_key(endpoint, params)
        
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        url = f"{self.BASE_URL}/{endpoint}"
        time.sleep(self.RATE_LIMIT_DELAY)
        
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if use_cache:
            self._set_cached(cache_key, data)
        
        return data

    # ===== Team Endpoints =====

    def get_team_squad(self, team_id: int) -> List[PlayerInfo]:
        """Get full squad for a team."""
        data = self._request("players/squads", {"team": team_id})
        players = []
        for team_data in data.get("response", []):
            for player in team_data.get("players", []):
                players.append(PlayerInfo(
                    id=player.get("id"),
                    name=player.get("name"),
                    firstname=player.get("firstname", ""),
                    lastname=player.get("lastname", ""),
                    age=player.get("age"),
                    nationality=player.get("nationality", ""),
                    height=player.get("height"),
                    weight=player.get("weight"),
                    injured=player.get("injured", False),
                    photo=player.get("photo")
                ))
        return players

    def get_team_stats(self, team_id: int, league_id: int, season: Optional[int] = None) -> Dict:
        """Get team statistics for a league/season."""
        season = season or self.season
        return self._request("teams/statistics", {
            "team": team_id,
            "league": league_id,
            "season": season
        })

    # ===== Player Endpoints =====

    def get_player_stats(self, player_id: int, league_id: int, season: Optional[int] = None) -> Optional[PlayerStats]:
        """Get detailed stats for a player in a league/season."""
        season = season or self.season
        data = self._request("players", {
            "id": player_id,
            "league": league_id,
            "season": season
        })
        
        response = data.get("response", [])
        if not response:
            return None
        
        player_data = response[0]
        player = player_data.get("player", {})
        stats_list = player_data.get("statistics", [])
        if not stats_list:
            return None
        
        stats = stats_list[0]
        games = stats.get("games", {})
        shots = stats.get("shots", {})
        passes = stats.get("passes", {})
        tackles = stats.get("tackles", {})
        duels = stats.get("duels", {})
        dribbles = stats.get("dribbles", {})
        fouls = stats.get("fouls", {})
        cards = stats.get("cards", {})
        penalty = stats.get("penalty", {})
        
        return PlayerStats(
            player_id=player_id,
            team_id=stats.get("team", {}).get("id"),
            league_id=league_id,
            season=season,
            appearances=games.get("appearences", 0) or 0,
            lineups=games.get("lineups", 0) or 0,
            minutes=games.get("minutes", 0) or 0,
            goals=games.get("goals", 0) or 0,
            assists=games.get("assists", 0) or 0,
            shots=shots.get("total"),
            shots_on_target=shots.get("on"),
            passes=passes.get("total"),
            key_passes=passes.get("key"),
            tackles=tackles.get("total"),
            duels_won=duels.get("won"),
            duels_total=duels.get("total"),
            dribbles_attempts=dribbles.get("attempts"),
            dribbles_success=dribbles.get("success"),
            fouls_drawn=fouls.get("drawn"),
            fouls_committed=fouls.get("committed"),
            yellow_cards=cards.get("yellow", 0) or 0,
            red_cards=cards.get("red", 0) or 0,
            penalty_scored=penalty.get("scored"),
            penalty_missed=penalty.get("missed"),
            rating=games.get("rating")
        )

    def get_team_top_scorers(self, team_id: int, league_id: int, season: Optional[int] = None, limit: int = 5) -> List[Dict]:
        """Get top scorers for a team in a league/season."""
        season = season or self.season
        data = self._request("players/topscorers", {
            "league": league_id,
            "season": season
        })
        
        scorers = []
        for item in data.get("response", []):
            player = item.get("player", {})
            stats = item.get("statistics", [{}])[0]
            team = stats.get("team", {})
            if team.get("id") == team_id:
                scorers.append({
                    "player_id": player.get("id"),
                    "name": player.get("name"),
                    "goals": stats.get("goals", {}).get("total", 0),
                    "assists": stats.get("goals", {}).get("assists", 0),
                    "rating": stats.get("games", {}).get("rating")
                })
        
        scorers.sort(key=lambda x: x["goals"], reverse=True)
        return scorers[:limit]

    # ===== Fixture Endpoints =====

    def get_fixtures(self, league_id: int, season: Optional[int] = None, 
                     from_date: Optional[str] = None, to_date: Optional[str] = None,
                     team_id: Optional[int] = None, status: Optional[str] = None) -> List[Dict]:
        """Get fixtures for a league/season with optional filters."""
        season = season or self.season
        params = {"league": league_id, "season": season}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if team_id:
            params["team"] = team_id
        if status:
            params["status"] = status
        
        data = self._request("fixtures", params)
        return data.get("response", [])

    def get_fixture_stats(self, fixture_id: int) -> Dict:
        """Get detailed stats for a specific fixture."""
        return self._request("fixtures/statistics", {"fixture": fixture_id})

    def get_head_to_head(self, team1_id: int, team2_id: int, last: int = 10) -> List[Dict]:
        """Get head-to-head history between two teams."""
        data = self._request("fixtures/headtohead", {
            "h2h": f"{team1_id}-{team2_id}",
            "last": last
        })
        return data.get("response", [])

    def get_odds(self, fixture_id: int, bet_type: int = 1) -> Dict:
        """Get odds for a fixture. bet_type=1 is Match Winner."""
        return self._request("odds", {
            "fixture": fixture_id,
            "bet": bet_type
        })

    # ===== League/Season Endpoints =====

    def get_leagues(self, country: Optional[str] = None, 
                    season: Optional[int] = None, 
                    team_id: Optional[int] = None) -> List[Dict]:
        """Get available leagues."""
        params = {}
        if country:
            params["country"] = country
        if season:
            params["season"] = season
        if team_id:
            params["team"] = team_id
        
        data = self._request("leagues", params)
        return data.get("response", [])

    # ===== Batch Operations =====

    def get_team_aggregate_stats(self, team_id: int, league_id: int, 
                                  season: Optional[int] = None) -> Dict[str, Any]:
        """Get aggregated team stats: squad + team stats + top scorers."""
        season = season or self.season
        squad = self.get_team_squad(team_id)
        team_stats = self.get_team_stats(team_id, league_id, season)
        top_scorers = self.get_team_top_scorers(team_id, league_id, season)
        
        # Calculate squad depth metrics
        avg_age = sum(p.age for p in squad if p.age) / max(len([p for p in squad if p.age]), 1)
        injured_count = sum(1 for p in squad if p.injured)
        
        return {
            "team_id": team_id,
            "league_id": league_id,
            "season": season,
            "squad_size": len(squad),
            "avg_age": round(avg_age, 1),
            "injured_count": injured_count,
            "top_scorers": top_scorers,
            "team_stats": team_stats
        }

    def get_fixture_full_context(self, fixture_id: int, 
                                  home_team_id: int, away_team_id: int,
                                  league_id: int) -> Dict[str, Any]:
        """Get full context for a fixture: both teams + H2H + odds."""
        season = self.season
        home_stats = self.get_team_aggregate_stats(home_team_id, league_id, season)
        away_stats = self.get_team_aggregate_stats(away_team_id, league_id, season)
        h2h = self.get_head_to_head(home_team_id, away_team_id)
        odds = self.get_odds(fixture_id)
        
        return {
            "fixture_id": fixture_id,
            "home_team": home_stats,
            "away_team": away_stats,
            "head_to_head": h2h,
            "odds": odds
        }

    def save_to_db(self, player_stats: PlayerStats):
        """Save player stats to SQLite for ML training."""
        conn = sqlite3.connect(str(self.DB_PATH))
        conn.execute("""
            INSERT OR REPLACE INTO players 
            (id, name, team_id, league_id, season, stats, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            player_stats.player_id,
            "",  # name not in PlayerStats, would need lookup
            player_stats.team_id,
            player_stats.league_id,
            player_stats.season,
            json.dumps(asdict(player_stats)),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()


# ===== CLI / Testing =====

if __name__ == "__main__":
    import sys
    
    api_key = os.getenv("API_FOOTBALL_KEY", "")
    if not api_key:
        print("ERROR: Set API_FOOTBALL_KEY environment variable")
        sys.exit(1)
    
    service = PlayerStatsService(api_key=api_key, season=2026)
    
    # Test: Get World Cup 2026 league info
    print("=== Leagues ===")
    leagues = service.get_leagues(season=2026)
    wc_leagues = [l for l in leagues if "world cup" in l.get("league", {}).get("name", "").lower()]
    for l in wc_leagues[:5]:
        print(f"  {l['league']['id']}: {l['league']['name']} ({l['country']['name']})")
    
    # Test: Get Argentina squad (team_id=26 is Argentina in API-Football)
    print("\n=== Argentina Squad ===")
    squad = service.get_team_squad(team_id=26)
    for p in squad[:5]:
        print(f"  {p.name} (Age: {p.age}, Injured: {p.injured})")
    
    print(f"\nTotal squad players: {len(squad)}")
    print("PlayerStatsService OK")
