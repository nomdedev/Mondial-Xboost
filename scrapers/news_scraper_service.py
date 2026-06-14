"""
NewsScraperService — Selenium + Brave Browser
==============================================
Scrapes news from multiple sources about football injuries, tactics,
team news, and match previews. Uses Brave browser for session cookies.

Sources:
- ESPN Soccer Injury Tracker
- TalkSport World Cup Injury Tracker
- BBC Sport Football
- Marca (Spanish)
- AS (Spanish)

Usage:
    from news_scraper_service import NewsScraperService
    scraper = NewsScraperService()
    articles = scraper.scrape_all()
    for article in articles:
        print(article.title, article.url)
"""
import os
import re
import json
import time
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup


@dataclass
class NewsArticle:
    source: str
    url: str
    title: str
    content: str
    published_at: Optional[str]
    mentions_teams: List[str]
    mentions_players: List[str]
    injury_keywords: List[str]
    sentiment: Optional[float] = None  # Populated by LLM later
    relevance_score: float = 0.0  # 0-1, calculated by keyword matching


class NewsScraperService:
    CACHE_DIR = Path("data/raw/news")
    DB_PATH = Path("data/raw/news/cache.db")
    BRAVE_USER_DATA = Path(os.path.expanduser("~/AppData/Local/BraveSoftware/Brave-Browser/User Data"))
    
    SOURCES = {
        "espn_injuries": {
            "url": "https://www.espn.com/soccer/story/_/id/48572979/2026-fifa-world-cup-injuries-tracker-which-stars-miss-latest-info",
            "type": "injury_tracker",
            "priority": 1
        },
        "talksport_injuries": {
            "url": "https://talksport.com/football/world-cup/4311921/world-cup-2026-injury-tracker-full-squads-messi/",
            "type": "injury_tracker",
            "priority": 1
        },
        "bbc_football": {
            "url": "https://www.bbc.com/sport/football",
            "type": "news_feed",
            "priority": 2
        },
        "marca_football": {
            "url": "https://www.marca.com/futbol.html",
            "type": "news_feed",
            "priority": 2
        }
    }
    
    INJURY_KEYWORDS = [
        "injury", "injured", "sidelined", "ruled out", "doubtful",
        "fitness concern", "knock", "strain", "sprain", "fracture",
        "surgery", "recovery", "rehabilitation", "muscle",
        "lesión", "lesionado", "duda", "baja", "recuperación",
        "cirugía", "muscular", "tobillo", "rodilla"
    ]
    
    WC2026_TEAMS = [
        "argentina", "brazil", "france", "england", "spain", "germany",
        "portugal", "netherlands", "belgium", "italy", "uruguay",
        "colombia", "mexico", "usa", "canada", "japan", "south korea",
        "australia", "morocco", "senegal", "nigeria", "egypt",
        "croatia", "denmark", "switzerland", "poland", "serbia",
        "ecuador", "chile", "peru", "paraguay", "venezuela",
        "panama", "costa rica", "honduras", "jamaica", "qatar",
        "saudi arabia", "iran", "iraq", "uae", "uzbekistan",
        "china", "indonesia", "thailand", "new zealand"
    ]

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver: Optional[webdriver.Chrome] = None
        self._ensure_cache()

    def _ensure_cache(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                published_at TEXT,
                mentions_teams TEXT,
                mentions_players TEXT,
                injury_keywords TEXT,
                relevance_score REAL,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _init_driver(self):
        """Initialize Brave browser with copied user data."""
        options = Options()
        
        if self.headless:
            options.add_argument("--headless=new")
        
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Use Brave if available
        brave_path = Path(os.path.expanduser("~")) / "AppData/Local/BraveSoftware/Brave-Browser/Application/brave.exe"
        if brave_path.exists():
            options.binary_location = str(brave_path)
            
            # Copy user data to temp to avoid locking
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="brave_news_")
            user_data_copy = Path(temp_dir) / "User Data"
            if self.BRAVE_USER_DATA.exists():
                import shutil
                shutil.copytree(self.BRAVE_USER_DATA, user_data_copy, dirs_exist_ok=True)
                options.add_argument(f"--user-data-dir={user_data_copy}")
        
        # Try to find chromedriver
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
        })
        
        return self.driver

    def _close_driver(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

    def _fetch_page(self, url: str, wait_for: Optional[str] = None, timeout: int = 20) -> str:
        """Fetch page with Selenium, optionally waiting for element."""
        if not self.driver:
            self._init_driver()
        
        self.driver.get(url)
        
        if wait_for:
            try:
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
                )
            except TimeoutException:
                pass
        
        # Scroll to load lazy content
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        return self.driver.page_source

    def _parse_injury_tracker(self, html: str, source: str, url: str) -> List[NewsArticle]:
        """Parse injury tracker pages (ESPN, TalkSport)."""
        soup = BeautifulSoup(html, 'lxml')
        articles = []
        
        # ESPN uses article body with paragraphs
        if "espn" in source:
            content_div = soup.find('div', class_=re.compile('article-body|story-body'))
            if content_div:
                paragraphs = content_div.find_all('p')
                content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
                title = soup.find('h1')
                title = title.get_text(strip=True) if title else "ESPN Injury Tracker"
                
                articles.append(self._create_article(
                    source="espn_injuries", url=url, title=title,
                    content=content, published_at=None
                ))
        
        # TalkSport uses similar structure
        elif "talksport" in source:
            article = soup.find('article') or soup.find('div', class_=re.compile('article|content'))
            if article:
                paragraphs = article.find_all('p')
                content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
                title = soup.find('h1')
                title = title.get_text(strip=True) if title else "TalkSport Injury Tracker"
                
                articles.append(self._create_article(
                    source="talksport_injuries", url=url, title=title,
                    content=content, published_at=None
                ))
        
        return articles

    def _parse_news_feed(self, html: str, source: str, url: str) -> List[NewsArticle]:
        """Parse news feed pages (BBC, Marca)."""
        soup = BeautifulSoup(html, 'lxml')
        articles = []
        
        # BBC uses promo cards
        if "bbc" in source:
            promos = soup.find_all('div', class_=re.compile('promo|card'))
            for promo in promos[:10]:  # Top 10 articles
                link = promo.find('a', href=True)
                if not link:
                    continue
                
                article_url = urljoin(url, link['href'])
                title_elem = promo.find(['h2', 'h3', 'span'], class_=re.compile('title|headline'))
                title = title_elem.get_text(strip=True) if title_elem else "BBC Article"
                
                # Fetch article content
                try:
                    article_html = self._fetch_page(article_url, wait_for='article', timeout=10)
                    article_soup = BeautifulSoup(article_html, 'lxml')
                    content_div = article_soup.find('article') or article_soup.find('div', class_=re.compile('story-body'))
                    if content_div:
                        paragraphs = content_div.find_all('p')
                        content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
                    else:
                        content = ""
                except Exception:
                    content = ""
                
                articles.append(self._create_article(
                    source="bbc_football", url=article_url, title=title,
                    content=content, published_at=None
                ))
        
        # Marca uses article cards
        elif "marca" in source:
            cards = soup.find_all('article', class_=re.compile('news-item|card'))
            for card in cards[:10]:
                link = card.find('a', href=True)
                if not link:
                    continue
                
                article_url = urljoin(url, link['href'])
                title_elem = card.find(['h2', 'h3', 'h4'])
                title = title_elem.get_text(strip=True) if title_elem else "Marca Article"
                
                try:
                    article_html = self._fetch_page(article_url, timeout=10)
                    article_soup = BeautifulSoup(article_html, 'lxml')
                    content_div = article_soup.find('div', class_=re.compile('article-content|news-text'))
                    if content_div:
                        paragraphs = content_div.find_all('p')
                        content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
                    else:
                        content = ""
                except Exception:
                    content = ""
                
                articles.append(self._create_article(
                    source="marca_football", url=article_url, title=title,
                    content=content, published_at=None
                ))
        
        return articles

    def _create_article(self, source: str, url: str, title: str, 
                        content: str, published_at: Optional[str]) -> NewsArticle:
        """Create article with keyword extraction."""
        content_lower = content.lower()
        title_lower = title.lower()
        
        # Find mentioned teams
        mentions_teams = [
            team for team in self.WC2026_TEAMS
            if team.lower() in content_lower or team.lower() in title_lower
        ]
        
        # Find injury keywords
        injury_keywords = [
            kw for kw in self.INJURY_KEYWORDS
            if kw.lower() in content_lower or kw.lower() in title_lower
        ]
        
        # Calculate relevance score
        relevance = 0.0
        if injury_keywords:
            relevance += 0.4
        if mentions_teams:
            relevance += 0.3
        if any(word in title_lower for word in ["world cup", "mundial", "injury", "lesión"]):
            relevance += 0.3
        
        return NewsArticle(
            source=source,
            url=url,
            title=title,
            content=content[:5000],  # Limit content length
            published_at=published_at,
            mentions_teams=mentions_teams,
            mentions_players=[],  # Extracted by LLM later
            injury_keywords=injury_keywords,
            relevance_score=min(relevance, 1.0)
        )

    def scrape_source(self, source_key: str) -> List[NewsArticle]:
        """Scrape a single source."""
        config = self.SOURCES.get(source_key)
        if not config:
            return []
        
        print(f"Scraping {source_key}...")
        
        try:
            html = self._fetch_page(config["url"], timeout=20)
            
            if config["type"] == "injury_tracker":
                articles = self._parse_injury_tracker(html, source_key, config["url"])
            else:
                articles = self._parse_news_feed(html, source_key, config["url"])
            
            print(f"  Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return []

    def scrape_all(self, sources: Optional[List[str]] = None) -> List[NewsArticle]:
        """Scrape all or selected sources."""
        sources = sources or list(self.SOURCES.keys())
        all_articles = []
        
        try:
            for source_key in sources:
                articles = self.scrape_source(source_key)
                all_articles.extend(articles)
                time.sleep(2)  # Rate limiting between sources
        finally:
            self._close_driver()
        
        # Sort by relevance
        all_articles.sort(key=lambda a: a.relevance_score, reverse=True)
        return all_articles

    def save_articles(self, articles: List[NewsArticle]):
        """Save articles to SQLite."""
        conn = sqlite3.connect(str(self.DB_PATH))
        
        for article in articles:
            conn.execute("""
                INSERT OR REPLACE INTO articles 
                (source, url, title, content, published_at, mentions_teams, 
                 mentions_players, injury_keywords, relevance_score, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article.source,
                article.url,
                article.title,
                article.content,
                article.published_at,
                json.dumps(article.mentions_teams),
                json.dumps(article.mentions_players),
                json.dumps(article.injury_keywords),
                article.relevance_score,
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
        print(f"Saved {len(articles)} articles to database")

    def get_recent_articles(self, hours: int = 24, min_relevance: float = 0.5) -> List[NewsArticle]:
        """Get recent articles from database."""
        since = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(str(self.DB_PATH))
        cursor = conn.execute("""
            SELECT source, url, title, content, published_at, mentions_teams,
                   mentions_players, injury_keywords, relevance_score
            FROM articles
            WHERE scraped_at > ? AND relevance_score >= ?
            ORDER BY relevance_score DESC
        """, (since.isoformat(), min_relevance))
        
        articles = []
        for row in cursor.fetchall():
            articles.append(NewsArticle(
                source=row[0],
                url=row[1],
                title=row[2],
                content=row[3],
                published_at=row[4],
                mentions_teams=json.loads(row[5]) if row[5] else [],
                mentions_players=json.loads(row[6]) if row[6] else [],
                injury_keywords=json.loads(row[7]) if row[7] else [],
                relevance_score=row[8]
            ))
        
        conn.close()
        return articles

    def export_to_json(self, articles: List[NewsArticle], filepath: Optional[str] = None) -> str:
        """Export articles to JSON for LLM processing."""
        filepath = filepath or f"data/raw/news/articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [asdict(a) for a in articles]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(path)


# ===== CLI =====

if __name__ == "__main__":
    import sys
    
    print("=== NewsScraperService Test ===")
    scraper = NewsScraperService(headless=True)
    
    # Test with ESPN only (most reliable)
    articles = scraper.scrape_all(sources=["espn_injuries", "talksport_injuries"])
    
    print(f"\nTotal articles: {len(articles)}")
    for a in articles[:5]:
        print(f"\n[{a.relevance_score:.2f}] {a.source}: {a.title}")
        print(f"  Teams: {', '.join(a.mentions_teams) if a.mentions_teams else 'None'}")
        print(f"  Injuries: {', '.join(a.injury_keywords) if a.injury_keywords else 'None'}")
        print(f"  Content: {a.content[:200]}...")
    
    # Save to DB
    scraper.save_articles(articles)
    
    # Export to JSON
    json_path = scraper.export_to_json(articles)
    print(f"\nExported to: {json_path}")
    
    print("\nNewsScraperService OK")
