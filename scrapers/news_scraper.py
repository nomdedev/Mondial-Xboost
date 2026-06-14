"""
News scraper for football availability/injury news.

Supports multiple sources and outputs structured claims about player availability.
For now it focuses on text-based extraction; LLM classification can be layered on top.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup


RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "news"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass
class NewsArticle:
    source: str
    url: str
    title: str
    content: str
    fetched_at: str
    relevance_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AvailabilityClaim:
    player_name: str
    team_name: str
    status: str  # available, doubt, injured, suspended
    severity: int  # 0-10
    source: str
    url: str
    reported_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class NewsScraper:
    """Scrape football news from configured sources."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout

    def fetch_html(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def parse_generic_article(self, source: str, url: str, html: str) -> NewsArticle:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        # Remove script/style
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Compact whitespace
        text = re.sub(r"\n+", "\n", text)
        return NewsArticle(
            source=source,
            url=url,
            title=title,
            content=text[:8000],
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

    def scrape_source(self, source: str, url: str) -> NewsArticle:
        html = self.fetch_html(url)
        article = self.parse_generic_article(source, url, html)
        article.relevance_score = self._score_relevance(article)
        return article

    def scrape_sources(self, sources: dict[str, str]) -> list[NewsArticle]:
        articles = []
        for name, url in sources.items():
            try:
                articles.append(self.scrape_source(name, url))
            except Exception as ex:
                print(f"Failed to scrape {name}: {ex}")
        return articles

    def _score_relevance(self, article: NewsArticle) -> float:
        """Simple keyword-based relevance score for injury/availability content."""
        text = (article.title + " " + article.content).lower()
        keywords = [
            "injury", "injured", "doubt", "availability", "suspended", "red card",
            "fitness", "knock", "strain", "sprain", "fracture", "ruled out",
            "lesión", "lesionado", "duda", "sancionado", "tarjeta roja",
        ]
        score = sum(1 for kw in keywords if kw in text) / len(keywords)
        return min(score * 3, 1.0)

    def save_articles(self, articles: Iterable[NewsArticle], name: str | None = None) -> Path:
        if name is None:
            name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = RAW_DIR / f"{name}.json"
        data = [a.to_dict() for a in articles]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def get_default_sources() -> dict[str, str]:
    """Return default news sources."""
    return {
        "espn_injuries": "https://www.espn.com/soccer/story/_/id/48572979/2026-fifa-world-cup-injuries-tracker-which-stars-miss-latest-info",
        "talksport_injuries": "https://talksport.com/football/world-cup/4311921/world-cup-2026-injury-tracker-full-squads-messi/",
    }


def main():
    scraper = NewsScraper()
    sources = get_default_sources()
    print(f"Scraping {len(sources)} sources...")
    articles = scraper.scrape_sources(sources)
    path = scraper.save_articles(articles)
    print(f"Saved {len(articles)} articles to {path}")
    for article in articles:
        print(f"- [{article.source}] {article.title[:80]}... (relevance: {article.relevance_score:.2f})")


if __name__ == "__main__":
    main()
