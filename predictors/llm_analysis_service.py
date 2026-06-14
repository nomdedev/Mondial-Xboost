"""
LLM Analysis Service — OpenRouter Integration
===============================================
Uses LLM (GPT-4o / Claude) for:
- Narrative analysis of matches (tactics, momentum, form)
- Calibration of probabilities based on qualitative factors
- Player availability classification
- News sentiment analysis

Usage:
    from llm_analysis_service import LLMAnalysisService
    service = LLMAnalysisService(api_key="YOUR_KEY")
    analysis = service.analyze_match(fixture_data)
"""
import json
import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMMatchAnalysis:
    fixture_id: str
    tactical_analysis: str
    momentum_assessment: str
    key_factors: list[str]
    risk_factors: list[str]
    probability_adjustment: float  # -0.1 to +0.1 adjustment to home win prob
    confidence: float  # 0-1
    model_used: str


class LLMAnalysisService:
    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    ANALYSIS_MODEL = "anthropic/claude-3.5-sonnet"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or self.DEFAULT_MODEL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mondial-xboost.app",
            "X-Title": "MondialXboost xBoost"
        })

    def _call_llm(self, messages: list[dict[str, str]],
                  model: str | None = None,
                  temperature: float = 0.3,
                  max_tokens: int = 2000) -> str:
        """Call OpenRouter API."""
        model = model or self.model

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = self.session.post(
            f"{self.BASE_URL}/chat/completions",
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def analyze_match(self, fixture_data: dict[str, Any]) -> LLMMatchAnalysis:
        """Generate tactical analysis and probability calibration for a match."""

        home_team = fixture_data.get("home_team", {})
        away_team = fixture_data.get("away_team", {})
        h2h = fixture_data.get("head_to_head", [])
        recent_form = fixture_data.get("recent_form", {})
        news = fixture_data.get("news", [])

        prompt = f"""You are a football analyst specializing in World Cup predictions.
Analyze the following match and provide tactical insights and probability adjustments.

MATCH: {home_team.get('name', 'Home')} vs {away_team.get('name', 'Away')}

HOME TEAM DATA:
- Squad: {home_team.get('squad_size', 'N/A')} players
- Avg Age: {home_team.get('avg_age', 'N/A')}
- Injured: {home_team.get('injured_count', 0)} players
- Top Scorers: {json.dumps(home_team.get('top_scorers', [])[:3])}
- Form (last 5): {recent_form.get('home', 'N/A')}

AWAY TEAM DATA:
- Squad: {away_team.get('squad_size', 'N/A')} players
- Avg Age: {away_team.get('avg_age', 'N/A')}
- Injured: {away_team.get('injured_count', 0)} players
- Top Scorers: {json.dumps(away_team.get('top_scorers', [])[:3])}
- Form (last 5): {recent_form.get('away', 'N/A')}

HEAD-TO-HEAD (last {len(h2h)} matches):
{json.dumps(h2h[:5], indent=2)}

RECENT NEWS:
{chr(10).join(f"- {n.get('title', '')}" for n in news[:5])}

Please provide your analysis in this exact JSON format:
{{
    "tactical_analysis": "Detailed tactical breakdown...",
    "momentum_assessment": "Which team has better momentum...",
    "key_factors": ["factor 1", "factor 2", "factor 3"],
    "risk_factors": ["risk 1", "risk 2"],
    "probability_adjustment": 0.0,  // -0.1 to +0.1 adjustment to home win probability
    "confidence": 0.8  // 0-1, how confident you are in this assessment
}}

Be objective and data-driven. Consider injuries, form, tactical matchups, and motivation."""

        messages = [
            {"role": "system", "content": "You are an expert football analyst. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._call_llm(messages, model=self.ANALYSIS_MODEL, temperature=0.3)

            # Extract JSON from response
            json_str = self._extract_json(response)
            analysis = json.loads(json_str)

            return LLMMatchAnalysis(
                fixture_id=fixture_data.get("fixture_id", ""),
                tactical_analysis=analysis.get("tactical_analysis", ""),
                momentum_assessment=analysis.get("momentum_assessment", ""),
                key_factors=analysis.get("key_factors", []),
                risk_factors=analysis.get("risk_factors", []),
                probability_adjustment=analysis.get("probability_adjustment", 0.0),
                confidence=analysis.get("confidence", 0.5),
                model_used=self.ANALYSIS_MODEL
            )

        except Exception as e:
            print(f"LLM analysis error: {e}")
            return LLMMatchAnalysis(
                fixture_id=fixture_data.get("fixture_id", ""),
                tactical_analysis="Error generating analysis",
                momentum_assessment="",
                key_factors=[],
                risk_factors=[],
                probability_adjustment=0.0,
                confidence=0.0,
                model_used="error"
            )

    def classify_availability(self, news_text: str) -> dict[str, Any]:
        """Classify player availability from news text."""

        prompt = f"""Analyze the following text and extract player availability information.

TEXT:
{news_text[:3000]}

For each player mentioned, determine:
1. Player name
2. Team
3. Status: Available / Doubtful / Injured / Ruled Out / Unknown
4. Injury type (if applicable)
5. Expected return date (if mentioned)

Respond in JSON format:
{{
    "claims": [
        {{
            "player": "Player Name",
            "team": "Team Name",
            "status": "Available|Doubtful|Injured|Ruled Out|Unknown",
            "injury_type": "muscle strain|ankle|knee|etc",
            "expected_return": "2026-06-20 or null",
            "confidence": 0.9
        }}
    ]
}}"""

        messages = [
            {"role": "system", "content": "You are a medical/football analyst. Extract availability information. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._call_llm(messages, model=self.DEFAULT_MODEL, temperature=0.2, max_tokens=1500)
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except Exception as e:
            return {"claims": [], "error": str(e)}

    def analyze_sentiment(self, news_articles: list[dict[str, str]]) -> dict[str, float]:
        """Analyze sentiment of news articles per team."""

        articles_text = "\n\n".join([
            f"Title: {a.get('title', '')}\nContent: {a.get('content', '')[:500]}"
            for a in news_articles[:10]
        ])

        prompt = f"""Analyze the sentiment of the following news articles regarding football teams.

ARTICLES:
{articles_text}

For each team mentioned, provide a sentiment score from -1 (very negative) to +1 (very positive).
Consider: injuries (negative), good form (positive), tactical praise (positive), internal conflicts (negative).

Respond in JSON:
{{
    "team_sentiments": {{
        "Team Name": 0.5,
        "Another Team": -0.3
    }},
    "overall_market_sentiment": 0.1
}}"""

        messages = [
            {"role": "system", "content": "You are a sentiment analysis expert. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._call_llm(messages, model=self.DEFAULT_MODEL, temperature=0.2, max_tokens=1000)
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except Exception as e:
            return {"team_sentiments": {}, "error": str(e)}

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        # Try to find JSON block
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            return text[start:end+1]

        return text


# ===== CLI =====

if __name__ == "__main__":
    import sys

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    service = LLMAnalysisService(api_key=api_key)

    # Test with sample data
    test_fixture = {
        "fixture_id": "test-123",
        "home_team": {
            "name": "Argentina",
            "squad_size": 26,
            "avg_age": 28.5,
            "injured_count": 1,
            "top_scorers": [{"name": "Messi", "goals": 5}]
        },
        "away_team": {
            "name": "Brazil",
            "squad_size": 26,
            "avg_age": 27.2,
            "injured_count": 0,
            "top_scorers": [{"name": "Vinicius Jr", "goals": 4}]
        },
        "head_to_head": [
            {"date": "2024-11-14", "home": "Argentina", "away": "Brazil", "score": "1-0"}
        ],
        "recent_form": {
            "home": "WWWDW",
            "away": "WWLWW"
        },
        "news": [
            {"title": "Messi training well ahead of Brazil clash"},
            {"title": "Brazil squad fully fit for World Cup qualifier"}
        ]
    }

    print("=== Testing LLM Match Analysis ===")
    analysis = service.analyze_match(test_fixture)

    print(f"\nTactical Analysis: {analysis.tactical_analysis[:200]}...")
    print(f"Momentum: {analysis.momentum_assessment[:200]}...")
    print(f"Key Factors: {analysis.key_factors}")
    print(f"Risk Factors: {analysis.risk_factors}")
    print(f"Probability Adjustment: {analysis.probability_adjustment}")
    print(f"Confidence: {analysis.confidence}")
    print(f"Model Used: {analysis.model_used}")

    print("\nLLMAnalysisService OK")
