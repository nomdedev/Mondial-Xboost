"""
Compare internal Elo ratings against World Football Elo Ratings (eloratings.net).

Outputs:
    backtest/results/elo_comparison.json

Usage:
    python scripts/compare_elo_worldfootball.py
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from predictors.feature_engineering import compute_elo_ratings, load_historical_results

# Allow printing unicode team names on Windows terminals
sys.stdout.reconfigure(encoding="utf-8")

# Optional dependency: used only if available for name normalization  # noqa: E402
try:
    import country_converter as coco

    _cc = coco.CountryConverter()
except Exception:  # pragma: no cover
    _cc = None

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "worldfootball_elo_rendered.json"
REPORT_PATH = PROJECT_ROOT / "backtest" / "results" / "elo_comparison.json"
CACHE_MAX_AGE_HOURS = 24

# Map common football name variants to a form country_converter understands.
NAME_ALIASES = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Czech Republic": "Czechia",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Côte d'Ivoire",
    "Bosnia/Herzeg": "Bosnia and Herzegovina",
    "Equat Guinea": "Equatorial Guinea",
    "C African Rep": "Central African Republic",
    "St Vincent/Gren": "Saint Vincent and the Grenadines",
    "Papua N Guinea": "Papua New Guinea",
    "Dominican Rep": "Dominican Republic",
    "Trinidad/Tobago": "Trinidad and Tobago",
    "Turkey": "Türkiye",
}


@contextmanager
def _silence():
    """Silence stdout/stderr (country_converter is noisy)."""
    null = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(null)


def _normalize_name(name: str) -> str | None:
    """Return a canonical country name or None if not recognized."""
    name = str(name).strip()
    name = NAME_ALIASES.get(name, name)
    if _cc is not None:
        with _silence():
            norm = _cc.convert(name, to="name_short", not_found=None)
        if norm:
            return str(norm)
    return None


def _cache_is_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    mtime = datetime.fromtimestamp(CACHE_PATH.stat().st_mtime, tz=UTC)
    age_hours = (datetime.now(UTC) - mtime).total_seconds() / 3600
    return age_hours < CACHE_MAX_AGE_HOURS


def fetch_worldfootball_elo() -> list[dict[str, int | str]]:
    """Scrape the rendered ratings table from eloratings.net."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _cache_is_fresh():
        print(f"Using cached Elo data: {CACHE_PATH}")
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    print("Fetching World Football Elo Ratings from eloratings.net (Selenium)...")
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError as exc:
        raise RuntimeError(
            "Selenium/webdriver-manager required. Install: pip install selenium webdriver-manager"
        ) from exc

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    try:
        driver.get("https://eloratings.net/")
        # Wait for the SlickGrid to populate via JS
        driver.implicitly_wait(10)
        rows = driver.find_elements(By.CSS_SELECTOR, ".slick-row")
        data: list[dict[str, int | str]] = []
        for row in rows:
            cells = row.find_elements(By.CSS_SELECTOR, ".slick-cell")
            text = [c.text for c in cells]
            if len(text) >= 4:
                try:
                    data.append(
                        {
                            "rank": int(text[0]),
                            "name": text[1],
                            "rating": int(text[2]),
                        }
                    )
                except ValueError:
                    continue
    finally:
        driver.quit()

    if not data:
        raise RuntimeError("No ratings extracted from eloratings.net")

    CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Cached {len(data)} ratings to {CACHE_PATH}")
    return data


def compute_internal_elo() -> pd.DataFrame:
    """Compute the latest internal Elo rating per team, including match counts."""
    df = load_historical_results()
    df = compute_elo_ratings(df)

    home = df[["date", "home_team", "home_elo_before", "home_elo_provisional"]].rename(
        columns={
            "home_team": "team",
            "home_elo_before": "elo",
            "home_elo_provisional": "provisional",
        }
    )
    away = df[["date", "away_team", "away_elo_before", "away_elo_provisional"]].rename(
        columns={
            "away_team": "team",
            "away_elo_before": "elo",
            "away_elo_provisional": "provisional",
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["matches"] = long.groupby("team").cumcount() + 1
    long = long.sort_values(["team", "date"])
    latest = long.drop_duplicates(subset=["team"], keep="last")
    return latest[["team", "elo", "provisional", "matches"]].copy()


def build_comparison(internal: pd.DataFrame, external: list[dict]) -> dict:
    """Join internal and external ratings on normalized team names."""
    external_norm: dict[str, int] = {}
    external_unmatched: list[str] = []
    for item in external:
        norm = _normalize_name(item["name"])  # type: ignore[arg-type]
        if norm:
            external_norm[norm] = item["rating"]  # type: ignore[index]
        else:
            external_unmatched.append(item["name"])  # type: ignore[arg-type]

    # Resolve collisions: multiple internal names may normalize to the same country
    # (e.g. "Spain" and "Central Spain"). Keep the one with the most matches.
    internal["normalized_name"] = internal["team"].apply(_normalize_name)
    matched = internal.dropna(subset=["normalized_name"]).copy()
    matched = matched[matched["normalized_name"].isin(external_norm)]
    matched = matched.sort_values("matches", ascending=False)
    matched = matched.drop_duplicates(subset=["normalized_name"], keep="first")

    rows = []
    internal_unmatched: list[str] = []
    for _, r in internal.iterrows():
        norm = r["normalized_name"]
        if norm and norm in external_norm and r["team"] in set(matched["team"]):
            diff = float(r["elo"] - external_norm[norm])
            rows.append(
                {
                    "team": r["team"],
                    "normalized_name": norm,
                    "internal_elo": round(float(r["elo"]), 2),
                    "external_elo": external_norm[norm],
                    "difference": round(diff, 2),
                    "absolute_difference": round(abs(diff), 2),
                    "provisional": bool(r["provisional"]),
                }
            )
        else:
            internal_unmatched.append(r["team"])

    comparison_df = pd.DataFrame(rows)

    if comparison_df.empty:
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "No teams could be matched between internal and external Elo sources.",
        }

    def _stats(df: pd.DataFrame) -> dict:
        subset = df["absolute_difference"]
        signed = df["difference"]
        return {
            "count": int(len(df)),
            "mean_absolute_difference": round(float(subset.mean()), 2),
            "median_absolute_difference": round(float(subset.median()), 2),
            "max_absolute_difference": round(float(subset.max()), 2),
            "mean_difference": round(float(signed.mean()), 2),
        }

    all_stats = _stats(comparison_df)
    stable_stats = _stats(comparison_df[~comparison_df["provisional"]])

    # Stats for top 100 external-rated teams (most relevant for predictions)
    top100_df = comparison_df.nlargest(100, "external_elo")
    top100_stats = _stats(top100_df)

    # Correlation for stable teams
    stable = comparison_df[~comparison_df["provisional"]]
    correlation = (
        float(stable["internal_elo"].corr(stable["external_elo"]))
        if len(stable) > 2
        else None
    )

    # Verdict: use top 100 most relevant teams; flag overall stable separately
    threshold = 100.0
    if top100_stats["mean_absolute_difference"] <= threshold:
        verdict = "PASS"
    elif top100_stats["mean_absolute_difference"] <= threshold * 1.5:
        verdict = "WARNING"
    else:
        verdict = "BLOCK"

    # Largest diffs for stable teams
    top_diffs = (
        stable.nlargest(10, "absolute_difference")[
            ["team", "internal_elo", "external_elo", "absolute_difference"]
        ]
        .to_dict(orient="records")
    )

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "internal_methodology": {
            "initial_elo": 1500.0,
            "k_range": "20-60 by tournament importance",
            "home_advantage_points": 100,
            "goal_adjusted_k": False,
            "source_file": "Oloraculo.Web/Data/historical_results.csv",
        },
        "external_source": "https://eloratings.net/ (World Football Elo Ratings)",
        "total_internal_teams": int(len(internal)),
        "total_external_teams": int(len(external)),
        "matched_teams": int(all_stats["count"]),
        "matched_stable_teams": int(stable_stats["count"]),
        "all_teams_stats": all_stats,
        "stable_teams_stats": stable_stats,
        "top100_teams_stats": top100_stats,
        "correlation": correlation,
        "verdict": verdict,
        "top_differences": top_diffs,
        "unmatched_external_teams_count": int(len(external_unmatched)),
        "unmatched_internal_teams_count": int(len(internal_unmatched)),
        "per_team": comparison_df.to_dict(orient="records"),
    }
    return report


def main() -> None:
    external = fetch_worldfootball_elo()
    internal = compute_internal_elo()
    report = build_comparison(internal, external)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nReport saved to {REPORT_PATH}")
    print(f"Verdict: {report.get('verdict', 'N/A')}")
    stable = report.get("stable_teams_stats", {})
    top100 = report.get("top100_teams_stats", {})
    print(
        f"Stable teams compared: {stable.get('count')} | "
        f"mean abs diff: {stable.get('mean_absolute_difference')} pts | "
        f"max abs diff: {stable.get('max_absolute_difference')} pts | "
        f"correlation: {report.get('correlation', 0):.3f}"
    )
    print(
        f"Top-100 teams compared: {top100.get('count')} | "
        f"mean abs diff: {top100.get('mean_absolute_difference')} pts | "
        f"max abs diff: {top100.get('max_absolute_difference')} pts"
    )


if __name__ == "__main__":
    main()
