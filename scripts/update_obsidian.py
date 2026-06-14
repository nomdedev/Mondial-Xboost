import json
import os
from datetime import datetime
from pathlib import Path

# Check for Obsidian vault
obsidian_paths = [
    Path(os.path.expanduser('~')) / 'Obsidian' / 'Mondial-xBoost',
    Path(os.path.expanduser('~')) / 'Documents' / 'Obsidian' / 'Mondial-xBoost',
    Path(os.path.expanduser('~')) / 'Obsidian Vault' / 'Mondial-xBoost',
    Path('D:/martin/Obsidian/Mondial-xBoost'),
]

vault_path = None
for p in obsidian_paths:
    if p.exists():
        vault_path = p
        break

if vault_path:
    print(f'Found Obsidian vault: {vault_path}')

    # Create daily note
    today = datetime.now().strftime('%Y-%m-%d')
    daily_dir = vault_path / 'Daily'
    daily_dir.mkdir(exist_ok=True)

    # Load predictions
    with open('data/processed/latest_predictions.json') as f:
        data = json.load(f)
    preds = data['predictions']

    # Create markdown content
    md_content = f"""# Oloráculo xBoost - {today}

## Pipeline Execution Summary
- **Date**: {datetime.now().isoformat()}
- **Status**: COMPLETED (with warnings)
- **Predictions Generated**: {len(preds)}

## Pipeline Steps
1. News Scraper - 2 articles scraped
2. ETL - football-data.co.uk failed (encoding issue on WC.csv), using cached data
3. Feature Engineering - 15,866 training rows, 47 future fixtures
4. XGBoost Predictions - Models trained and saved
5. LLM Analysis - Skipped (OPENROUTER_API_KEY not set)
6. Export - Predictions exported to JSON

## Top Predictions (Next 30 Days)

| Match | Date | Prediction | Probabilities |
|-------|------|------------|---------------|
"""

    for p in preds[:15]:
        md_content += f"| {p['home_team']} vs {p['away_team']} | {p.get('date', 'N/A')} | **{p['top_pick']}** | H:{p['prob_home_win']:.2f} D:{p['prob_draw']:.2f} A:{p['prob_away_win']:.2f} |\n"

    home_count = sum(1 for p in preds if p['top_pick'] == 'Home')
    draw_count = sum(1 for p in preds if p['top_pick'] == 'Draw')
    away_count = sum(1 for p in preds if p['top_pick'] == 'Away')

    md_content += f"""
## Distribution
- Home wins: {home_count}
- Draws: {draw_count}
- Away wins: {away_count}

## Notes
- ETL step encountered encoding error with World Cup data from football-data.co.uk
- LLM analysis skipped due to missing API key
- All predictions based on blended XGBoost + cold-start model

## Data Sources
- Historical matches: 42,315
- Training features: 15,866 rows
- Future fixtures: 47 matches
"""

    daily_file = daily_dir / f'{today}.md'
    with open(daily_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f'Created daily note: {daily_file}')
else:
    print('Obsidian vault not found')
    print('Searched paths:')
    for p in obsidian_paths:
        print(f'  {p} - exists: {p.exists()}')
