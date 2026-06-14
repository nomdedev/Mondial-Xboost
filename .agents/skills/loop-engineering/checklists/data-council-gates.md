# Data Council Gates — Mondial-Xboost

Ejecutar con `python scripts/run-data-council.py`.

## Gates

- [ ] **elo_calibration**: `python scripts/compare_elo_worldfootball.py`
  - Evidencia: `backtest/results/elo_comparison.json`
  - Criterio: `top100.mean_absolute_difference < 100` pts y `correlation > 0.90`

- [ ] **player_weights_ok**: `python scripts/run-data-council.py`
  - Evidencia: `.agents/logs/data-council-report.json`
  - Criterio: `sum(DEFAULT_WEIGHTS.values()) == 1.0`

- [ ] **no_leakage**: `python scripts/audit_leakage.py`
  - Evidencia: `backtest/results/audit_leakage.json` o stdout
  - Criterio: Sin hallazgos `FAIL`

- [ ] **feature_drift** (manual hasta automatizar):
  - Comparar `data/features/train_historical.parquet` vs baseline.
  - Criterio: Ningún feature con |z-score| > 3 en la media.

- [ ] **schema**: `python -c "from predictors.feature_engineering import FEATURE_COLS; ..."`
  - Criterio: Parquet columns == FEATURE_COLS.

- [ ] **freshness**:
  - `historical_results.csv` no mayor a 7 días.
  - Cache API-Football no mayor a 24 horas.

## Veredicto

- **PASS**: todos los checks PASS/SKIP.
- **WARNING**: algún check WARNING pero no crítico.
- **BLOCK**: algún check FAIL.
