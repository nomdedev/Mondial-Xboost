# ML Model Gates — Mondial-Xboost

Antes de aceptar un modelo nuevo, ejecutar `python scripts/run_backtest_gate.py`.

## Gates

- [ ] **Train/test temporal**: no usar `train_test_split` estratificado por índice.
- [ ] **Backtest walk-forward**: al menos 3 Mundiales (2014, 2018, 2022).
- [ ] **Métricas mínimas**:
  - log-loss < 1.05
  - Brier < 0.22
  - top-1 accuracy > 45%
  - ROI > -5%
- [ ] **Calibración**: ECE < 0.10 preferido.
- [ ] **Overfit gap**: train/test accuracy gap < 5%.
- [ ] **Feature importance estable**: top-5 features no varían drásticamente entre folds.
- [ ] **Modelo versionado**: manifest `data/models/model_manifest.json` (deseado).
- [ ] **FEATURE_COLS canónico**: usa `predictors.feature_engineering.FEATURE_COLS`.

## Veredicto

- **PASS**: cumple todas las métricas mínimas y no empeora baseline.
- **WARNING**: incumple métrica secundaria (ej. ROI) pero calibración es buena.
- **BLOCK**: empeora log-loss/accuracy vs baseline o falla gate requerido.
