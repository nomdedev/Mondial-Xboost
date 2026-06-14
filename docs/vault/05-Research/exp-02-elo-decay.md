# Exp-02: Elo con decay temporal + Elo reciente

## Hipótesis

El problema de Exp-01 (Elo/H2H global) es que partidos muy antiguos contaminan el rating actual. Un Elo con **decaimiento temporal** debería:

1. Penalizar la inactividad (los equipos inactivos regresan a 1500).
2. Capturar la forma reciente a través de un **Elo reciente** (ventana de 8 años).

Esto debería mejorar especialmente la predicción de equipos con historial irregular o que cambiaron de nivel con el tiempo.

## Cambios realizados

- `predictors/feature_engineering.py`:
  - `compute_elo_ratings()` ahora acepta `decay_halflife_years` y `recent_years`.
  - Si un equipo no juega, su rating regresa exponencialmente hacia `DEFAULT_ELO` (1500) con el half-life configurado.
  - Se computa un segundo rating `elo_diff_recent` usando solo partidos dentro de la ventana configurada.
  - `FEATURE_COLS` pasa a 23 columnas incluyendo `elo_diff_recent`.
- `scripts/predict.py`: `build_features_for_prediction()` ahora también produce `elo_diff_recent`.
- `scripts/train.py` y `scripts/mondial_cli.py`: flags `--elo-decay` y `--elo-recent`.

## Métricas

### Split temporal honesto (< 2024 train, >= 2024 test)

| Modelo | Accuracy | Log loss | Observación |
|--------|----------|----------|-------------|
| Baseline honesto (Elo reiniciado en 2010) | 46.09% | 1.060 | Predice siempre away |
| Exp-01 (Elo/H2H global sin decay) | 59.19% | 0.906 | Mejora fuerte |
| **Exp-02 (decay=4y, recent=8y)** | **61.97%** | **0.820** | **Mejor resultado honesto** |
| Modelo canónico con leakage (2010-2026) | 61.93% | 0.814 | No comparable |

**Mejora sobre baseline honesto**: +15.9 puntos de accuracy.
**Mejora sobre Exp-01**: +2.8 puntos de accuracy.

### World Cup backtest

```bash
mondial backtest
```

Ejecutado con el pipeline actualizado (histórico completo para features look-back):

```
WC 2014: log_loss=0.9478 brier=0.1874 acc=59.38% roi=23.75%
WC 2018: log_loss=0.9640 brier=0.1916 acc=56.25% roi=7.85%
WC 2022: log_loss=1.0653 brier=0.2060 acc=50.00% roi=-7.76%
Average: log_loss=0.9924 brier=0.1950 acc=55.21% roi=7.94%
Verdict: PASS
```

### Feature importance

Top feature del modelo Exp-02: **`elo_diff_recent`**.

Esto confirma que el Elo reciente es más informativo que el Elo histórico sin decay.

## Conclusiones

- El decay temporal y el Elo reciente son **efectivos**.
- Exp-02 supera al baseline honesto y alcanza el rendimiento del canónico entrenado con leakage temporal.
- La feature `elo_diff_recent` es la más importante, lo que sugiere que la forma reciente pesa más que el historial lejano.

## Decisión

**Adoptar como nuevo baseline**: los cambios deberían integrarse al modelo canónico `xgboost_football` porque:
- No rompen compatibilidad (solo agregan una feature).
- Mejoran métricas honestas de forma significativa.
- El pipeline de predicción ya los soporta.

## Comandos

```bash
# Reproducir Exp-02
./mondial entrenar --name xgboost_football_exp_02_elo_decay --elo-decay 4 --elo-recent 8

# Entrenar el canónico con la nueva configuración
./mondial entrenar --elo-decay 4 --elo-recent 8
```
