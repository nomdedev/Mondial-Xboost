# Exp-01: Elo y H2H globales

## Hipótesis

Usar todo el historial disponible (1872-2026) para computar Elo y H2H debería mejorar las predicciones de equipos con pocos partidos recientes, porque sus ratings no se reinician en 1500 al inicio del training window.

## Cambios realizados

- `predictors/feature_engineering.py`:
  - `build_features()` ahora acepta `min_date=None`. Recibe el histórico completo para computar features look-back y filtra las filas de salida al final.
  - `build_training_dataset()` pasa el histórico completo a `build_features()` y delega el corte temporal a `min_date`.
- `backtest/world_cup_backtest.py`:
  - El backtest ahora también usa el historial completo para computar Elo/H2H y limita las filas de entrenamiento con `min_date`.

## Métricas

### Split temporal honesto (< 2024 train, >= 2024 test)

| Modelo | Accuracy | Log loss | Distribución predicciones |
|--------|----------|----------|---------------------------|
| Baseline honesto (Elo reiniciado en 2010) | 46.09% | 1.060 | [0, 0, 2584] — siempre away |
| Exp-01 (Elo/H2H global) | 59.19% | 0.906 | [829, 12, 1749] |
| Modelo canónico entrenado con 2010-2026 (leakage) | 61.93% | 0.814 | — |

**Interpretación**: Exp-01 mejora **+13.1 puntos de accuracy** sobre el baseline honesto. El baseline honesto con Elo reiniciado en 2010 colapsa a predecir siempre away, lo que indica que el modelo no generaliza sin información histórica previa. El modelo canónico parece mejor porque fue entrenado con datos que incluyen el período de test (leakage temporal); no es comparable directamente.

### World Cup backtest

```
WC 2014: log_loss=0.9539 brier=0.1891 acc=59.38% roi=23.86%
WC 2018: log_loss=0.9781 brier=0.1946 acc=53.12% roi=6.52%
WC 2022: log_loss=1.0725 brier=0.2080 acc=50.00% roi=-1.16%
Average: log_loss=1.0015 brier=0.1972 acc=54.17% roi=9.74%
Verdict: PASS
```

## Conclusiones

- El cambio es **prometedor** y justifica seguir por esta línea.
- El problema principal no es solo "predecir todos los partidos" (la cobertura ya existe), sino **predecir bien cuando falta historial reciente**.
- El modelo canónico actual tiene leakage temporal porque se entrena con todo 2010-2026. La validación honesta muestra que el rendimiento real es más modesto.

## Próximos pasos recomendados

1. **Exp-02: Elo con decay temporal** — dar menos peso a partidos antiguos en el rating Elo (K o factor de decaimiento por antigüedad).
2. **Exp-03: Modelo de cold-start** — entrenar un submodelo para partidos donde al menos un equipo tiene <10 partidos previos.
3. **Exp-04: Features de torneo** — agregar importancia del partido, descanso, distancia.

## Decisión

**Adoptar parcialmente**: integrar el uso de historial completo para Elo/H2H en el pipeline, pero mantener el modelo canónico `xgboost_football` como referencia hasta que un experimento supere claramente las métricas honestas.
