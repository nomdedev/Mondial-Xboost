# Componentes experimentales / no integrados

Este documento lista código que existe en el repositorio pero **no forma parte del pipeline canónico de entrenamiento y predicción**. Se mantienen para desarrollo futuro o como referencia.

## Pipeline canónico activo

```
historical_results.csv
        ↓
predictors/feature_engineering.py  →  FEATURE_COLS (22 columnas)
        ↓
predictors/xgboost_engine.py       →  XGBoostFootballPredictor
        ↓
predictors/api.py                  →  FastAPI bridge
        ↓
Oloraculo.Web/Services/XGBoostBridgeService.cs
```

## Componentes aún no integrados

### 1. `predictors/player_weights.py`
- **Qué hace:** asigna pesos a atributos de jugadores (lesión, forma, minutos, etc.).
- **Estado:** `DEFAULT_WEIGHTS` suma 1.0 y pasa validación del Data Council.
- **Por qué no está integrado:** aún no hay pipeline de datos de jugadores conectado a `feature_engineering.py`.
- **Próximo paso:** definir fuente de datos de jugadores y agregar features derivadas a `FEATURE_COLS`.

### 2. `predictors/llm_analysis_service.py`
- **Qué hace:** llama a OpenRouter (GPT-4o-mini / Claude 3.5 Sonnet) para análisis táctico y ajuste de probabilidades.
- **Estado:** implementado pero no consumido por `PredictionService` ni por el pipeline ML.
- **Por qué no está integrado:** depende de API key externa y no hay métricas de calibración validadas.
- **Próximo paso:** evaluar con un dataset de partidos si el ajuste mejora log-loss/Brier.

### 3. `predictors/etl.py`
- **Qué hace:** ETL alternativo que descarga datos de football-data.co.uk, Wikipedia y API-Football.
- **Estado:** funcional pero no usado por el entrenamiento canónico.
- **Por qué no está integrado:** el dataset canónico actual proviene de `Oloraculo.Web/Data/historical_results.csv`.
- **Próximo paso:** unificar ETLs para que generen el mismo schema y alimenten `historical_results.csv`.

### 4. `scrapers/`
- **Qué hace:** scrapers de noticias, odds y datos de jugadores.
- **Estado:** parcialmente implementado.
- **Por qué no está integrado:** no hay consumidor estable en el pipeline.
- **Próximo paso:** integrar al ETL o como feature opcional.

## Criterio para promover un componente a canónico

1. Tener fuente de datos estable y reproducible.
2. Demostrar mejora estadísticamente significativa en backtest (`run-backtest-gate.py`).
3. Pasar audit de `data-leakage-auditor` y `ml-model-gatekeeper`.
4. Actualizar `FEATURE_COLS`, documentación y tests.
