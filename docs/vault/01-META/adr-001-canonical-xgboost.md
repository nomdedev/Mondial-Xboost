# ADR-001: Motor y API XGBoost Canónicos

## Estado
Aceptado — 2026-06-13

## Contexto
El proyecto tenía dos implementaciones paralelas de XGBoost:
- `predictors/xgboost_engine.py` + `predictors/api.py` (usado por backtest y el bridge C#).
- `predictors/xgboost_predictor.py` + `api/main.py` (API alternativa con CORS `*`, modelos en `models/`).

Esta duplicación causaba:
- Diferentes listas de features (`FEATURE_COLS` duplicado en ~6 archivos).
- Diferentes formatos de serialización (joblib vs JSON).
- Diferentes contratos C#↔Python.
- Imposibilidad de aplicar gates objetivos.

## Decisión
1. **Motor canónico**: `predictors/xgboost_engine.py` (`XGBoostFootballPredictor`).
2. **API canónica**: `predictors/api.py`.
3. **Algoritmo único**: XGBoost es el único algoritmo de ML activo. LightGBM, CatBoost, RandomForest y GradientBoosting de sklearn fueron evaluados y descartados.
4. **Directorio de modelos**: `data/models/`.
5. **Features canónicas**: `predictors.feature_engineering.FEATURE_COLS` (21 columnas).
6. **Motores/APIs legacy**: movidos a `scripts/archive/`; no usables en producción.
7. **Loop de ingeniería**: `scripts/loop_engineering.py` realiza hyperparameter tuning exclusivo de XGBoost con persistencia en caliente.

## Evidencia de descarte de otros algoritmos
Resultados de 98 trials en Google Colab (temporal split, ~50k partidos):

| Modelo | Val Acc | Test Acc | Overfit Gap | Observación |
|--------|---------|----------|-------------|-------------|
| XGBoost | 56.0% | 52.1% | 0.45% | Rápido, buena generalización. |
| LightGBM | 56.59% | 52.43% | -0.2% | Similar a XGBoost. |
| CatBoost | 56.72% | 53.09% | -3.41% | Mejor test, pero más lento. |
| RandomForest | 52.26% | 50.31% | 18.3% | Overfitting severo. |
| GradientBoosting | — | — | — | No terminó; ~800s/trial. |

**Por qué XGBoost:**
- Competitivo en precisión (52.1% test).
- Mucho más rápido que CatBoost y GradientBoosting.
- Menor riesgo de overfitting que RandomForest.
- Mejor integración con el ecosistema Python/.NET del proyecto.
- No requiere dependencias adicionales (LightGBM/CatBoost).

## Consecuencias

## Consecuencias

### Positivas
- Una sola fuente de verdad para features y modelo.
- Gates automáticos reproducibles.
- Backtest y producción usan el mismo código.
- Contrato C#↔Python documentado y testeado.

### Negativas
- Cambios en `FEATURE_COLS` requieren re-entrenar el modelo.
- `api/main.py` ya no está disponible; cualquier consumidor directo debe migrar.

## Compliance
- `scripts/verify_gates.py` verifica que no existan imports de `xgboost_predictor` en código productivo.
- `bridge-integrator` agent es el dueño de mantener el contrato actualizado.
