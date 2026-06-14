# Contributing to Mondial-Xboost

## Pipeline canónico

El único motor ML activo es **XGBoost**. No agregues nuevos algoritmos sin evidencia de mejora en backtest.

```
MondialXboost.Web/Data/historical_results.csv
        ↓
predictors/feature_engineering.py  →  FEATURE_COLS
        ↓
predictors/xgboost_engine.py       →  XGBoostFootballPredictor
        ↓
predictors/api.py                  →  FastAPI bridge
        ↓
MondialXboost.Web/Services/XGBoostBridgeService.cs
```

## CLI local

El proyecto incluye un wrapper `mondial` (y `mondial.cmd` en Windows) que detecta el venv y ejecuta todas las tareas del pipeline:

```bash
./mondial instalar                       # instala dependencias y el paquete
./mondial entrenar                       # entrena el modelo canónico
./mondial entrenar --elo-decay 4 --elo-recent 8   # Elo con decay temporal
./mondial entrenar-gpu                   # entrena usando GPU (XGBOOST_DEVICE=cuda)
./mondial entrenar-cold-start            # entrena modelo de cold-start
./mondial predecir --home Brazil --away Morocco
./mondial predecir --home Brazil --away Morocco --blend
./mondial test                           # pytest tests/
./mondial lint                           # ruff check
./mondial gates                          # verify_gates
./mondial backtest                       # backtest de World Cup
./mondial bridge                         # smoke test del bridge C# <-> Python
./mondial elo                            # compara Elo contra World Football Elo
./mondial auditar                        # audita leakage temporal
./mondial loop --trials 50               # tuning con Optuna
./mondial data-council                   # revisión del data council
./mondial dashboard                      # dashboard de entrenamiento
./mondial servidor                       # levanta el bridge FastAPI
./mondial health                         # consulta /health del servidor
./mondial manifest                       # muestra model_manifest.json
./mondial limpiar                        # borra caché y artefactos de test
./mondial info                           # información del entorno
```

En Windows con cmd/PowerShell:

```cmd
mondial.cmd instalar
mondial.cmd entrenar
mondial.cmd entrenar-gpu
mondial.cmd predecir --home Brazil --away Morocco
```

Si instalás el paquete con `pip install -e .`, también queda disponible el comando `mondial` global.

## Antes de commitear

```bash
# Lint + tests + gates
./mondial lint
./mondial test
./mondial gates
```

## Reglas de feature engineering

- Ninguna feature puede conocer el resultado del partido a predecir.
- Rolling stats deben usar `shift(1)`.
- Elo se actualiza secuencialmente en orden cronológico.
- Home advantage (+100 Elo) solo aplica si `neutral=False`.

## Modelos

- Los artefactos `.pkl` nunca se commitean (ver `.gitignore`).
- Al entrenar se genera `data/models/model_manifest.json`.
- Si modificas `FEATURE_COLS`, actualiza:
  - `README.md`
  - `docs/vault/03-Architecture/experimental-components.md`
  - tests relacionados

## Tests

- No sobreescribas el modelo canónico `xgboost_football` desde tests. Usa nombres de test o `tmp_path`.
- Los tests de integración del bridge usan `XGBOOST_MODEL_NAME=test_api_model`.

## .NET

Los gates .NET se saltan si no hay SDK disponible. Valida localmente con:

```bash
dotnet build
dotnet test
dotnet format --verify-no-changes
```
