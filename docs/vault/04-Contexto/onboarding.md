# Onboarding

## Qué es este proyecto

Mondial-Xboost predice resultados de partidos de fútbol para el Mundial 2026. El pipeline canónico usa datos históricos, feature engineering anti-leakage y XGBoost.

## Cómo empezar

1. Clonar repo.
2. Crear venv: `python -m venv venv && source venv/Scripts/activate` (Windows) o `venv/bin/activate` (Unix).
3. Instalar deps: `pip install -r requirements.txt -r requirements-dev.txt`.
4. Entrenar modelo: `python scripts/train.py`.
5. Correr gates: `python scripts/run_tests.py && python scripts/verify_gates.py`.

## Arquitectura en una línea

- `MondialXboost.Web/` → app .NET 9 Blazor Server.
- `predictors/` → Python: feature engineering, XGBoost, FastAPI bridge.
- `tests/` → pytest.
- `scripts/` → utilidades de entrenamiento, gates, backtest.
- `docs/vault/` → documentación extendida.

## Convenciones

- Python 3.11+ con type hints.
- `ruff` para lint/format.
- No data leakage en features.
- No commitear `.pkl`; sí commitear `model_manifest.json`.

## Quién hace qué

- `data-engineer` → curación de `historical_results.csv`.
- `ml-engineer` → feature engineering y modelo.
- `backend-engineer` → bridge C# ↔ Python.
- `qa-agent` → tests, backtests, gates.
