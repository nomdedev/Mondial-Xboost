# Arquitectura del Sistema

## Capas
1. **Presentación**: Blazor Server + dashboard opcional
2. **Orquestación**: Agentes + pipeline 7 fases
3. **Predicción**: Motor canónico XGBoost (outcome 1X2 + regresión de goles). Los predictores clásicos de .NET (Elo, Poisson) siguen existiendo pero el pipeline Python usa XGBoost.
4. **Feature Engineering**: Equipo, jugador, H2H, contexto
5. **Datos**: CSV canónico (`historical_results.csv`), SQLite (app .NET), APIs externas opcionales.

## Bridges
- C# `XGBoostBridgeService` → HTTP → Python FastAPI `/predict`
- Python `predictors/api.py` expone `/health`, `/train`, `/predict`

## Data flow diario
1. Validar/actualizar CSV canónico (`historical_results.csv`).
2. Feature engineering con anti-leakage.
3. Entrenar o cargar modelo XGBoost (`scripts/train.py` o `/train`).
4. Predecir fixtures vía FastAPI `/predict`.
5. Simulación Monte Carlo del torneo (app .NET).
6. Exportar README/JSON.
