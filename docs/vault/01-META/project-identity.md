# Mondial-Xboost — Identidad del Proyecto

## Propósito
Sistema de predicción de resultados del Mundial 2026 combinando modelos estadísticos, ML (XGBoost) y LLM.

## Stack
- .NET 9, Blazor Server, MudBlazor, EF Core 9, SQLite
- Python 3.11, XGBoost, pandas, scikit-learn, FastAPI
- OpenRouter (LLM), API-Football v3

## Estructura clave
- `MondialXboost.Web/` — App C# y predictores clásicos
- `predictors/` — ML Python + FastAPI bridge
- `scrapers/` — Scrapers de noticias
- `tests/` — pytest
- `.agents/` — Equipo de agentes
- `docs/vault/` — Memoria extendida

## Métricas de éxito
- log-loss < 0.65
- Brier < 0.20
- accuracy > 45%
- coverage > 95%
