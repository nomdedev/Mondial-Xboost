# Contrato del Bridge C# ↔ Python

## Componentes

- **C#**: `MondialXboost.Web.Services.XGBoostBridgeService`
- **Python**: `predictors.api:app` (FastAPI)
- **URL base**: configurable via `Mondial-XboostConfig.XGBoostBridgeUrl` (default `http://127.0.0.1:8000`)

## Endpoints

### `GET /health`

**Response 200:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /train`

**Query params:**
- `min_date` (string, optional): fecha mínima para entrenar, default `"2010-01-01"`.

**Response 200:**
```json
{
  "status": "trained",
  "metrics": { ... },
  "paths": { "outcome": "...", "home_goals": "...", "away_goals": "...", "meta": "..." }
}
```

Además, `train_and_save()` genera/actualiza `data/models/model_manifest.json` con metadata reproducible del modelo.

### `POST /predict`

**Request body:**
```json
{
  "historical_path": null,
  "fixtures": [
    {
      "date": "2026-06-15",
      "home_team": "Argentina",
      "away_team": "Brazil",
      "neutral": true
    }
  ]
}
```

**Response 200:**
```json
{
  "predictions": [
    {
      "home_team": "Argentina",
      "away_team": "Brazil",
      "date": "2026-06-15",
      "prob_away_win": 0.30,
      "prob_draw": 0.25,
      "prob_home_win": 0.45,
      "expected_home_goals": 1.50,
      "expected_away_goals": 1.10,
      "top_pick": "Home"
    }
  ]
}
```

## Reglas

1. Las probabilidades deben sumar 1.0 (tolerancia 0.01).
2. `top_pick` ∈ `{Home, Draw, Away}`.
3. `expected_home_goals` y `expected_away_goals` ≥ 0.
4. El bridge puede devolver 503 si no hay modelo cargado.
5. El cliente C# debe manejar timeouts y degradación graceful.

## Evolución

Cualquier cambio en este contrato requiere actualizar simultáneamente:
- `MondialXboost.Web/Services/XGBoostBridgeService.cs`
- `predictors/api.py`
- `tests/test_api.py`
- `MondialXboost.Web.Tests/Services/XGBoostBridgeServiceTests.cs`
