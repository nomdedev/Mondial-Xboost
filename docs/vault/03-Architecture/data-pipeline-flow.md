# Flujo de Datos de Oloráculo xBoost

## Visión general
El pipeline transforma datos crudos de múltiples fuentes en probabilidades de resultado y distribuciones de marcador para cada partido.

```
Fuentes  →  ETL/Curación  →  Feature Engineering  →  XGBoost  →  UI/Export
```

## 1. Fuentes de datos
| Fuente | Tipo | Uso |
|--------|------|-----|
| `historical_results.csv` | CSV local (~49K) | Entrenamiento y Elo |
| API-Football v3 | API REST | Fixtures, alineaciones, estadísticas |
| football-data.co.uk | CSV descargable | Odds, ligas europeas |
| News scrapers | Web scraping | Lesiones, disponibilidad, rumor |

## 2. ETL y curación
- Normalización de nombres de equipos (`home_team_name` → `home_team`).
- Normalización de goles (`home_goals` → `home_score`).
- Parseo de fechas y campo `neutral`.
- Deduplicación y validación de scores.
- Responsable: `data-engineer` + `data-quality-agent`.

## 3. Feature Engineering
Archivo: `predictors/feature_engineering.py`

### 3.1 Elo ratings
- `compute_elo_ratings()`: ratings previos al partido, `elo_diff`.

### 3.2 Forma reciente
- `_build_team_history()` + `_compute_team_rolling()`: medias móviles de puntos, goles anotados/recibidos, win/draw/loss rate en ventanas 5 y 10.
- `shift(1)` garantiza que no se incluye el partido actual.

### 3.3 Head-to-head
- `_compute_h2h()`: último resultado, goles promedio, diferencia de victorias, años desde el último enfrentamiento.

### 3.4 Jugadores (experimental)
- `player_weights.py` produce `attack_modifier`, `defense_modifier`, `squad_depth_score`.
- No integrado en `FEATURE_COLS` del pipeline canónico.

## 4. Entrenamiento ML
Archivo: `predictors/xgboost_engine.py`

- `XGBClassifier` multiclass para outcome 1X2.
- `CalibratedClassifierCV` con calibración isotónica (mejor resultado de autoresearch).
- Dos `XGBRegressor` para goles local/visitante.
- Persistencia en `data/models/` con `pickle`.
- `data/models/model_manifest.json` guarda metadata reproducible: hash del dataset, feature cols, hiperparámetros, métricas de entrenamiento, hashes de artefactos.

## 5. Bridge C# ↔ Python
Archivo: `predictors/api.py` (FastAPI)

Endpoints:
- `GET /health` — estado y modelo cargado.
- `POST /train` — reentrena con histórico actual.
- `POST /predict` — devuelve probabilidades y expected goals.

Cliente C#: `Oloraculo.Web/Services/XGBoostBridgeService.cs`

## 6. Predicción en C#
`Oloraculo.Web/Predictors/XGBoostPredictor.cs` implementa `IPredictor` y consume el bridge Python como predictor principal. La app .NET conserva predictores clásicos (Elo, Poisson, etc.) en la escalera, pero el pipeline canónico Python se basa exclusivamente en XGBoost.

## 7. Salidas
- Probabilidades 1X2 normalizadas.
- Expected goals y distribución de marcadores.
- Picks con confianza.
- README/JSON con métricas diarias.

## Prevención de data leakage
- Todas las features usan `as_of_date`: solo datos anteriores al partido.
- Elo se actualiza **después** de guardar el rating previo.
- Forma reciente usa `shift(1)`.
- H2H excluye el partido actual.
- Train/test split temporal: entrenar con datos previos al torneo a evaluar.

## Monitoreo
- `drift-detector` compara distribución de features vs baseline.
- `elo-auditor` compara ratings locales vs `eloratings.net`.
- `player-data-auditor` valida stats de jugadores.
