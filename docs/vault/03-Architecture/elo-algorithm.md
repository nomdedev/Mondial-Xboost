# Algoritmo de Elo en Oloráculo xBoost

## Objetivo
Estimar la fuerza relativa de cada selección nacional a partir del historial de resultados. El valor de Elo se usa como feature principal (`elo_diff`) en el modelo XGBoost y como predictor baseline en la escalera C#.

## Fuente de datos
- `Oloraculo.Web/Data/historical_results.csv` (~49K partidos internacionales desde 1872).
- Columnas requeridas: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `neutral`.

## Implementación
Archivo: `predictors/feature_engineering.py` → `compute_elo_ratings()`

### 1. Estado inicial
- Cada equipo comienza con **Elo = 1500**.
- Los ratings se actualizan iterativamente en orden cronológico.

### 2. Factor K por importancia del partido
| Tipo de partido | K |
|-----------------|---|
| Mundial / Copa del Mundo | 60 |
| Continental (Euro, Copa América, Asian Cup, Africa Cup) | 50 |
| Eliminatorias | 40 |
| Grandes ligas (football-data.co.uk) | 30 |
| Amistoso / otros | 20 |

> **Nota de data leakage:** El K depende únicamente del torneo, **no** del resultado ni de la diferencia de goles. Ajustar K por goles del partido actual introduciría leakage porque los goles son el label.

### 3. Ventaja local
Se añaden **+100 puntos** al rating del equipo local al calcular el resultado esperado:

```
dr = (home_elo - away_elo) + 100
expected_home = 1 / (1 + 10^(-dr / 400))
```

### 4. Actualización
```
result_home = 1 si home_score > away_score
              0.5 si empate
              0 si away_score > home_score
delta = K * (result_home - expected_home)
home_elo  += delta
away_elo  -= delta
```

### 5. Flags de confianza
- `home_elo_provisional` / `away_elo_provisional`: `True` si el equipo ha jugado < 30 partidos en el histórico.

## Outputs
- `home_elo_before`, `away_elo_before`: ratings previos al partido (sin leakage).
- `elo_diff`: diferencia local - visitante.

## Validación
- Comparación periódica con **World Football Elo Ratings** (`eloratings.net`) mediante `scripts/compare_elo_worldfootball.py`.
- Métrica objetivo: diferencia media absoluta < 100 puntos para equipos con ≥30 partidos.

## Diferencias con World Football Elo
| Aspecto | Oloráculo | World Football Elo |
|---------|-----------|--------------------|
| K base | 20-60 por torneo | 20-60 por importancia |
| Ajuste por goles | No (evita leakage) | Sí (factor G) |
| Home advantage | +100 fijo | ~+100 (puede variar) |
| Actualización | Solo histórico propio | Histórico completo y curado |
| Equipos cubiertos | 336+ (incluye no FIFA) | 244 reconocidas |

## Backtest con Elo-only
El predictor Elo baseline asigna probabilidades usando `expected_home` y empate fijo ~25%. Sirve de referencia para medir el valor añadido de XGBoost.
