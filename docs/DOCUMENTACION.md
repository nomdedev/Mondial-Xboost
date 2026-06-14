# Mondial-Xboost - Documentación del Sistema

## Índice

1. [Visión General](#visión-general)
2. [Fuentes de Datos](#fuentes-de-datos)
3. [Feature Engineering](#feature-engineering)
4. [Modelos XGBoost](#modelos-xgboost)
5. [Pesos y Calibración](#pesos-y-calibración)
6. [Flujo de Predicción](#flujo-de-predicción)
7. [Métricas y Validación](#métricas-y-validación)
8. [Sistema de Agentes](#sistema-de-agentes)

---

## Visión General

Mondial-Xboost es un sistema de predicción de partidos de fútbol basado en machine learning. Combina datos históricos, estadísticas de equipo, y modelos XGBoost para generar probabilidades de outcomes.

### Arquitectura

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Fuentes Datos  │────▶│  ETL Pipeline │────▶│  Feature Eng.   │
│  (CSV/Wikipedia)│     │  (Limpieza)   │     │  (21 features)  │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│    Dashboard    │◀────│  Bridge C#   │◀────│  XGBoost (5     │
│  (React/HTML)   │     │  (FastAPI)   │     │   modelos)      │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

### Stats del Sistema

- **42,315 partidos** en dataset (21 ligas, 2012-2026)
- **18,870 partidos** con features completos para entrenamiento
- **21 features** por partido
- **5 modelos** XGBoost (outcome, home_goals, away_goals, btts, over_2.5)
- **Accuracy actual**: 44.65% (baseline random: 33%)

---

## Fuentes de Datos

### football-data.co.uk (GRATIS)

CSVs descargables directamente. Cobertura desde 2012.

**Ligas disponibles:**
- Europa: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie, Primeira Liga, Süper Lig, Super League Greece, Scottish Premiership, Bundesliga Austria, Superliga Denmark, Veikkausliiga Finland, Eliteserien Norway, Allsvenskan Sweden, Super League Switzerland, Ekstraklasa Poland, Liga I Romania, Premier League Russia
- América: Primera División Argentina, Série A Brazil, Liga MX, MLS
- Asia: Super League China, J-League
- Otros: Premier Division Ireland

**Veracidad:** Fuente establecida desde 2001. Usada en papers académicos. Cuotas verificables contra bookmakers.

### Wikipedia (GRATIS)

Datos históricos de Mundiales (2022, 2018, 2014). Crowdsourced con referencias cruzadas.

**Veracidad:** Historial de ediciones público. Referencias a FIFA.com. Densidad de verificación alta en eventos recientes.

### Calidad de Datos

| Dimensión | Estado | Detalle |
|-----------|--------|---------|
| Veracidad | ✅ Alta | Fuentes verificables, referencias cruzadas |
| Completitud | ✅ Media-Alta | 73k crudos → 42k únicos → 18k con features |
| Actualización | ⚠️ Manual | Pipeline diario a las 20:00 ART |
| Sesgo | ⚠️ Posible | Ligas europeas sobre-representadas históricamente |

---

## Feature Engineering

### Elo Ratings

Sistema de rating dinámico que se actualiza después de cada partido.

**Fórmulas:**
```
E_A = 1 / (1 + 10^((R_B - R_A)/400))
R'_A = R_A + K * (S_A - E_A)
```

- **K = 30** (factor de actualización)
- **R_inicial = 1500** (rating inicial)
- **S_A = 1** (win), **0.5** (draw), **0** (loss)

### Forma Reciente (Rolling Windows)

Rolling windows de 5 y 10 partidos usando solo información anterior al partido a predecir.

**Features calculadas:**
- `points_avg_5/10`: Promedio de puntos (normalizado a 0-1)
- `goals_scored_avg_10`: Promedio goles anotados
- `goals_conceded_avg_10`: Promedio goles recibidos
- `win_rate_10`: Tasa de victorias
- `draw_rate_10`: Tasa de empates
- `loss_rate_10`: Tasa de derrotas

**Protección contra leakage:** `shift(1)` en rolling windows + filtro `date < match_date`.

### Head-to-Head

Historial directo entre dos equipos.

**Features:**
- `h2h_last_result`: Último resultado (1=home win, 0.5=draw, 0=away win)
- `h2h_goals_avg`: Promedio goles en enfrentamientos
- `h2h_wins_diff`: Diferencia de victorias (home - away)
- `h2h_years_since`: Años desde último enfrentamiento

### Lista Completa de Features (21)

| Feature | Tipo | Descripción |
|---------|------|-------------|
| elo_diff | Contexto | Diferencia de Elo (home - away) |
| home_points_avg_5 | Forma | Promedio puntos últimos 5 partidos |
| home_points_avg_10 | Forma | Promedio puntos últimos 10 partidos |
| home_goals_scored_avg_10 | Forma | Promedio goles anotados (10) |
| home_goals_conceded_avg_10 | Forma | Promedio goles recibidos (10) |
| home_win_rate_10 | Forma | Tasa de victorias (10) |
| home_draw_rate_10 | Forma | Tasa de empates (10) |
| home_loss_rate_10 | Forma | Tasa de derrotas (10) |
| home_matches_played | Contexto | Partidos jugados en historial |
| away_points_avg_5 | Forma | Promedio puntos últimos 5 (away) |
| away_points_avg_10 | Forma | Promedio puntos últimos 10 (away) |
| away_goals_scored_avg_10 | Forma | Promedio goles anotados (away, 10) |
| away_goals_conceded_avg_10 | Forma | Promedio goles recibidos (away, 10) |
| away_win_rate_10 | Forma | Tasa victorias away (10) |
| away_draw_rate_10 | Forma | Tasa empates away (10) |
| away_loss_rate_10 | Forma | Tasa derrotas away (10) |
| away_matches_played | Contexto | Partidos jugados away |
| h2h_last_result | H2H | Último resultado entre equipos |
| h2h_goals_avg | H2H | Promedio goles en enfrentamientos |
| h2h_wins_diff | H2H | Diferencia de victorias |
| h2h_years_since | H2H | Años desde último enfrentamiento |

---

## Modelos XGBoost

### Arquitectura

| Modelo | Tipo | Target | Notas |
|--------|------|--------|-------|
| Outcome | Multi-class | Resultado 1X2 | Con calibración isotónica por defecto |
| Home Goals | Regression | Goles local | Esperanza continua |
| Away Goals | Regression | Goles visitante | Esperanza continua |

> Los modelos BTTS y Over 2.5 del motor legacy (`scripts/archive/xgboost_predictor_legacy.py`)
> no son parte del pipeline canónico actual.

### Hiperparámetros Actuales

```python
{
    "n_estimators": 200,        # Número de árboles
    "max_depth": 6,             # Profundidad máxima
    "learning_rate": 0.1,       # Tasa de aprendizaje
    "subsample": 0.8,           # Fracción de muestras por árbol
    "colsample_bytree": 0.8,    # Fracción de features por árbol
    "reg_alpha": 0,             # Regularización L1
    "reg_lambda": 1,            # Regularización L2
    "min_child_weight": 1,      # Mínimo peso hijo
    "gamma": 0,                 # Reducción mínima de loss
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": 42
}
```

### Proceso de Entrenamiento

1. **Split Temporal**: 80% entrenamiento (partidos antiguos) → 20% test (partidos recientes)
2. **Feature Matrix**: 21 features numéricas. Sin categorización de equipos.
3. **XGBoost Training**: Gradient boosting con 200 árboles. Early stopping en 10 rounds.
4. **Calibración**: Temperatura scaling en probabilidades.

### Feature Importance (Modelo Actual)

| Feature | Importancia | Descripción |
|---------|-------------|-------------|
| elo_diff | 7.46% | Diferencia de Elo |
| home_matches_played | 4.99% | Experiencia local |
| home_goals_scored_avg_10 | 4.91% | Potencia ofensiva |
| away_matches_played | 4.91% | Experiencia visitante |
| away_points_avg_10 | 4.89% | Forma visitante |
| h2h_years_since | 4.85% | Recencia H2H |
| home_points_avg_10 | 4.84% | Forma local |
| away_win_rate_10 | 4.83% | Tasa victorias away |
| home_win_rate_10 | 4.78% | Tasa victorias local |
| h2h_goals_avg | 4.61% | Promedio goles H2H |

---

## Pesos y Calibración

### Temperatura Scaling

Ajusta la "confianza" del modelo. T > 1 = más conservador, T < 1 = más confiado.

```python
P_calibrada = softmax(logits / T)
```

T se optimiza minimizando NLL en validation set.

### Home Advantage

Ajuste empírico basado en historial. Varía por liga.

```python
P_home_adj = P_home * (1 + HA_factor)
```

- HA_factor ≈ 0.15 para ligas europeas
- HA_factor ≈ 0.10 para internacionales (mundiales)

### Ensemble Weights

Cuando múltiples modelos predicen, combinamos con pesos basados en performance:

```python
P_final = Σ(w_i * P_i) / Σ(w_i)
w_i = exp(-loss_i) / Σ(exp(-loss_j))
```

---

## Flujo de Predicción

### Paso 1: Input del Usuario

```json
{
    "home_team": "Argentina",
    "away_team": "Brasil",
    "date": "2026-06-15"
}
```

### Paso 2: Búsqueda de Historial

El sistema busca partidos históricos de ambos equipos.

**Ejemplo:**
- Argentina (Home): Últimos 10 partidos: 7V-2E-1D, Goles: 18 anotados, 5 recibidos, Elo: 1850
- Brasil (Away): Últimos 10 partidos: 6V-3E-1D, Goles: 15 anotados, 7 recibidos, Elo: 1820

### Paso 3: Generación de Features

Se calculan las 21 features para este partido específico.

```
elo_diff = 30
home_points_avg_10 = 0.77
home_goals_scored_avg_10 = 1.8
away_points_avg_10 = 0.70
h2h_last_result = 1.0 (Argentina ganó último)
...
```

### Paso 4: Predicción XGBoost

El modelo genera probabilidades para cada outcome.

**Ejemplo:**
- Home Win: 45%
- Draw: 28%
- Away Win: 27%

### Paso 5: Calibración y Output

Ajuste de probabilidades y generación de recomendaciones.

**Después de calibración:**
- Home Win: 48%
- Draw: 26%
- Away Win: 26%

**Recomendación:** Argentina (moderate confidence)

---

## Métricas y Validación

### Métricas Actuales

| Métrica | Valor | Descripción | Objetivo |
|---------|-------|-------------|----------|
| Accuracy | 44.65% | % predicciones correctas | > 50% |
| Log Loss | 1.08 | Penaliza confianza errónea | < 1.0 |
| Brier Score | 0.22 | Error cuadrático medio | < 0.20 |

### Explicación de Métricas

**Accuracy:**
```
Accuracy = (predicciones correctas) / (total predicciones)
```
Baseline random: 33%. Nuestro modelo: 44.65%.

**Log Loss:**
```
LogLoss = -Σ(y_true * log(y_pred)) / N
```
Penaliza más las predicciones confiadas que están equivocadas. Ideal: < 1.0.

**Brier Score:**
```
Brier = Σ(y_pred - y_true)² / N
```
Error cuadrático medio. Mide calibración. Ideal: ≈ 0.2.

### Estrategia de Validación

**Temporal Split:**
- 80% entrenamiento (partidos más antiguos)
- 20% test (partidos más recientes)
- No shuffle (simula uso real)

**Walk-Forward Validation:**
- Entrenar hasta 2022, testear 2023
- Entrenar hasta 2023, testear 2024
- Valida estabilidad temporal

---

## Sistema de Agentes

### Agentes Disponibles

| Agente | Rol | Tareas |
|--------|-----|--------|
| Data Agent | Ingesta | Busca nuevas fuentes, valida calidad, detecta gaps |
| ML Agent | Entrenamiento | Optimiza hiperparámetros, prueba features, compara modelos |
| Analysis Agent | Evaluación | Analiza predicciones, detecta sesgos, sugiere mejoras |

### Loop de Mejora Automática

1. **Exploración**: Prueba combinaciones aleatorias de hiperparámetros
2. **Entrenamiento**: Cada configuración se entrena y evalúa
3. **Selección**: Mantiene el modelo con mejor métrica objetivo
4. **Reporte**: Resultados guardados en JSON para análisis

### Espacio de Hiperparámetros

| Parámetro | Valores Posibles |
|-----------|------------------|
| n_estimators | [100, 200, 300, 500] |
| max_depth | [4, 6, 8, 10] |
| learning_rate | [0.05, 0.1, 0.15, 0.2] |
| subsample | [0.7, 0.8, 0.9] |
| colsample_bytree | [0.7, 0.8, 0.9] |

**Total combinaciones:** 576

### Métricas Objetivo

- **accuracy**: Maximizar
- **log_loss**: Minimizar
- **brier_score**: Minimizar

---

## Archivos del Proyecto

```
Mondial-Xboost/
├── api/
│   └── main.py                 # FastAPI server
├── dashboard/
│   └── index.html              # Dashboard web
├── docs/
│   └── index.html              # Documentación (este archivo)
├── models/
│   └── xgboost_outcome_v1.pkl  # Modelo entrenado
├── predictors/
│   ├── feature_engineering.py  # Feature generation
│   ├── xgboost_predictor.py   # Model training
│   ├── etl.py                 # Data pipeline
│   └── llm_analysis_service.py # LLM integration
├── scrapers/
│   ├── football_data_scraper.py
│   ├── wikipedia_scraper.py
│   └── news_scraper_service.py
├── data/
│   ├── raw/                   # CSVs descargados
│   ├── processed/             # Parquet files
│   └── features/              # Feature matrices
└── start-platform.sh          # Startup script
```

---

## Loop Engineering

Sistema de calidad ejecutable para garantizar reproducibilidad antes de mergear o
desplegar.

### Scripts de gates

| Script | Propósito | Evidencia |
|--------|-----------|-----------|
| `scripts/verify_gates.py` | Orquestador de gates | `.agents/logs/pipeline-state.json` |
| `scripts/run_data_council.py` | Calidad de datos | `.agents/logs/data-council-report.json` |
| `scripts/run_backtest_gate.py` | Backtest con thresholds | `backtest/results/world_cup_backtest_summary.json` |
| `scripts/run_bridge_smoke_test.py` | Integridad C#↔Python | `backtest/results/bridge_smoke.json` |

### Thresholds actuales

| Métrica | Threshold |
|---------|-----------|
| log-loss promedio | < 1.05 |
| Brier score | < 0.22 |
| Top-1 accuracy | > 45% |
| ROI simulado | > -5% |
| Elo top-100 mean abs diff | < 100 pts |
| Elo top-100 correlation | > 0.90 |

Ver detalles en `.agents/skills/loop-engineering/SKILL.md`.

---

## Próximos Pasos

1. **Mejorar accuracy**: Actual 44.65% → Target 50%+
2. **Agregar features**: xG, posesión, tiros a puerta, valor de mercado
3. **Calibración**: Implementar temperatura scaling automático
4. **Más modelos**: Poisson regression, Bradley-Terry
5. **Datos en tiempo real**: Integrar API-Football para alineaciones

---

*Documentación generada automáticamente. Última actualización: 2024-01-15*
