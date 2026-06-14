# Estrategia de investigación: predicción para todos los partidos

## Estado actual (baseline)

El motor canónico `XGBoostFootballPredictor` predice cualquier partido porque `build_features_for_prediction` rellena valores faltantes con defaults cuando un equipo no tiene historial. Sin embargo, **la calidad de esas predicciones es baja en ciertos escenarios**:

- **Cold-start real**: el training dataset usa `min_date="2010-01-01"`, por lo que el modelo casi nunca ve equipos con `matches_played == 0` durante el entrenamiento. Los defaults (0.5 pts avg, 1.3 goles, Elo 1500) son ciegos.
- **Overfitting temporal**: un split temporal simple (`< 2024` vs `>= 2024`) da ~46% accuracy, muy por debajo del ~62% del training set. El modelo no generaliza bien al futuro.
- **Elo truncado**: los ratings Elo se reinician en 1500 para cada equipo al inicio del training window (2010), perdiendo toda la información previa.
- **Sin features externas**: no usamos ranking FIFA, valor de plantel, edad promedio, Confederación, ni contexto del torneo más allá de un K-factor aproximado.
- **Targets únicos**: solo entrenamos outcome 1X2 + goles. No aprovechamos mercados correlacionados (over/under, BTTS) como regularización.

### Métricas de referencia

| Split | Accuracy | Log loss | Notas |
|-------|----------|----------|-------|
| Training (fit en 2010-2023, eval en el mismo set) | ~62% | ~0.81 | In-sample; no es válido para comparar. |
| Temporal (< 2024 train, >= 2024 test) — baseline Elo reiniciado en 2010 | 46.1% | 1.06 | Colapsa a predecir siempre away. |
| Temporal (< 2024 train, >= 2024 test) — Exp-01 Elo/H2H global | 59.2% | 0.91 | Mejora fuerte; distribución más balanceada. |

El gap entre el modelo canónico (entrenado con 2010-2026 y evaluado en 2024-2026) y una validación honesta demuestra que el baseline tiene leakage temporal. El verdadero problema es la **generalización al futuro**, no la cobertura de predicción.

## Principios del loop de investigación

1. **Métrica única de éxito**: mejorar el accuracy en un split temporal estricto (últimos 2-3 años no vistos en training).
2. **Un cambio a la vez**: cada experimento modifica una sola hipótesis para poder atribuir el efecto.
3. **Backtest obligatorio**: todo cambio debe pasar por `mondial backtest` o un split temporal antes de tocar el modelo canónico.
4. **Documentación antes del código**: cada experimento se describe en este vault con hipótesis, riesgo esperado y resultados.
5. **No tocar `xgboost_football`**: los experimentos usan nombres como `xgboost_football_exp_<id>`.

## Experimentos arriesgados propuestos

Ordenados por relación riesgo/impacto esperado.

### 1. Elo y H2H globales (bajo riesgo, alto impacto probable)
**Hipótesis**: perder el historial Elo previo a 2010 es costoso. Computar Elo y H2H desde 1872 pero entrenar el modelo solo desde 2010 debería mejorar las features de equipos con pocos partidos modernos.

**Cambio**:
- `compute_elo_ratings` y `_compute_h2h` reciben todo `historical_results.csv`.
- El corte `min_date` se aplica solo al seleccionar las filas de entrenamiento, no al computar features.
- Agregar feature `total_historical_matches` del equipo (partidos previos totales, no solo desde 2010).

**Riesgo**: datos antiguos pueden introducir sesgos (reglas, competición, K diferentes). Mitigación: usar solo para Elo/H2H, no para rolling stats.

### 2. Modelo de cold-start explícito (riesgo medio, alto impacto)
**Hipótesis**: un modelo entrenado exclusivamente con features estáticas/generalizables (Elo global, ranking FIFA, Confederación, población, GDP proxy) predice mejor partidos donde faltan stats recientes.

**Cambio**:
- Entrenar `XGBoostColdStartPredictor` con el subset de partidos donde al menos un equipo tiene <10 partidos previos (en el training set).
- Features: elo_diff global, ranking FIFA diff, neutral, h2h histórico, confederación del home/away.
- En predicción, si `home_matches_played < 10` o `away_matches_played < 10`, blendear con el modelo cold-start.

**Riesgo**: pocos datos de cold-start en el training moderno. Mitigación: data augmentation o downsamplear partidos con historial.

### 3. Embeddings de equipos (riesgo medio-alto, alto impacto)
**Hipótesis**: un embedding aprendido por equipo captura mejor su "identidad" que un rating Elo único.

**Cambio**:
- Agregar capa de embeddings de equipos con una red neuronal simple (PyTorch/TensorFlow) o usar `category_encoder` en XGBoost.
- Entrenar junto con las features existentes.

**Riesgo**: overfitting con 336 equipos. Mitigación: regularización fuerte, embeddings de baja dimensión (8-16), dropout.

### 4. Features de torneo y contexto (riesgo bajo, impacto medio)
**Hipótesis**: el tipo de partido (amistoso vs eliminatoria vs Mundial) y la importancia del torneo afectan el resultado más allá del K-factor.

**Cambio**:
- `tournament` one-hot o target encoding.
- `round` / `stage` del torneo.
- `days_since_last_match` para cada equipo (fatiga/descanso).
- Distancia geográfica / husos horarios (proxy de viaje).

### 5. Multi-task learning: 1X2 + over/under + BTTS (riesgo medio, impacto medio-alto)
**Hipótesis**: predecir múltiples mercados correlacionados actúa como regularización y mejora la calibración de probabilidades.

**Cambio**:
- Extender `XGBoostFootballPredictor` para entrenar 5 targets: outcome, home_goals, away_goals, btts, over_2_5.
- Usar las probabilidades de goles para derivar outcome en lugar de solo el clasificador.

### 6. Blend probabilístico Poisson-XGBoost (riesgo medio, impacto medio)
**Hipótesis**: un modelo Poisson bayesiano simple calibra mejor las colas (goles) que XGBoost puro.

**Cambio**:
- Entrenar un modelo Poisson independiente con Elo diff y neutral.
- Combinar probabilidades de outcome y goles vía stacking con pesos aprendidos en validación.

### 7. Data augmentation para cold-start (riesgo alto, impacto potencialmente alto)
**Hipótesis**: el modelo no aprende a manejar equipos nuevos porque no los ve en entrenamiento.

**Cambio**:
- Generar partidos sintéticos donde se oculta el historial reciente de un equipo conocido (máscara de cold-start).
- Entrenar con mezcla de datos reales y sintéticos.

**Riesgo**: los datos sintéticos pueden distorsionar la distribución. Mitigación: solo ocultar stats, mantener Elo real.

### 8. Ampliar training window a todo el historial (riesgo alto, impacto incierto)
**Hipótesis**: más datos siempre ayudan, incluso de épocas con reglas diferentes.

**Cambio**:
- Entrenar con todo `historical_results.csv` (1872-2026).

**Riesgo**: cambios estructurales en el fútbol a lo largo de 150 años. Mitigación: samplear con pesos mayores a partidos recientes.

## Pipeline de experimentos

```bash
# Crear rama/experimento
mondial entrenar --name xgboost_football_exp_01_elo_global
mondial backtest
mondial test
mondial gates
```

Todo experimento debe reportar en `docs/vault/05-Research/exp-NN-descripcion.md`:
- Hipótesis.
- Cambios realizados.
- Accuracy y log loss en split temporal.
- Diferencia vs baseline.
- Decisión: adoptar / descartar / iterar.

## Resultados de experimentos

| # | Experimento | Accuracy honesta | Log loss | Decisión |
|---|-------------|------------------|----------|----------|
| 01 | Elo/H2H global | 59.19% | 0.906 | Prometedor; ver [`exp-01-elo-global.md`](./exp-01-elo-global.md). |
| **02** | **Elo decay + Elo reciente** | **61.97%** | **0.820** | **Adoptado como canónico**; ver [`exp-02-elo-decay.md`](./exp-02-elo-decay.md). |
| 03 | Cold-start model | 54.48% (solo) / 61.85% (blend) | 1.014 / 0.822 | No adoptado; ver [`exp-03-cold-start.md`](./exp-03-cold-start.md). |

## Próximos experimentos prioritarios

1. **Exp-03: Modelo de cold-start explícito** — entrenar un submodelo para partidos donde falta historial reciente, usando Elo global, ranking FIFA proxy y confederación.
2. **Exp-04: Features de contexto del torneo** — importancia del partido, descanso, distancia geográfica, fase del torneo.
3. **Exp-05: Multi-task learning** — predecir simultáneamente 1X2, BTTS y over/under.
4. **Exp-06: Blend Poisson-XGBoost** — combinar probabilidades de goles con un modelo Poisson bayesiano.
5. **Exp-07: Data augmentation para cold-start** — generar ejemplos sintéticos de equipos sin historial.
6. **Exp-08: Entrenar con todo el historial** — usar datos desde 1872 (riesgo alto por cambios estructurales).
