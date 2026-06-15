# Estrategias para Romper el Techo de 60.80% Accuracy

> Generado el 2026-06-15 tras consultar a 3 agentes expertos (Requirements Analyst, Architect, Database Engineer)

---

## Diagnóstico Actual

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Best test accuracy | 60.80% | Estancado tras 9 experimentos, ~470 trials |
| Best log loss | 0.873 | Gap grande vs casas de apuesta (~0.70-0.75) |
| Overfit gap | ~0-3% | No estamos overfitteando — **underfitteamos** |
| Feature top | h2h_last_result / elo_diff | Dependencia excesiva de 2-3 features |
| Split | 70/15/15 temporal fijo | Sin purging, sin embargo, sin CV combinatoria |

**Conclusión:** No es un problema de hiperparámetros. Para ganar los próximos 5-10 puntos necesitas **información nueva y transformaciones radicales**, no más tuning de Optuna sobre las mismas 23 features.

---

## Ranking Consolidado (Riesgo ÷ Impacto ÷ Esfuerzo)

| # | Estrategia | Riesgo | Impacto Potencial | Esfuerzo | Categoría |
|---|-----------|--------|-------------------|----------|-----------|
| 1 | **Label Smoothing con Priors de Mercado** | ALTO | +0.5-2% acc, -0.02-0.05 log_loss | BAJO | Datos |
| 2 | **Purged CPCV + Temporal Embargo** | MUY ALTO | Revela estabilidad real, evita overfitting a split único | MEDIO | Validación |
| 3 | **Bayesian Shrinkage con Cuotas** | ALTO | +3-6% acc (si hay cuotas en datos) | MEDIO | Features |
| 4 | **Ensemble Two-Speed (Recente + Histórico)** | ALTO | +1-3% acc, mejor calibración mixta | MEDIO | Modelo |
| 5 | **Reframing Poisson + Simulación MC** | ALTO | +2-5% acc, mejor calibración | ALTO | Modelo |
| 6 | **RPS Diferenciable como Loss** | MUY ALTO | +0.5-1% acc, -0.02 log_loss | ALTO | Loss |
| 7 | **Meta-Ensemble Draw Specialist** | MUY ALTO | +2-4% acc (recupera draws) | MEDIO | Modelo |
| 8 | **Auto-Distilación Temporal** | ALTO | +1-2% acc, -0.02 log_loss | ALTO | Modelo |
| 9 | **Grafo + PageRank Temporal** | MUY ALTO | +2-5% acc | ALTO | Features |
| 10 | **Entropy-Weighted + Change-Point** | MUY ALTO | +2-4% acc | ALTO | Features |
| 11 | **Group-Feature Dropout + Gating** | MUY ALTO | Robustez a features faltantes | ALTO | Modelo |
| 12 | **Double Descent XGBoost + Entropy Pen** | MUY ALTO | +2-4% acc (régimen interpolación) | ALTO | Modelo |
| 13 | **Noise-Augmented MoE (Elo)** | MUY ALTO | +1-2% acc en fixtures inciertos | MUY ALTO | Modelo |
| 14 | **Adversarial Time Decay** | MUY ALTO | Elimina leakage temporal | MEDIO | Validación |
| 15 | **Poisson-Skellam Conjunto** | EXTREMO | +3-5% acc (teórico) | MUY ALTO | Modelo |
| 16 | **Autoencoder Latent Features** | MUY ALTO | +1-4% acc | MUY ALTO | Features |
| 17 | **Augmentación Poisson-Gamma** | MUY ALTO | +1-3% acc en cola larga | MUY ALTO | Datos |
| 18 | **Domain Confusion Adversarial** | EXTREMO | Generalización inter-torneo | MUY ALTO | Modelo |
| 19 | **Conformal Prediction** | MUY ALTO | Garantías de cobertura (no acc) | ALTO | Inferencia |

---

## Estrategias Detalladas

---

### 1. Label Smoothing con Priors de Cuotas / Elo

**Riesgo:** ALTO | **Esfuerzo:** BAJO | **Impacto estimado:** +0.5-2% acc, -0.02-0.05 log_loss

**Qué hacer:** Reemplazar etiquetas duras {1,0,0} por suaves: mezcla entre resultado real y prior de mercado (o Elo cuando no hay cuotas). Ratio sugerido: 60% real + 40% prior.

**Hipótesis:** El resultado de un partido individual tiene alta varianza (un rebote cambia todo). Las cuotas de mercado agregan sabiduría colectiva. Etiquetas suaves reducen log_loss porque penalizan menos las predicciones correctas "en espíritu".

**Por qué no lo hemos hecho:** Requiere datos de cuotas históricas (Bet365/Pinnacle) que pueden no estar disponibles en todas las fuentes. Riesgo de over-smoothing si el ratio es incorrecto.

**Implementación:**
```python
def smooth_labels(row, alpha=0.6):
    actual = np.zeros(3)
    actual[row["outcome"]] = 1.0
    if "market_home_prob" in row:
        prior = np.array([row["market_away_prob"], row["market_draw_prob"], row["market_home_prob"]])
    else:
        dr = row["elo_diff"]
        prior = elo_to_probs(dr)
    return alpha * actual + (1 - alpha) * prior
```

---

### 2. Purged Combinatorial Cross-Validation (CPCV)

**Riesgo:** MUY ALTO | **Esfuerzo:** MEDIO | **Impacto estimado:** Revela estabilidad real

**Qué hacer:** Reemplazar el split temporal único por múltiples splits con **embargo** (gap de N días entre train y test). Evaluar media y **desviación estándar** entre folds.

**Hipótesis:** El split actual (2023/2024) da una estimación ruidosa. CPCV revela si el modelo es consistentemente bueno o solo acertó por suerte en esa frontera. Un modelo con 61% pero std > 5% entre folds es frágil.

**Por qué no lo hemos hecho:** Complejidad de implementación, pérdida de datos por embargo (~10-20%), folds correlacionados.

**Implementación:**
```python
def purged_cv(df, n_splits=5, embargo_days=60):
    dates = sorted(df["date"].unique())
    fold_size = len(dates) // (n_splits + 2)
    for i in range(n_splits):
        train_end = dates[(i + 1) * fold_size]
        test_start = dates[(i + 2) * fold_size]
        embargo = (df["date"] >= train_end) & (df["date"] < test_start)
        train = df[df["date"] < train_end]
        test = df[df["date"] >= test_start]
        # ...
```

---

### 3. Bayesian Shrinkage con Cuotas de Apuestas

**Riesgo:** ALTO | **Esfuerzo:** MEDIO | **Impacto estimado:** +3-6% acc

**Qué hacer:** Usar cuotas históricas (Bet365/Pinnacle) como **prior bayesiano**. Entrenar XGBoost normal, luego mezclar su output con las cuotas vía shrinkage: cuando el modelo está inseguro (alta entropía), pesa más las cuotas; cuando está seguro, las ignora.

**Hipótesis:** Las cuotas de Pinnacle tienen ~65-70% de accuracy — si agregas valor ENCIMA de eso, llegas a 65%+. El mercado colectivo captura información que 23 features no pueden (lesiones, clima, ánimo).

**Por qué no lo hemos hecho:** Datos de cuotas no disponibles para todos los partidos. Riesgo de señal falsa (cuotas manipuladas). Si las cuotas contienen TODA la señal, el modelo no aporta nada.

---

### 4. Ensemble Two-Speed (Recente + Histórico)

**Riesgo:** ALTO | **Esfuerzo:** MEDIO | **Impacto estimado:** +1-3% acc

**Qué hacer:** Entrenar 3 modelos: (A) solo últimos 3 años (~5k), (B) histórico completo (~15k), (C) Meta-gate (NN pequeña) que asigna pesos dinámicos por fixture según recencia de cada equipo, torneo, y diferencia Elo.

**Hipótesis:** El fútbol cambia (tácticas, VAR, fitness). Modelo-R captura el estado actual. Modelo-H provee señal para equipos sin historia reciente. Meta-gate aprende cuándo confiar en cada uno.

**Por qué podría fallar:** Modelo-R con solo 5k filas puede overfittear. Meta-gate puede aprender atajos espurios.

---

### 5. Reframing Poisson + Simulación Monte Carlo

**Riesgo:** ALTO | **Esfuerzo:** ALTO | **Impacto estimado:** +2-5% acc

**Qué hacer:** En vez de clasificar 1X2 directamente, modelar la distribución de goles de cada equipo (Poisson o Negative Binomial) y simular 10,000 partidos para derivar probabilidades 1X2. Ya tienes regresores de goles — **integralos al pipeline de clasificación**.

**Hipótesis:** Un 3-0 y un 1-0 son ambos "Home Win" pero tienen dinámicas muy distintas. Modelar goles primero captura granularidad que el clasificador pierde. La simulación produce distribuciones mejor calibradas.

**Por qué no lo hemos hecho:** Ya hay `home_goals_model` y `away_goals_model` pero no se usan para clasificación. Poisson puede estar mal especificado (fútbol moderno tiene menos goles). Simulación cara computacionalmente.

**Implementación:**
```python
def simulate_match(home_lambda, away_lambda, n_sims=10000):
    home_goals = np.random.poisson(home_lambda, n_sims)
    away_goals = np.random.poisson(away_lambda, n_sims)
    home_win = np.mean(home_goals > away_goals)
    draw = np.mean(home_goals == away_goals)
    away_win = 1 - home_win - draw
    return np.array([away_win, draw, home_win])
```

---

### 6. RPS Diferenciable como Loss Function

**Riesgo:** MUY ALTO | **Esfuerzo:** ALTO | **Impacto estimado:** +0.5-1% acc, -0.02 log_loss

**Qué hacer:** Reemplazar `multi:softprob` (cross-entropy) por Ranked Probability Score (RPS) como objetivo custom. RPS penaliza más predecir 80% Home cuando el resultado fue Away que predecir 60% — la cross-entropy no distingue entre estos errores.

**Hipótesis:** 1X2 es ordinal (Away < Draw < Home). Cross-entropy trata las clases como categóricas sin orden. RPS alinea la optimización con la estructura ordinal del dominio.

**Por qué podría fallar:** Custom objectives en XGBoost necesitan gradiente + hessiano por clase. Si la aproximación no es perfecta, el entrenamiento diverge. Es 5-10x más lento.

---

### 7. Meta-Ensemble Draw Specialist

**Riesgo:** MUY ALTO | **Esfuerzo:** MEDIO | **Impacto estimado:** +2-4% acc

**Qué hacer:** Entrenar 4 XGBoost: 3 especialistas (uno por clase, con class weights altos para su clase) + 1 canónico. Meta-learner (LogisticRegression L1) aprende cuándo confiar en cada especialista.

**Hipótesis:** El XGBoost multiclass actual sacrifica recall en draws (~24% de partidos) porque optimiza accuracy global. Un especialista en draw entrenado con focal loss encuentra patrones que el canónico ignora.

**Por qué podría fallar:** Meta-ensemble puede overfittear. Si no hay señal para draws, ningún especialista la encuentra.

---

### 8. Auto-Distilación Temporal

**Riesgo:** ALTO | **Esfuerzo:** ALTO | **Impacto estimado:** +1-2% acc, -0.02 log_loss

**Qué hacer:** Entrenar K=5 modelos XGBoost en ventanas temporales deslizantes (2016, 2018, 2020, 2022, 2024). Cada "profesor" captura una era del fútbol. Destilar todos en un único estudiante que imita el promedio del ensemble (KL-divergence).

**Hipótesis:** Cada era del fútbol (pre-VAR, COVID, post-Moneyball) tiene dinámicas distintas. El estudiante aprende a integrar todas las perspectivas → más robusto.

**Por qué podría fallar:** Si el fútbol no ha cambiado mucho, los profesores son redundantes. La destilación puede propagar errores compartidos.

---

## Priorización Recomendada

### Fase 1 (Próximos 7 días) — Bajo riesgo, alto retorno
1. **Label Smoothing** (solo modificar la loss, no requiere datos nuevos)
2. **Purged CPCV** (sin cambiar el modelo, solo la validación)
3. **Bayesian Shrinkage** (si hay cuotas en los datos históricos)

### Fase 2 (Siguientes 14 días) — Riesgo medio-alto
4. **Ensemble Two-Speed** (complementa al cold-start actual)
5. **Reframing Poisson** (ya tienes los regresores de goles)
6. **Meta-Ensemble Draw Specialist** (ataque directo al talón de Aquiles)

### Fase 3 (Exploratorio) — Riesgo muy alto
7. **RPS Loss** o **Double Descent**
8. **Grafo + PageRank** o **Autoencoder Latent**
9. **Augmentación Poisson-Gamma** o **Domain Confusion**

---

## Modificaciones Inmediatas al Código Existente

### En `loop_engineering.py` (cambios de bajo esfuerzo)

1. **Ampliar espacio de búsqueda de Optuna** para incluir parámetros no explorados:
   - `max_delta_step`, `tree_method='hist'`, `max_bin`, `grow_policy='lossguide'`
   - `num_parallel_tree` (para Random Forest mode dentro de XGBoost)
   - `scale_pos_weight` por clase en multi-class

2. **Función de score compuesto más agresiva**: en lugar de `accuracy - 0.5*overfit - 0.05*log_loss`, probar variantes que penalicen más el log_loss o que incorporen Brier score.

3. **Early stopping real** durante el tuning: usar `eval_set` con `early_stopping_rounds=50` para no desperdiciar trials en modelos que ya overfittean.

### En `feature_engineering.py` (cambios de alto impacto)

4. **Entropy-Weighted Rolling Averages**: Ponderar cada partido en la rolling average por `1 / p_elo(resultado)`. Resultados sorprendentes pesan más porque señalizan cambios reales.

5. **Strength of Schedule (SOS)**: Para cada equipo, el Elo promedio de sus últimos 5 oponentes. Captura si un equipo viene de jugar contra rivales fuertes o débiles.

6. **Features de momentum**: Diferencia entre la forma reciente (últimos 3 partidos) y la forma de temporada (últimos 20). Captura si un equipo está en racha.

---

## Comandos para Ejecutar Pruebas

```bash
# Probar Label Smoothing
python scripts/loop_engineering.py --batch 12 --trials 50

# Probar con walk-forward validation
python scripts/loop_engineering.py --batch 13 --trials 100 --walk-forward

# Auto-loop completo con nueva configuración
python scripts/auto_loop_engineering.py --trials 100 --name exp-10-agressive --backtest
```

---

## Riesgos Globales a Considerar

- **Ruido irreducible del fútbol**: ~25% de partidos son esencialmente aleatorios. 65% puede ser el techo realista.
- **Over-engineering**: Implementar 5 estrategias simultáneamente hace imposible saber cuál funcionó.
- **Deuda técnica**: Cada estrategia agrega complejidad al pipeline. Documentar y mantener es costo real.
- **Data leakage**: Features como "cuotas de mercado" pueden contener información del futuro si no se manejan con cuidado temporal.
