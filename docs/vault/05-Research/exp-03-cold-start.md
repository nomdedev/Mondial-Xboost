# Exp-03: Modelo de cold-start explícito

## Hipótesis

El modelo canónico depende de stats de forma reciente (últimos 5/10 partidos). Cuando un equipo tiene pocos partidos recientes, esas stats son ruidosas o defaults. Un modelo entrenado solo con features estáticas/generalizables debería predecir mejor esos casos.

## Cambios realizados

- `predictors/cold_start_model.py`: nuevo `ColdStartPredictor` basado en XGBoost.
  - Features: Elo global, Elo reciente, neutral, H2H, experiencia histórica/reciente, target encoding de torneo.
  - Entrenado con ejemplos reales de cold-start balanceados con ejemplos warm.
- `predictors/blended_predictor.py`: `BlendedFootballPredictor` que combina canónico y cold-start con un peso suave basado en el déficit de historial reciente.
- `predictors/feature_engineering.py`: agrega `home_recent_matches` y `away_recent_matches` (ventana de 8 años) para detectar cold-start.
- `scripts/predict.py` y `mondial CLI`: flags `--blend` y `--cold-start-only`, comando `entrenar-cold-start`.

## Métricas

Evaluación en split temporal honesto (< 2024 train, >= 2024 test) usando el canónico Exp-02:

| Modelo | Accuracy | Log loss | Notas |
|--------|----------|----------|-------|
| Canónico Exp-02 | 61.97% | 0.820 | Baseline actual |
| Cold-start only | 54.48% | 1.014 | Peor globalmente |
| Blend suave | 61.85% | 0.822 | Similar al canónico |

### Subset cold-start (32 partidos de test)

| Modelo | Accuracy |
|--------|----------|
| Canónico | 68.75% |
| Cold-start only | 71.88% |
| Blend | 68.75% |

El cold-start only es ligeramente mejor en el subset cold, pero la muestra es muy pequeña (32 partidos) para concluir.

## Conclusiones

- El modelo de cold-start **no mejora las métricas globales**.
- La señal de Elo + torneo + H2H ya es capturada eficientemente por el canónico Exp-02.
- Los casos de cold-start real son raros en el training moderno (~500 filas), limitando lo que el submodelo puede aprender.
- El blending suave mantiene el rendimiento del canónico sin degradar, lo cual es positivo.

## Decisión

**No adoptar como canónico por ahora**. Se deja el código disponible para futuras iteraciones, especialmente si se incorporan features externas como ranking FIFA, valor de plantel o confederación.

## Comandos

```bash
# Entrenar cold-start model
./mondial entrenar-cold-start --name cold_start

# Predecir con blend
./mondial predecir --home Brazil --away Morocco --blend

# Predecir solo con cold-start
./mondial predecir --home Brazil --away Morocco --cold-start-only
```
