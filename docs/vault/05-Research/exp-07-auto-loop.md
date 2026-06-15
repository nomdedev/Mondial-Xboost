# Exp-07: Auto Loop Engineering — mondial v1

## Hipótesis

Un loop de tuning automatizado puede descubrir hiperparámetros superiores al canónico y estabilizarlos contra el overfitting, generando un modelo reproducible y documentado.

## Cambios realizados

- Se ejecutó un batch de Optuna sobre XGBoost.
- Se analizaron 269 trials con un score compuesto (accuracy - 0.5·overfit_gap - 0.05·log_loss).
- Se generó una estrategia estabilizada a partir del mejor trial.
- Se reentrenó y guardó el modelo `mondial v1`.

## Métricas

### Tuning

| Métrica | Valor |
| --- | --- |
| Mejor test accuracy | 60.64% |
| Log loss (mejor) | 0.8773 |
| Brier (mejor) | 0.1715 |
| Overfit gap (mejor) | -0.85% |
| Walk-forward acc (mejor) | 0.00% |
| Score compuesto (mejor) | 0.5668 |
| Test accuracy promedio | 59.69% |
| Overfit gap promedio | 11.92% |
| Log loss promedio | 0.8934 |

### Modelo final (entrenamiento)

| Métrica | Valor |
| --- | --- |
| Filas usadas | 15,794 |
| Accuracy entrenamiento | 0.5944 |
| Log loss entrenamiento | 0.8754 |
| Feature top | h2h_last_result (0.2052) |

### Hiperparámetros finales

| Parámetro | Valor |
| --- | --- |
| n_estimators | 100 |
| max_depth | 3 |
| learning_rate | 0.0369 |
| subsample | 0.7731 |
| colsample_bytree | 0.6699 |
| reg_lambda | 1.8002 |
| reg_alpha | 0.0044 |
| min_child_weight | 8 |
| gamma | 0.036 |

### World Cup backtest

| Métrica | Valor |
| --- | --- |
| Verdict | PASS |
| Accuracy promedio | 55.21% |
| Log loss promedio | 0.9924 |
| Brier promedio | 0.1950 |
| ROI simulado | 7.94% |

- WC 2014: acc=59.38%, log_loss=0.9478, roi=23.75%
- WC 2018: acc=56.25%, log_loss=0.9640, roi=7.85%
- WC 2022: acc=50.00%, log_loss=1.0653, roi=-7.76%

## Conclusiones

- El mejor trial alcanzó 60.64% de test accuracy.
- La estrategia estabilizada usa 100 estimadores, max_depth=3, learning_rate=0.0369.
- Feature más importante: **h2h_last_result**.

## Decisión

_Pendiente de revisión humana: adoptar / descartar / iterar._

## Comandos

```bash
# Reproducir este experimento
./mondial auto-loop --trials 100 --name mondial v1

# Usar el modelo entrenado
./mondial predecir --home Brazil --away Morocco --model mondial v1
```
