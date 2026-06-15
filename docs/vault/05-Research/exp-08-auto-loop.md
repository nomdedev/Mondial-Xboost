# Exp-08: Auto Loop Engineering — xgboost_football_exp_08_auto_loop

## Hipótesis

Un loop de tuning automatizado puede descubrir hiperparámetros superiores al canónico y estabilizarlos contra el overfitting, generando un modelo reproducible y documentado.

## Cambios realizados

- Se ejecutó un batch de Optuna sobre XGBoost.
- Se analizaron 369 trials con un score compuesto (accuracy - 0.5·overfit_gap - 0.05·log_loss).
- Se generó una estrategia estabilizada a partir del mejor trial.
- Se reentrenó y guardó el modelo `xgboost_football_exp_08_auto_loop`.

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
| Test accuracy promedio | 59.75% |
| Overfit gap promedio | 10.93% |
| Log loss promedio | 0.8924 |

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

## Conclusiones

- El mejor trial alcanzó 60.64% de test accuracy.
- La estrategia estabilizada usa 100 estimadores, max_depth=3, learning_rate=0.0369.
- Feature más importante: **h2h_last_result**.

## Decisión

_Pendiente de revisión humana: adoptar / descartar / iterar._

## Comandos

```bash
# Reproducir este experimento
./mondial auto-loop --trials 100 --name xgboost_football_exp_08_auto_loop

# Usar el modelo entrenado
./mondial predecir --home Brazil --away Morocco --model xgboost_football_exp_08_auto_loop
```
