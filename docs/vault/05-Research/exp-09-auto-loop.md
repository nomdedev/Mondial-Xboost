# Exp-09: Auto Loop Engineering — xgboost_football_exp_09_auto_loop

## Hipótesis

Un loop de tuning automatizado puede descubrir hiperparámetros superiores al canónico y estabilizarlos contra el overfitting, generando un modelo reproducible y documentado.

## Cambios realizados

- Se ejecutó un batch de Optuna sobre XGBoost.
- Se analizaron 469 trials con un score compuesto (accuracy - 0.5·overfit_gap - 0.05·log_loss).
- Se generó una estrategia estabilizada a partir del mejor trial.
- Se reentrenó y guardó el modelo `xgboost_football_exp_09_auto_loop`.

## Métricas

### Tuning

| Métrica | Valor |
| --- | --- |
| Mejor test accuracy | 60.60% |
| Log loss (mejor) | 0.8806 |
| Brier (mejor) | 0.1721 |
| Overfit gap (mejor) | -1.04% |
| Walk-forward acc (mejor) | 59.24% |
| Score compuesto (mejor) | 0.5672 |
| Test accuracy promedio | 59.77% |
| Overfit gap promedio | 10.57% |
| Log loss promedio | 0.8917 |

### Modelo final (entrenamiento)

| Métrica | Valor |
| --- | --- |
| Filas usadas | 15,794 |
| Accuracy entrenamiento | 0.5925 |
| Log loss entrenamiento | 0.8778 |
| Feature top | h2h_last_result (0.2142) |

### Hiperparámetros finales

| Parámetro | Valor |
| --- | --- |
| n_estimators | 128 |
| max_depth | 3 |
| learning_rate | 0.0241 |
| subsample | 0.7922 |
| colsample_bytree | 0.6749 |
| reg_lambda | 0.0881 |
| reg_alpha | 0.0594 |
| min_child_weight | 6 |
| gamma | 0.4829 |

## Conclusiones

- El mejor trial alcanzó 60.60% de test accuracy.
- La estrategia estabilizada usa 128 estimadores, max_depth=3, learning_rate=0.0241.
- Feature más importante: **h2h_last_result**.

## Decisión

_Pendiente de revisión humana: adoptar / descartar / iterar._

## Comandos

```bash
# Reproducir este experimento
./mondial auto-loop --trials 100 --name xgboost_football_exp_09_auto_loop

# Usar el modelo entrenado
./mondial predecir --home Brazil --away Morocco --model xgboost_football_exp_09_auto_loop
```
