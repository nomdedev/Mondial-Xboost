# Exp-05: Auto Loop Engineering — xgboost_football_exp_05_auto_loop

## Hipótesis

Un loop de tuning automatizado puede descubrir hiperparámetros superiores al canónico y estabilizarlos contra el overfitting, generando un modelo reproducible y documentado.

## Cambios realizados

- Se ejecutó un batch de Optuna sobre XGBoost.
- Se analizaron 117 trials con un score compuesto (accuracy - 0.5·overfit_gap - 0.05·log_loss).
- Se generó una estrategia estabilizada a partir del mejor trial.
- Se reentrenó y guardó el modelo `xgboost_football_exp_05_auto_loop`.

### Ajustes de estrategia aplicados

- subsample=1.0: se reduce a 0.9 para bajar varianza

## Métricas

### Tuning

| Métrica | Valor |
| --- | --- |
| Mejor test accuracy | 60.52% |
| Log loss (mejor) | 0.8781 |
| Brier (mejor) | 0.1719 |
| Overfit gap (mejor) | -0.72% |
| Walk-forward acc (mejor) | 59.28% |
| Score compuesto (mejor) | 0.5649 |
| Test accuracy promedio | 59.63% |
| Overfit gap promedio | 11.53% |
| Log loss promedio | 0.8963 |

### Modelo final (entrenamiento)

| Métrica | Valor |
| --- | --- |
| Filas usadas | 15,794 |
| Accuracy entrenamiento | 0.5943 |
| Log loss entrenamiento | 0.8740 |
| Feature top | elo_diff (0.2431) |

### Hiperparámetros finales

| Parámetro | Valor |
| --- | --- |
| n_estimators | 584 |
| max_depth | 3 |
| learning_rate | 0.0103 |
| subsample | 0.9 |
| colsample_bytree | 0.6224 |
| reg_lambda | 0.0037 |
| reg_alpha | 7.9342 |
| min_child_weight | 1 |
| gamma | 0.804 |

## Conclusiones

- El mejor trial alcanzó 60.52% de test accuracy.
- La estrategia estabilizada usa 584 estimadores, max_depth=3, learning_rate=0.0103.
- Feature más importante: **elo_diff**.

## Decisión

_Pendiente de revisión humana: adoptar / descartar / iterar._

## Comandos

```bash
# Reproducir este experimento
./mondial auto-loop --trials 100 --name xgboost_football_exp_05_auto_loop

# Usar el modelo entrenado
./mondial predecir --home Brazil --away Morocco --model xgboost_football_exp_05_auto_loop
```
