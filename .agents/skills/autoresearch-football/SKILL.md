---
name: autoresearch-football
description: >
  Loop autónomo de mejora continua para predictores de fútbol en Mondial-Xboost.
  Itera sobre features, hiperparámetros, ensemble weights y prompts de LLM,
  midiendo siempre contra métricas objetivo (log-loss, Brier, RPS, ROI simulado).
version: 1.0.0
---

# Autoresearch Football — Loop de Mejora de Predictores

## Propósito
Ejecutar un ciclo autónomo de **modificar → verificar → conservar/descartar** sobre
cualquier componente del sistema de predicción de Mondial-Xboost, usando
métricas de performance futbolísticas como verdad única.

## Invariantes de seguridad
- Nunca hacer push/deploy sin aprobación explícita del usuario.
- Loop acotado por defecto. Usar `Iteraciones: ilimitadas` solo con autorización.
- Todos los resultados se loguean en `.agents/logs/autoresearch/{fecha-hora}/`.
- Cada experimento debe poder reproducirse (seed fijo, dataset versionado).
- **Prohibido usar fixtures futuros para entrenar** (data leakage).

## Métricas objetivo

| Métrica | Target | Dirección |
|---------|--------|-----------|
| Log-loss | < 0.65 | Menor es mejor |
| Brier score | < 0.20 | Menor es mejor |
| RPS | < 0.18 | Menor es mejor |
| Top-pick accuracy | > 45% | Mayor es mejor |
| ROI simulado | > 5% | Mayor es mejor |
| ECE | < 0.05 | Menor es mejor |
| Coverage | > 95% | Mayor es mejor |

## Subcomandos

| Comando | Descripción | Iteraciones default |
|---|---|---|
| `/autoresearch:features` | Probar agregar/eliminar/transformar features | 20 |
| `/autoresearch:hyperparams` | Buscar mejores hiperparámetros de XGBoost | 25 |
| `/autoresearch:ensemble` | Optimizar pesos del ensemble de predictores | 20 |
| `/autoresearch:llm-prompt` | Iterar prompts de análisis/calibración con LLM | 15 |
| `/autoresearch:backtest` | Validar un cambio con walk-forward temporal | 10 |
| `/autoresearch:worldcup` | Backtest sobre Mundiales 2010, 2014, 2018, 2022 | 4 |
| `/autoresearch:scenario` | Generar casos borde por equipo, jugador, fixture | 15 |
| `/autoresearch:debug` | Cazar bugs en predictores o ETL | 15 |
| `/autoresearch:evals` | Analizar resultados de iteraciones previas | N/A |

## Protocolo del loop

```
1. BASELINE
   - Correr backtest walk-forward con la versión actual.
   - Registrar métricas objetivo en .agents/logs/autoresearch/{id}/baseline.json

2. HIPÓTESIS
   - Proponer UN cambio (feature, hiperparámetro, peso, prompt).
   - Justificar por qué podría mejorar la métrica objetivo.

3. EXPERIMENTO
   - Aplicar el cambio en una rama/copy aislada.
   - Re-entrenar / re-predecir con seed fijo.
   - Correr backtest walk-forward con los mismos folds temporales.

4. VERIFICACIÓN
   - Comparar métricas vs baseline.
   - Si mejora en ≥1 métrica objetivo sin empeorar críticamente otra → KEEP.
   - Si empeora o no cambia → DISCARD.

5. REGISTRO
   - Guardar resultado en .agents/logs/autoresearch/{id}/experiments.tsv
   - Incluir: timestamp, hipótesis, cambio, métricas delta, veredicto.

6. HANDOFF
   - Si KEEP y es cambio de código: pasar a reviewer + tester.
   - Si KEEP y es ajuste de configuración: documentar en Obsidian.
   - Si DISCARD: continuar siguiente iteración.
```

## Formato de experimento

```tsv
experiment_id	timestamp	hypothesis	change_summary	log_loss	brier	rps	accuracy	roi	ece	coverage	verdict	note
EXP-001	2026-06-13T20:00:00Z	Agregar h2h_goals_avg	+h2h_goals_avg feature	0.612	0.183	0.171	0.47	0.08	0.042	0.97	KEEP	Mejora log-loss y accuracy
EXP-002	2026-06-13T20:05:00Z	Subir max_depth a 8	max_depth=8	0.658	0.195	0.179	0.44	0.02	0.051	0.97	DISCARD	Overfitting en fold 3
```

## Flags universales

| Flag | Uso |
|---|---|
| `Iteraciones: N` | Número máximo de iteraciones |
| `Iteraciones: ilimitadas` | Sin límite (requiere aprobación explícita) |
| `--metrica log-loss` | Optimizar solo esta métrica |
| `--folds 5` | Número de folds temporales en walk-forward |
| `--seed 2026` | Seed reproducible |
| `--chain tester` | Pasar a tester tras KEEP |

## Anti-patterns
- Cambiar más de una cosa por experimento.
- Entrenar con datos del futuro.
- Ignorar métricas de calibration por mejorar accuracy.
- Dejar experimentos sin registro.
- Aplicar un cambio que solo mejora en un fold pero empeora en los demás.
