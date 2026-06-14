---
name: loop-engineering
version: 2.0.0
description: >
  Sistema de calidad ejecutable para el pipeline de Mondial-Xboost. Define gates
  objetivos por fase, los scripts que los verifican y el registro de estado. Cada
  feature solo avanza si los gates requeridos dan PASS.
---

# Loop Engineering v2.0 — Gates y Verificación Automática

## Propósito

Convertir el pipeline de agentes en un **proceso medible y reproducible**. Este
skill especifica:

1. Qué verificar en cada fase.
2. Cómo verificarlo (comando concreto).
3. Dónde queda la evidencia.
4. Qué hacer si falla.

La fuente de verdad machine-readable está en
`.agents/skills/loop-engineering/gates.json`.

## Contratos canónicos

### Modelo canónico
- Motor: `predictors/xgboost_engine.py` (`XGBoostFootballPredictor`)
- API: `predictors/api.py` (FastAPI, puerto 8000)
- Modelos persistidos en: `data/models/xgboost_football_*.pkl`
- Features: `predictors.feature_engineering.FEATURE_COLS` (única fuente de verdad)

### Bridge C# ↔ Python
- Endpoint: `POST /predict`
- Request: `{"fixtures": [{"date", "home_team", "away_team", "neutral"}]}`
- Response: `{"predictions": [{"home_team", "away_team", "prob_away_win", "prob_draw", "prob_home_win", "expected_home_goals", "expected_away_goals", "top_pick"}]}`

No debe existir otro motor XGBoost productivo. Los archivos legacy viven en
`scripts/archive/`.

## Gates por fase

Los comandos asumen que el directorio de trabajo es la raíz del repo.

### Fase 1 — Domain
| Gate | Comando / evidencia | Criterio |
|------|---------------------|----------|
| Glosario | `docs/vault/02-Domain/glossary.md` | Términos usados en código/docs están definidos |
| Fuentes | Revisión manual + `scripts/run_data_council.py` | Fuentes cubren features requeridas |
| Métricas | `docs/vault/01-META/project-identity.md` | Métricas son medibles con datos disponibles |

### Fase 2 — Architecture
| Gate | Comando / evidencia | Criterio |
|------|---------------------|----------|
| Contratos | `docs/vault/03-Architecture/xgboost-bridge-contract.md` | C#↔Python documentado y alineado |
| No-leakage | `scripts/audit_leakage.py` | Elo/rolling/H2H no usan datos futuros |
| Extensibilidad | Revisión de `IPredictor` | Nuevos predictores no rompen la interfaz |

### Fase 3 — Backend / Data (Data Council)
| Gate | Comando | Evidencia | Criterio |
|------|---------|-----------|----------|
| Feature schema | `python -c "from predictors.feature_engineering import FEATURE_COLS; ..."` | `data/features/train_historical.parquet` | Columnas == FEATURE_COLS |
| Elo calibration | `python scripts/compare_elo_worldfootball.py` | `backtest/results/elo_comparison.json` | top100.mean_abs_diff < 100 pts, corr > 0.90 |
| No leakage | `python scripts/audit_leakage.py` | `backtest/results/audit_leakage.json` | Sin hallazgos FAIL |
| Player weights | `python scripts/run_data_council.py` | `.agents/logs/data-council-report.json` | Pesos suman 1.0 |

### Fase 4 — Security
| Gate | Comando | Criterio |
|------|---------|----------|
| Secret scan | `python scripts/verify_gates.py --gate secret_scan` | Sin secretos en código/logs |
| SQL injection | Revisión manual / `grep -R "FromSqlRaw"` | EF Core parametrizado |
| XSS | Revisión de Blazor | Sin raw HTML sin sanitizar |
| Rate limits | Revisión de `ApiFootballService` | Rate limits configurados |

### Fase 5 — Develop
| Gate | Comando | Criterio |
|------|---------|----------|
| Build | `dotnet build Mondial-Xboost.sln` | Exit code 0 |
| Format | `dotnet format Mondial-Xboost.sln --verify-no-changes` | Sin cambios de formato |
| Python lint | `python -m ruff check predictors scripts backtest tests` | Sin errores |
| Code review | `reviewer` agent | Veredicto APPROVED / MINOR |

### Fase 6 — QA
| Gate | Comando | Evidencia | Criterio |
|------|---------|-----------|----------|
| .NET tests | `dotnet test MondialXboost.Web.Tests` | test output | ≥90% passing |
| Python tests | `pytest tests/ -q` | pytest output | 100% passing |
| Backtest baseline | `python scripts/run_backtest_gate.py` | `backtest/results/world_cup_backtest_summary.json` | log_loss < 1.05, brier < 0.22, acc > 45% |
| Bridge smoke | `python scripts/run_bridge_smoke_test.py` | `backtest/results/bridge_smoke.json` | Probs suman ~1, picks válidos |

### Fase 7 — Production
| Gate | Comando / evidencia | Criterio |
|------|---------------------|----------|
| Health check | `curl http://127.0.0.1:8000/health` | status 200 |
| Daily predictions | Log de predicciones | Probs suman ~1, picks coherentes |
| Cost control | Revisión de uso OpenRouter/API-Football | Dentro de presupuesto |
| Metrics published | `README.md` / JSON actualizado | Métricas públicas y reproducibles |

## Scripts del loop

- `scripts/verify_gates.py` — orquestador de todos los gates.
- `scripts/run_data_council.py` — validaciones de calidad de datos.
- `scripts/run_backtest_gate.py` — backtest con thresholds.
- `scripts/run_bridge_smoke_test.py` — prueba de humo del bridge.

## Acciones ante fallo

| Fase | Fallo | Acción |
|------|-------|--------|
| 1-2 | Gate rechazado | Rebotar a fase anterior, documentar motivo |
| 3 | Data Council BLOCK | Rebotar a Fase 2 (revisar contrato o fuente de datos) |
| 4 | Security FAIL | Rebotar a Fase 3 o Fase 5 según origen |
| 5 | Build/lint FAIL | Corregir y re-ejecutar `verify_gates.py --phase 5` |
| 6 | Test/backtest FAIL | Rebotar a Fase 5 con plan de fix detallado |
| 7 | Incidente producción | Rollback o hotfix; documentar en `.agents/logs/incidents.md` |

## Registro de estado

Cada ejecución de `verify_gates.py` actualiza `.agents/logs/pipeline-state.json`:

```json
{
  "featureId": "LOOP-ENGINEERING",
  "phase": 6,
  "gate": "backtest_baseline",
  "gateId": "backtest_baseline",
  "status": "PASS",
  "verifiedBy": "verify_gates.py",
  "timestamp": "2026-06-13T20:00:00Z",
  "evidence": "backtest/results/world_cup_backtest_summary.json",
  "output": "..."
}
```

## Invocación

```bash
# Todos los gates
python scripts/verify_gates.py

# Solo fase 6
python scripts/verify_gates.py --phase 6

# Solo un gate
python scripts/verify_gates.py --gate elo_calibration

# Saltar gates que dependen de dotnet
python scripts/verify_gates.py --skip dotnet

# Salida JSON
python scripts/verify_gates.py --json
```

## Notas

- Si `dotnet` no está disponible, los gates .NET se marcan `SKIP` con un mensaje
  claro; deben ejecutarse en un entorno con .NET 9 SDK antes de mergear.
- Los modelos legacy (`scripts/archive/xgboost_predictor_legacy.py`,
  `scripts/archive/api_main_legacy.py`) no deben usarse en producción.
