# Pipeline Activo

## Estado actual
- Plan aprobado: Full Architecture
- Fase 2 completada: feature engineering + XGBoost funcionando
- Fase 3 en progreso: validación de datos, comparación Elo, council de agentes

## Features completadas
- `OLOR-001` — XGBoost predictor + bridge C# ↔ Python (calibración isotónica por defecto)
- `OLOR-002` — News scraper de lesiones/disponibilidad (en diseño)
- `OLOR-003` — Council de agentes expertos de datos
- `OLOR-004` — Comparación de Elo vs World Football Elo Ratings
- `OLOR-005` — Visualización HTML del flujo de datos

## Documentación reciente
- `docs/vault/03-Architecture/elo-algorithm.md`
- `docs/vault/03-Architecture/player-weighting.md`
- `docs/vault/03-Architecture/data-pipeline-flow.md`
- `docs/flows/oloraculo-data-flow.html`

## Agentes nuevos
- `.agents/agents/data-council.md`
- `.agents/agents/elo-auditor.md`
- `.agents/agents/player-data-auditor.md`
- `.agents/agents/drift-detector.md`

## Scripts y reportes
- `scripts/compare_elo_worldfootball.py` → `backtest/results/elo_comparison.json`
- `scripts/autoresearch_worldcup.py` → `.agents/logs/autoresearch/`
- `scripts/verify-gates.py` → `.agents/logs/pipeline-state.json`
- `scripts/run-data-council.py` → `.agents/logs/data-council-report.json`
- `scripts/run-backtest-gate.py` → `backtest/results/world_cup_backtest_summary.json`
- `scripts/run-bridge-smoke-test.py` → `backtest/results/bridge_smoke.json`
- `backtest/world_cup_backtest.py` → backtest temporal

## Loop Engineering v2.0
- Skill: `.agents/skills/loop-engineering/SKILL.md`
- Config: `.agents/skills/loop-engineering/gates.json`
- Checklists: `.agents/skills/loop-engineering/checklists/`

## Agentes del loop
- `.agents/agents/data-council.md`
- `.agents/agents/elo-auditor.md`
- `.agents/agents/player-data-auditor.md`
- `.agents/agents/drift-detector.md`
- `.agents/agents/data-leakage-auditor.md` *(nuevo)*
- `.agents/agents/ml-model-gatekeeper.md` *(nuevo)*
- `.agents/agents/bridge-integrator.md` *(nuevo)*
- `.agents/agents/data-freshness-monitor.md` *(nuevo)*
- `.agents/agents/production-sentry.md` *(nuevo)*

## Próximos pasos
- Integrar PlayerStatsService en feature engineering.
- LLM analysis service.
- Workflow CI/CD en GitHub Actions.
