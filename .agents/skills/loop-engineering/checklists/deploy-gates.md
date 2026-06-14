# Deploy Gates — Mondial-Xboost

Checklist antes de desplegar a producción/staging.

- [ ] Build + tests verdes (`dotnet build`, `dotnet test`, `pytest`).
- [ ] `python scripts/verify-gates.py` PASS (o SKIP justificado).
- [ ] `python scripts/run-data-council.py` PASS.
- [ ] `python scripts/run-backtest-gate.py` PASS.
- [ ] `python scripts/run-bridge-smoke-test.py` PASS.
- [ ] `appsettings.json` y `appsettings.Production.json` sin secrets.
- [ ] `.env` en `.gitignore` y no versionado.
- [ ] Python venv con `requirements.txt` instalado en target.
- [ ] Bridge ML levantado y saludable (`/health`).
- [ ] Base de datos SQLite migrada y con backup.
- [ ] Modelo entrenado copiado a `data/models/` del target.
- [ ] Plan de rollback documentado (commit anterior, snapshot de DB/modelo).
- [ ] Variables de entorno configuradas fuera del repo.

## Post-deploy

- [ ] Verificar health checks.
- [ ] Revisar logs de error las primeras 2 horas.
- [ ] Confirmar predicciones coherentes en producción.
