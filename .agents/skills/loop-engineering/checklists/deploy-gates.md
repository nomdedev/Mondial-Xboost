# Deploy Gates — Mondial-Xboost

Checklist antes de desplegar a producción/staging.

- [ ] Build + tests verdes (`dotnet build`, `dotnet test`, `pytest`).
- [ ] `python scripts/verify_gates.py` PASS (o SKIP justificado).
- [ ] `python scripts/run_data_council.py` PASS.
- [ ] `python scripts/run_backtest_gate.py` PASS.
- [ ] `python scripts/run_bridge_smoke_test.py` PASS.
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
