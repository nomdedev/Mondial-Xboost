# Bridge Integration Gates — Mondial-Xboost

Ejecutar con `python scripts/run_bridge_smoke_test.py`.

## Gates

- [ ] **FastAPI saludable**: `GET /health` responde 200.
- [ ] **Predict devuelve batch**: `POST /predict` con múltiples fixtures.
- [ ] **Suma de probabilidades**: `abs(sum(probs) - 1.0) < 0.01` para cada fixture.
- [ ] **Top pick válido**: `top_pick` en `{Home, Draw, Away}`.
- [ ] **Expected goals no negativos**: `expected_home_goals >= 0`, `expected_away_goals >= 0`.
- [ ] **C# bridge no bloquea**: `XGBoostBridgeService.PredictAsync` es async y usa timeout.
- [ ] **Graceful degradation**: si el bridge cae, el predictor ladder continúa sin XGBoost.

## Contrato

Ver `docs/vault/03-Architecture/xgboost-bridge-contract.md`.

## Comandos

```bash
python scripts/run_bridge_smoke_test.py
dotnet test MondialXboost.Web.Tests --filter "FullyQualifiedName~XGBoostBridge"
```
