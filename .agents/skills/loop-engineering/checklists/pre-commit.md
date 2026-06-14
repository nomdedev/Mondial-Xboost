# Pre-commit Checklist — Mondial-Xboost

Antes de cada commit, verificar:

- [ ] `dotnet build Mondial-Xboost.sln` pasa (entorno con .NET 9).
- [ ] `dotnet test MondialXboost.Web.Tests` pasa.
- [ ] `pytest tests/ -q` pasa.
- [ ] `python scripts/verify-gates.py --skip dotnet` pasa en Python.
- [ ] No archivos `.env`, `*.pkl`, `*.parquet` grandes o modelos entrenados en el diff (salvo `data/models/` si es intencional).
- [ ] No secrets en el diff (`python scripts/verify-gates.py --gate secret_scan`).
- [ ] Mensaje de commit describe qué fase/gate afecta.

Comando rápido:
```bash
python scripts/verify-gates.py --skip dotnet
```
