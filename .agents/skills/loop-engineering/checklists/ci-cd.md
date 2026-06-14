# CI/CD Checklist — Mondial-Xboost

Pipeline definida en `.github/workflows/ci.yml`.

## Jobs obligatorios

- [ ] Setup .NET 9 SDK.
- [ ] Setup Python 3.11.
- [ ] `dotnet restore`.
- [ ] `dotnet build Mondial-Xboost.sln`.
- [ ] `dotnet test MondialXboost.Web.Tests`.
- [ ] `pip install -r requirements.txt`.
- [ ] `pytest tests/ -q`.
- [ ] `python scripts/verify_gates.py`.
- [ ] `dotnet list package --vulnerable`.

## Jobs opcionales / programados

- [ ] `pip-audit` o `safety check` (una vez por semana).
- [ ] `python scripts/run_backtest_gate.py` en PRs que toquen ML.
- [ ] `python scripts/run_data_council.py` en PRs que toquen datos.

## Caché

- `~/.nuget/packages`
- `~/.cache/pip`
- `venv` (preferiblemente con `requirements-lock.txt`)

## Merge policy

No mergear si:
- Falla un gate requerido.
- `verify_gates.py` devuelve exit code distinto de 0.
- Hay vulnerabilidades de dependencias sin mitigar.
