# CI/CD Checklist — Mondial-Xboost

Pipeline definida en `.github/workflows/ci.yml`.

## Jobs obligatorios

- [ ] Setup .NET 9 SDK.
- [ ] Setup Python 3.11.
- [ ] `dotnet restore`.
- [ ] `dotnet build Oloraculo.sln`.
- [ ] `dotnet test Oloraculo.Web.Tests`.
- [ ] `pip install -r requirements.txt`.
- [ ] `pytest tests/ -q`.
- [ ] `python scripts/verify-gates.py`.
- [ ] `dotnet list package --vulnerable`.

## Jobs opcionales / programados

- [ ] `pip-audit` o `safety check` (una vez por semana).
- [ ] `python scripts/run-backtest-gate.py` en PRs que toquen ML.
- [ ] `python scripts/run-data-council.py` en PRs que toquen datos.

## Caché

- `~/.nuget/packages`
- `~/.cache/pip`
- `venv` (preferiblemente con `requirements-lock.txt`)

## Merge policy

No mergear si:
- Falla un gate requerido.
- `verify-gates.py` devuelve exit code distinto de 0.
- Hay vulnerabilidades de dependencias sin mitigar.
