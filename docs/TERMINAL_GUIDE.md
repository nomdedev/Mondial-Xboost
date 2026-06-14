# Guía de Terminal — Mondial-Xboost

Abrir PowerShell o CMD en la raíz del repo `D:\martin\Proyectos\Mondial-xBoost`.

## 1. Activar entorno Python

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# CMD
venv\Scripts\activate.bat
```

## 2. Variables de entorno útiles

```powershell
$env:PYTHONPATH = "$PWD"
```

## 3. Entrenar modelo

### Modelo canónico (rápido)

```powershell
python scripts/train.py
```

Con nombre personalizado:

```powershell
python scripts/train.py --name mundial2026
```

### Loop Engineering — buscar mejores hiperparámetros

```powershell
python scripts/train.py --loop --trials 50
```

10 batches seguidos:

```powershell
python scripts/train.py --loop --auto
```

Salida: modelos en `data/models/` y resultados en `data/models/loop_engineering.json`.

## 4. Ver todos los entrenamientos

```powershell
python scripts/training_dashboard.py
```

Ver top 10 configs:

```powershell
python scripts/training_dashboard.py --top 10
```

Ver evolución por batch:

```powershell
python scripts/training_dashboard.py --evolution
```

## 5. Hacer predicciones

### Un solo partido

```powershell
python scripts/predict.py --home Brazil --away Morocco --date 2026-06-20
```

### Varios partidos desde JSON

```powershell
python scripts/predict.py --fixtures fixtures.json
```

Ejemplo de `fixtures.json`:

```json
[
  {"date": "2026-06-20", "home_team": "Brazil", "away_team": "Morocco", "neutral": true},
  {"date": "2026-06-21", "home_team": "Argentina", "away_team": "France", "neutral": true}
]
```

### Predecir los últimos N partidos del histórico

```powershell
python scripts/predict.py --last-n 5
```

## 6. Iniciar el bridge ML (FastAPI)

```powershell
python -m predictors.api
```

O usar los helpers:

```powershell
.\start-ml-bridge.ps1     # PowerShell
start-ml-bridge.bat        # CMD
```

Verificar que está vivo:

```powershell
curl http://127.0.0.1:8000/health
```

## 7. Correr tests y gates

### Resumen rápido

```powershell
python scripts/run_tests.py
```

### Solo tests rápidos (sin backtest ni bridge)

```powershell
python scripts/run_tests.py --fast
```

### Tests + verify_gates completos

```powershell
python scripts/run_tests.py --gate
```

### Comandos individuales

```powershell
python -m pytest tests/ -v
python scripts/run_data_council.py
python scripts/run_backtest_gate.py
python scripts/run_bridge_smoke_test.py
python scripts/verify_gates.py
```

## 8. Feature engineering

```powershell
python -m predictors.feature_engineering
```

Salida: `data/features/train_historical.parquet`.

## 9. Backtest sobre Mundiales anteriores

```powershell
python -m backtest.world_cup_backtest
```

Salida: `backtest/results/world_cup_backtest_summary.json`.

## 10. App .NET (cuando dotnet esté disponible)

```powershell
dotnet restore
dotnet build MondialXboost.Web
dotnet test
dotnet run --project MondialXboost.Web
```

## 11. Makefile / helpers

Si tenés `make` instalado:

```bash
make train
make predict ARGS="--home Brazil --away Morocco"
make dashboard
make test
```

En Windows también podés usar `run.ps1`:

```powershell
.\run.ps1 scripts/train.py --loop --trials 20
```
