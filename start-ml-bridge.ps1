# Start the Python ML bridge for Mondial-Xboost
$venvPath = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
& $venvPath
$env:PYTHONPATH = $PSScriptRoot
python -m predictors.api
