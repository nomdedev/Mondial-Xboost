#Requires -Version 5.1
<#
.SYNOPSIS
    Helper para ejecutar scripts de Mondial-Xboost activando el venv automáticamente.
.EXAMPLE
    .\run.ps1 scripts/train.py --loop --trials 50
    .\run.ps1 scripts/predict.py --home Brazil --away Morocco
    .\run.ps1 scripts/training_dashboard.py
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Script,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$venvPath = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
} else {
    Write-Host "No se encontró el entorno virtual en $venvPath" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = $PSScriptRoot
$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
& $python $Script @Arguments
