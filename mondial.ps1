# Mondial-Xboost convenience wrapper (PowerShell)
$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

# 1. Prefer an active virtual environment
$PYTHON = $null
if ($env:VIRTUAL_ENV) {
    $candidate = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    if (Test-Path $candidate) { $PYTHON = $candidate }
    else {
        $candidate = Join-Path $env:VIRTUAL_ENV "bin\python"
        if (Test-Path $candidate) { $PYTHON = $candidate }
    }
}

# 2. Otherwise look for a project-local venv
if (-not $PYTHON) {
    $candidates = @(
        Join-Path $ROOT "venv\Scripts\python.exe"
        Join-Path $ROOT ".venv\Scripts\python.exe"
        Join-Path $ROOT "venv\bin\python"
        Join-Path $ROOT ".venv\bin\python"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { $PYTHON = $candidate; break }
    }
}

# 3. Fall back to the system interpreter
if (-not $PYTHON) { $PYTHON = "python" }

# Verify we can import the CLI module (dependencies installed)
& $PYTHON -c "import scripts.mondial_cli" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: no se pudo importar 'scripts.mondial_cli' con $PYTHON." -ForegroundColor Red
    Write-Host "Asegurate de tener el entorno virtual activado o instalar las dependencias:" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    Write-Host "  pip install -e ." -ForegroundColor Yellow
    exit 1
}

& $PYTHON (Join-Path $ROOT "scripts\mondial_cli.py") @args
exit $LASTEXITCODE
