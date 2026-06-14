@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

rem 1. Prefer an active virtual environment
if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        set "PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
    ) else if exist "%VIRTUAL_ENV%\bin\python.exe" (
        set "PYTHON=%VIRTUAL_ENV%\bin\python.exe"
    )
)

rem 2. Otherwise look for a project-local venv
if not defined PYTHON (
    if exist "%ROOT%\venv\Scripts\python.exe" (
        set "PYTHON=%ROOT%\venv\Scripts\python.exe"
    ) else if exist "%ROOT%\.venv\Scripts\python.exe" (
        set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
    )
)

rem 3. Fall back to the system interpreter
if not defined PYTHON set "PYTHON=python"

rem Verify we can import the CLI module (dependencies installed)
"%PYTHON%" -c "import scripts.mondial_cli" >nul 2>&1
if errorlevel 1 (
    echo Error: no se pudo importar 'scripts.mondial_cli' con %PYTHON%.
    echo Asegurate de tener el entorno virtual activado o instalar las dependencias:
    echo   pip install -r requirements.txt
    echo   pip install -e .
    exit /b 1
)

"%PYTHON%" "%ROOT%\scripts\mondial_cli.py" %*
