@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%ROOT%scripts\mondial_cli.py" %*
