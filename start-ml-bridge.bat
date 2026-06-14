@echo off
call "%~dp0venv\Scripts\activate.bat"
set PYTHONPATH=%~dp0
python -m predictors.api
