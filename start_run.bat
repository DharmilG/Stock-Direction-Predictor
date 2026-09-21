@echo off
setlocal
cd /d "%~dp0"
python start_run.py
if errorlevel 1 (
  echo.
  echo Project failed. See logs in the logs folder.
  exit /b %errorlevel%
)
endlocal
