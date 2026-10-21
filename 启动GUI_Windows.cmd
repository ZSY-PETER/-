@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8:backslashreplace"
set "PYTHONUNBUFFERED=1"
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python command not found. Activate the correct Python environment first.
  pause
  exit /b 1
)
python -X utf8 bridge_gui.py
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo [ERROR] GUI startup failed. Check Python environment and logs above.
  pause
)
endlocal & exit /b %EXIT_CODE%
