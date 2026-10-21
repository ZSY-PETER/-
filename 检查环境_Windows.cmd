@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8:backslashreplace"
python -X utf8 check_windows_model.py
echo.
pause
endlocal
