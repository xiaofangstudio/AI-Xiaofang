@echo off
cd /d "%~dp0"
python "clean_cache.py"
if errorlevel 1 (
  echo.
  echo [Tip] python not found. Please run "启动小方.bat" first to install Python.
)
echo.
pause
