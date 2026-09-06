@echo off
rem ============================================================
rem   Xiaofang historical-version launcher
rem   ASCII-safe. A local sitecustomize.py softens the cupy warning.
rem ============================================================
chcp 65001 >nul
title Xiaofang V0.5Alpha launcher
setlocal

echo.
echo  ============================================
echo    Xiaofang V0.5Alpha launcher
echo  ============================================
echo.

set "PY="
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
    where py >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    echo  [!!] Python not found. Please install Python 3.8 or newer first.
    pause
    exit /b 1
)
echo  [OK] Python: %PY%
%PY% --version 2>nul
echo.
echo  Loading Xiaofang V0.5Alpha, building model - please wait (~15-40 sec) ...
echo.
cd /d "%~dp0"
%PY% "xiaofang_v05alpha.py"
echo.
echo  [Program exited]
pause
endlocal