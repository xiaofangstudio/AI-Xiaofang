@echo off
rem ============================================================
rem   Xiaofang FlphaLit 0.9 Alpha launcher
rem   ASCII-safe. GPU-first boot; friendly Chinese notes in-engine.
rem ============================================================
chcp 65001 >nul
title FlphaLit 0.9 Alpha launcher
setlocal
echo.
echo  ============================================
echo    FlphaLit 0.9 Alpha launcher
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
    echo  [!!] Python not found. Install Python 3.8+ first.
    pause
    exit /b 1
)
echo  [OK] Python: %PY%
echo.
echo  Loading FlphaLit 0.9 Alpha - please wait...
echo.
cd /d "%~dp0"
%PY% "xiaofang_v09alpha.py"
echo.
pause
endlocal