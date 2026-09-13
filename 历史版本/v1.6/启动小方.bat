@echo off
rem ============================================================
rem   Xiaofang FlphaLit 1.6 (Official) launcher
rem   ASCII-safe (no non-ASCII bytes) + chcp 65001 + goto-flow
rem   so cmd never flash-closes. Run this .bat to start Xiaofang.
rem ============================================================
chcp 65001 >nul
title Xiaofang FlphaLit 1.6 Official launcher
setlocal

echo.
echo  ================================================
echo    Xiaofang FlphaLit 1.6 - env check and launch
echo  ================================================
echo.

set "PY="
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
    where py >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY goto NOPY

echo  [OK] Python ready: %PY%
%PY% --version 2>nul
echo.

rem ---------- deps ----------
%PY% -c "import numpy, colorama, pyfiglet, ddgs, psutil, requests, bs4" >nul 2>nul
if not errorlevel 1 goto DEPSOK

echo  [..] Some deps missing. Installing via pip ...
%PY% -m pip install --quiet --upgrade pip
%PY% -m pip install --quiet numpy colorama pyfiglet ddgs psutil requests beautifulsoup4
%PY% -c "import numpy, colorama, pyfiglet, ddgs, psutil, requests, bs4" >nul 2>nul
if errorlevel 1 goto DEPFAIL

:DEPSOK
echo  [OK] Deps ready.
echo.
rem ---------- run ----------
echo  [OK] Launching Xiaofang ...
echo.
cd /d "%~dp0"
%PY% "xiaofang_v16.py"
echo.
echo  [i] Xiaofang exited. Press any key to close.
pause
endlocal
goto :EOF

:NOPY
echo  [!!] Python not found. Install Python 3.10+ from:
echo       https://www.python.org/downloads/  then retry.
pause
exit /b 1

:DEPFAIL
echo  [!!] Dependency install failed. Install manually then relaunch:
echo      pip install numpy colorama pyfiglet ddgs psutil requests beautifulsoup4
pause
exit /b 1
