@echo off
rem ============================================================
rem   Xiaofang FlphaLit 1.5 Alpha (archived) launcher
rem   ASCII-safe (no non-ASCII bytes) so cmd always parses it.
rem   Uses goto-flow so it NEVER fails at parse time.
rem ============================================================
chcp 65001 >nul
title Xiaofang FlphaLit 1.5 Alpha (archived) launcher
setlocal

echo.
echo  ================================================
echo    Xiaofang FlphaLit 1.5 Alpha (archived) - launch
echo  ================================================
echo.

rem ---------- 1. detect Python ----------
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

rem ---------- 2. check / install deps ----------
%PY% -c "import numpy, colorama, pyfiglet, ddgs, psutil, requests, bs4" >nul 2>nul
if not errorlevel 1 goto DEPSOK

echo  [..] Missing deps. Installing via pip ...
%PY% -m pip install --quiet --upgrade pip
%PY% -m pip install --quiet numpy colorama pyfiglet ddgs psutil requests beautifulsoup4
%PY% -c "import numpy, colorama, pyfiglet, ddgs, psutil, requests, bs4" >nul 2>nul
if errorlevel 1 goto DEPFAIL

:DEPSOK
echo  [OK] Deps ready.
echo.

rem ---------- 3. optional GPU (CuPy) ----------
echo  [..] Checking optional CuPy GPU acceleration ...
%PY% -c "import cupy" >nul 2>nul
if not errorlevel 1 goto RUNX
echo  [..] Trying to install CuPy, optional -- CPU still works if it fails ...
%PY% -m pip install --quiet --prefer-binary cupy
echo.

:RUNX
rem ---------- 4. run Xiaofang ----------
echo  [OK] Everything ready. Launching Xiaofang FlphaLit 1.5 Alpha ...
echo.
cd /d "%~dp0"
%PY% "xiaofang_v15alpha.py"
echo.
echo  [i] Xiaofang exited. Press any key to close.
pause
endlocal
goto :EOF

:NOPY
echo  [!!] Python not found. Please install Python 3.10+ from:
echo       https://www.python.org/downloads/  then retry.
pause
exit /b 1

:DEPFAIL
echo  [!!] Dependency install failed.
echo      Please run manually then relaunch:
echo      pip install numpy colorama pyfiglet ddgs psutil requests beautifulsoup4
pause
exit /b 1