@echo off
rem ============================================================
rem   Xiaofang FlphaLit 1.3 Alpha launcher (archived version)
rem   ASCII-safe (no non-ASCII bytes) + chcp 65001 + goto-flow
rem   so cmd never flash-closes. Run this .bat in this folder.
rem ============================================================
chcp 65001 >nul
title Xiaofang FlphaLit 1.3 Alpha launcher
setlocal

echo.
echo  ================================================
echo    Xiaofang FlphaLit 1.3 Alpha - env check and launch
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
%PY% -c "import numpy, colorama, pyfiglet, ddgs, psutil" >nul 2>nul
if not errorlevel 1 goto DEPSOK

echo  [..] Missing deps. Installing via pip ...
%PY% -m pip install --quiet --upgrade pip
%PY% -m pip install --quiet numpy colorama pyfiglet ddgs psutil
%PY% -c "import numpy, colorama, pyfiglet, ddgs, psutil" >nul 2>nul
if errorlevel 1 goto DEPFAIL

:DEPSOK
echo  [OK] Deps ready.
echo.
rem ---------- run ----------
echo  [OK] Launching Xiaofang FlphaLit 1.3 Alpha ...
echo.
cd /d "%~dp0"
%PY% "xiaofang_v13alpha.py"
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
echo      pip install numpy colorama pyfiglet ddgs psutil
pause
exit /b 1