@echo off
rem ============================================================
rem   Xiaofang corpus trainer launcher (script B)
rem   Runs 学习语料.py IN THE FOREGROUND: this console window
rem   stays open and shows live progress (bar / percent / ETA).
rem   ASCII-safe (no non-ASCII bytes) + chcp 65001 + goto-flow.
rem ============================================================
chcp 65001 >nul
title Xiaofang corpus trainer (foreground)
setlocal

echo.
echo  ================================================
echo    Xiaofang - corpus trainer (foreground)
echo    This window IS the training console.
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

rem ---------- tier menu ----------
echo   Which tier should learn the corpus?
echo     1 = Lite   (~2.40B, fastest  - default)
echo     2 = Pro    (~2.48B, deeper)
echo     3 = Ultra  (~2.50B, deepest)
set "XF_TIER=lite"
set "TIER_PICK="
set /p "TIER_PICK=  Type 1 / 2 / 3 then press Enter (Enter = Lite): "
if "%TIER_PICK%"=="2" set "XF_TIER=pro"
if "%TIER_PICK%"=="3" set "XF_TIER=ultra"
echo  [OK] Tier: %XF_TIER%
echo.
echo   How long should it run?
echo     Enter  = until it finishes (max 2 passes)
echo     A number = hours, e.g. 3  -> stop after 3 hours
set "HRS="
set /p "HRS=  Hours (Enter = unlimited): "
set "TIMEFLAG="
if defined HRS set "TIMEFLAG=--hours %HRS%"
echo.

rem ---------- run (foreground, same console) ----------
echo  [OK] Launching corpus trainer ...
echo.
cd /d "%~dp0"
%PY% "学习语料.py" --tier %XF_TIER% %TIMEFLAG%
echo.
echo  [i] Trainer exited. Press any key to close.
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
