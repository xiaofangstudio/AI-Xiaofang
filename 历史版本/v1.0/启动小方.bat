@echo off
rem ============================================================
rem   Xiaofang FlphaLit 1.0 launcher (archived v1.0)
rem   ASCII-safe. goto-flow so it never fails at parse time.
rem ============================================================
chcp 65001 >nul
title Xiaofang FlphaLit 1.0 launcher
setlocal

echo.
echo  ================================================
echo    Xiaofang FlphaLit 1.0 - env check and launch
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

echo  [..] Checking optional CuPy GPU acceleration ...
%PY% -c "import cupy" >nul 2>nul
if not errorlevel 1 goto RUNX
echo  [..] Trying to install CuPy, optional -- CPU still works if it fails ...
%PY% -m pip install --quiet --prefer-binary cupy
echo.

:RUNX
echo  [OK] Everything ready. Launching Xiaofang FlphaLit 1.0 ...
echo.
cd /d "%~dp0"
%PY% "xiaofang_v10.py"
echo.
echo  [i] Xiaofang exited. Press any key to close.
pause
endlocal
goto :EOF

:NOPY
echo  [!!] Python not found. Trying to auto-install Python 3.12 ...
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe' -OutFile '%TEMP%\py_installer.exe'"
if not exist "%TEMP%\py_installer.exe" (
    echo  [!!] Download failed. Please install Python 3.12 from https://www.python.org/downloads/ and retry.
    pause
    exit /b 1
)
echo  [..] Installing Python silently (adds to PATH) ...
"%TEMP%\py_installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts"
set "PY=python"
goto :DEPSOK

:DEPFAIL
echo  [!!] Dependency install failed.
echo      Please run manually then relaunch:
echo      pip install numpy colorama pyfiglet ddgs psutil
pause
exit /b 1