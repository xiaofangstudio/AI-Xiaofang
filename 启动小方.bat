@echo off
rem ============================================================
rem  AI Xiaofang - launcher shim   (kept pure ASCII on purpose)
rem ------------------------------------------------------------
rem  Why ASCII inside: cmd.exe decodes a .bat using whatever code
rem  page is active while it reads the file, so Chinese characters
rem  written in a .bat often come out as garbage and break the
rem  "if exist" checks. Every Chinese message the user reads lives
rem  in Python files instead, which are always decoded as UTF-8.
rem
rem  What a double-click does
rem    first run   : detect the machine - CPU architecture and
rem                  Windows version - then look for Python. If it
rem                  is missing, download and silently install one,
rem                  then let pip pull every library the engine
rem                  needs. Only after all that does the 1 / 2 / 3
rem                  version menu show up.
rem    later runs  : the setup already passed, so it jumps straight
rem                  to the 1 / 2 / 3 menu.
rem
rem  Usage
rem    double-click this file            -> shows the version menu
rem    this-file.bat 2                   -> starts Pro directly
rem    this-file.bat --check             -> health check only
rem    this-file.bat --reinstall-env     -> forget the environment
rem                                         marker and set it up again
rem ============================================================
chcp 65001 >nul
title AI Xiaofang - launcher
setlocal EnableExtensions
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0"
if not exist "xiaofang_launcher.py" goto NOLAUNCH

set "MARKER=.xf_env_ok"
set "PF86=%ProgramFiles(x86)%"

rem ---- 1) machine check: architecture + Windows version --------------
call :DETECT_MACHINE

rem ---- 2) was the environment already finished on an earlier run? -----
if /i not "%~1"=="--reinstall-env" goto CHECK_MARKER
if exist "%MARKER%" del /q "%MARKER%" >nul 2>nul
echo    [env] re-running the environment setup because you asked for it.
:CHECK_MARKER
set "PYEXE="
set "PYVER="
set "PYCACHED="
if exist "%MARKER%" set /p PYCACHED=<"%MARKER%"
if not defined PYCACHED goto FIRST_RUN
if not exist "%PYCACHED%" goto FIRST_RUN
set "PYEXE=%PYCACHED%"
goto MENU

:FIRST_RUN
if exist "%MARKER%" del /q "%MARKER%" >nul 2>nul
echo.
echo  ============================================================
echo    AI Xiaofang  -  first run environment setup
echo  ============================================================
echo    machine : %MACHINE_LINE%
echo  ------------------------------------------------------------
echo    Looking for an existing Python ...
echo.

set "PY="
set "PYEXE="
set "PYVER="
call :FIND_PYTHON
if not defined PYEXE goto DOWNLOAD_PYTHON
echo    found  : %PYEXE%
goto SETUP_LIBS

rem ---- 3) no Python at all -> download and install one ----------------
:DOWNLOAD_PYTHON
echo    No Python on this machine yet.
echo    Fetching a matching installer from python.org, please wait ...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0xiaofang_bootstrap.ps1" -Arch %ARCH%
if errorlevel 1 goto NOPY

echo.
echo    Re-scanning for the freshly installed Python ...
set "PY="
set "PYEXE="
set "PYVER="
call :FIND_PYTHON
if not defined PYEXE goto NOPY
echo    found  : %PYEXE%

rem ---- 4) libraries: pip install everything the engine imports -------
:SETUP_LIBS
echo.
echo    Preparing libraries - numpy, psutil, requests, bs4, flask ...
echo.
"%PYEXE%" %PYVER% "xiaofang_launcher.py" --bootstrap
if errorlevel 1 goto BOOTSTRAP_FAIL

rem ---- 5) hand over to the Chinese menu launcher ---------------------
:MENU
"%PYEXE%" %PYVER% "xiaofang_launcher.py" %*
if errorlevel 1 (
    echo.
    echo  [!!] Launcher exited with an error. Press any key to close.
    pause >nul
)
endlocal
goto :EOF

:BOOTSTRAP_FAIL
echo.
echo    Library setup reported a problem. Scroll up for the details.
echo    Start this file again to retry.
echo.
pause
endlocal
exit /b 3

rem ============================================================
rem  Machine check: work out the CPU architecture so we know which
rem  installer flavour to fetch, plus a readable Windows version.
rem ============================================================
:DETECT_MACHINE
set "ARCH="
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=arm64"
if not defined ARCH if /i "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "ARCH=arm64"
if not defined ARCH if /i "%PROCESSOR_ARCHITECTURE%"=="AMD64" set "ARCH=amd64"
if not defined ARCH if /i "%PROCESSOR_ARCHITEW6432%"=="AMD64" set "ARCH=amd64"
if not defined ARCH if /i "%PROCESSOR_ARCHITECTURE%"=="x86" set "ARCH=win32"
if not defined ARCH set "ARCH=amd64"

set "OSVER="
for /f "tokens=4,5 delims=. " %%a in ('ver') do set "OSVER=%%a.%%b"
set "OSNAME="
if "%OSVER%"=="10.0" set "OSNAME=Windows 10 or 11"
if "%OSVER%"=="6.3" set "OSNAME=Windows 8.1"
if "%OSVER%"=="6.1" set "OSNAME=Windows 7"
if not defined OSNAME set "OSNAME=Windows"

set "ARCHNAME=%ARCH%"
if "%ARCH%"=="amd64" set "ARCHNAME=x64 - AMD64"
if "%ARCH%"=="arm64" set "ARCHNAME=ARM64"
if "%ARCH%"=="win32" set "ARCHNAME=x86 - 32-bit"

set "MACHINE_LINE=%OSNAME%, %ARCHNAME%"
set "XF_ARCH=%ARCH%"
set "XF_OSVER=%OSVER%"
goto :EOF

rem ============================================================
rem  Locate a usable Python, in this order:
rem    1. the "py" launcher, which always finds the newest 3.x
rem    2. plain "python" on PATH, but never the Microsoft Store
rem       stub in WindowsApps - that one only opens the Store
rem    3. the folders python.org actually installs into, because a
rem       just-finished install has not refreshed this PATH yet
rem ============================================================
:FIND_PYTHON
for /f "delims=" %%i in ('where py 2^>nul') do (
    echo %%i | find /i "WindowsApps" >nul
    if errorlevel 1 if not defined PYEXE set "PYEXE=%%i"
)
if defined PYEXE (
    set "PYVER=-3"
    goto :EOF
)

for /f "delims=" %%i in ('where python 2^>nul') do (
    echo %%i | find /i "WindowsApps" >nul
    if errorlevel 1 if not defined PY set "PY=%%i"
)
if defined PY set "PYEXE=%PY%"
if defined PY goto :EOF

call :SCAN_PYDIR "%LOCALAPPDATA%\Programs\Python"
if defined PY goto :EOF
call :SCAN_PYDIR "%ProgramFiles%\Python"
if defined PY goto :EOF
if defined PF86 call :SCAN_PYDIR "%PF86%\Python"
goto :EOF

:SCAN_PYDIR
if "%~1"=="" goto :EOF
if not exist "%~1" goto :EOF
for /f "delims=" %%d in ('dir /b /ad /o-n "%~1" 2^>nul') do (
    if not defined PY if exist "%~1\%%d\python.exe" set "PY=%~1\%%d\python.exe"
)
if defined PY set "PYEXE=%PY%"
goto :EOF

:NOLAUNCH
echo  [!!] xiaofang_launcher.py not found next to this .bat.
echo       Keep both files in the same folder and retry.
pause
exit /b 1

:NOPY
echo.
echo  [!!] Python is still not available.
echo.
echo       The automatic setup could not install it. The usual reason
echo       is that this machine has no internet access, or a firewall
echo       blocked the download from python.org.
echo.
echo       Please install it by hand, one time only:
echo         1. open   https://www.python.org/downloads/windows/
echo         2. download the "Windows installer, 64-bit"
echo         3. during setup, tick "Add python.exe to PATH"
echo         4. start this file again
echo.
pause
exit /b 1
