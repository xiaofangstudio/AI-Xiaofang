@echo off
rem ============================================================
rem  AI Xiaofang - one-click EXTREME TRAINING launcher
rem  (kept pure ASCII on purpose - Chinese lives in Python files)
rem ------------------------------------------------------------
rem  What double-clicking this does
rem    boots AI Xiaofang exactly like normal (same warm-up, same
rem    weights cache), then instead of entering the chat REPL it
rem    runs a tight, max-speed training loop that hammers the
rem    network with FOUR real corpora:
rem      1. the 3 novels on the Desktop (xiao fang shuo hua) - learn
rem         to speak natural Chinese
rem      2. the Data code bank (Python knowledge + distilled usage
rem         of its own functions) - learn to REALLY understand code
rem      3. its own source-code Chinese comments
rem      4. the conversation replay
rem    refined weights are written back to the same training NPZ.
rem    Memory is pinned to the lite hard cap (5.0 GB) so it stays
rem    safely below the 6 GB limit. Live progress bar shows the real
rem    phase, real steps, real loss and real ETA (no fake prints).
rem
rem  To stop: press Ctrl+C. Weights + replay are fully saved on
rem  exit, then the process ends with nothing left running.
rem ============================================================
chcp 65001 >nul
title AI Xiaofang - extreme training
setlocal EnableExtensions
cd /d "%~dp0"

set "XIAOFANG_TRAIN=on"
set "XF_TIER=lite"
set "XF_TRAIN_SEQ=384"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem ---- pick the python the launcher already fixed up -------------
set "PYEXE="
if exist ".xf_env_ok" set /p PYEXE=<".xf_env_ok"
if not defined PYEXE for /f "delims=" %%i in ('where py 2^>nul') do if not defined PYEXE set "PYEXE=%%i"
if not defined PYEXE for /f "delims=" %%i in ('where python 2^>nul') do if not defined PYEXE set "PYEXE=%%i"

if not defined PYEXE (
    echo.
    echo  [!!] Python not found. Run "start Xiaofang" once first
    echo       so the environment and interpreter are set up.
    echo.
    pause
    exit /b 1
)
if not exist "xiaofang_covi1.py" (
    echo  [!!] xiaofang_covi1.py not found next to this .bat.
    pause
    exit /b 1
)

echo.
echo  EXTREME TRAINING - AI Xiaofang hammers its own source-code
echo  Chinese comments plus conversation replay at full speed.
echo  Target memory: below 6 GB. Stop anytime with Ctrl+C.
echo.
"%PYEXE%" "xiaofang_covi1.py"
if errorlevel 1 (
    echo.
    echo  [!!] Training exited with an error. Scroll up for the cause.
    pause >nul
)
endlocal