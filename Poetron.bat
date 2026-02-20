@echo off
REM Simple launcher for Poetron - Alternative to .exe
REM This batch file can be used before building the .exe

echo ============================================================
echo   POETRON - Haiku Generator Launcher
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Check if this is first run
if not exist "models\" (
    echo [INFO] First time setup detected
    echo [INFO] Running setup_and_run.py...
    echo.
    python setup_and_run.py
) else (
    echo [INFO] Launching Poetron...
    echo.
    python interactive_haiku.py
)

if errorlevel 1 (
    echo.
    echo [ERROR] An error occurred
    pause
)
