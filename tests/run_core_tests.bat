@echo off
REM ============================================================
REM Core Component Test Runner
REM Tests all core components with real API calls
REM ============================================================

cd /d "%~dp0.."

echo.
echo   ============================================================
echo   CORE COMPONENT TEST SUITE
echo   ============================================================
echo.

REM Check for arguments
if "%1"=="--keep" (
    python tests/test_core.py --keep
) else if "%1"=="--skip" (
    python tests/test_core.py --skip-download
) else if "%1"=="--verbose" (
    python tests/test_core.py --verbose
) else if "%1"=="-v" (
    python tests/test_core.py --verbose
) else (
    python tests/test_core.py
)

echo.
pause
