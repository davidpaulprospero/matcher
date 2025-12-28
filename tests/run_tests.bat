@echo off
REM ============================================================
REM Feature Test Runner
REM Tests recent features with actual file processing
REM ============================================================

cd /d "%~dp0.."

echo.
echo   ============================================================
echo   FEATURE TEST SUITE
echo   ============================================================
echo.

REM Check for arguments
if "%1"=="--keep" (
    python tests/test_features.py --keep-videos
) else if "%1"=="--skip" (
    python tests/test_features.py --skip-download
) else (
    python tests/test_features.py
)

echo.
pause
