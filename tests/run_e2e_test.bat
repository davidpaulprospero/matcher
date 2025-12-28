@echo off
REM ============================================================
REM End-to-End Pipeline Test
REM Actually runs main.py and validates outputs
REM ============================================================

cd /d "%~dp0.."

echo.
echo   ============================================================
echo   END-TO-END PIPELINE TEST
echo   ============================================================
echo.
echo   This will:
echo   1. Create a temp project folder
echo   2. Download a voiceover and transcribe it
echo   3. Run the REAL pipeline (main.py)
echo   4. Validate all outputs (OTIO, XML, logs)
echo.

REM Check for arguments
if "%1"=="--keep" (
    python tests/test_e2e.py --keep
) else if "%1"=="--quiet" (
    python tests/test_e2e.py --quiet
) else if "%1"=="-q" (
    python tests/test_e2e.py --quiet
) else (
    python tests/test_e2e.py
)

echo.
pause
