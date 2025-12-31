@echo off
REM ============================================================
REM Test Recent Feature Additions
REM ============================================================
REM Tests the following recent features:
REM   1. Delta matching - only match new videos
REM   2. Cache loading - transcripts from cache
REM   3. Chapter-based topic matching
REM   4. OTIO V9/V10 track inclusion
REM   5. JSON serialization (numpy types)
REM ============================================================

cd /d "%~dp0.."

echo.
echo   ============================================================
echo   RECENT FEATURES TEST
echo   ============================================================
echo.
echo   Testing:
echo     - Delta matching index
echo     - Topic extraction and chapter detection
echo     - JSON serialization (numpy types)
echo     - OTIO V9/V10 tracks
echo     - Config chapter options
echo.

REM Check for arguments
if "%1"=="--verbose" (
    python tests/test_recent_features.py --verbose
) else if "%1"=="-v" (
    python tests/test_recent_features.py --verbose
) else (
    python tests/test_recent_features.py
)

echo.
pause
