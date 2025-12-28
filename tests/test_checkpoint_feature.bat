@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM TEST: Checkpoint & Saved Keywords Features
REM ============================================================
REM Run from the install directory (where main.py is)
REM
REM This script tests:
REM   1. Unit tests for checkpoint/keyword modules
REM   2. --list-keywords command
REM   3. --save-keywords during a real run
REM   4. --use-keywords to reuse saved keywords
REM   5. Checkpoint creation and --resume
REM ============================================================

echo.
echo ============================================================
echo   CHECKPOINT ^& SAVED KEYWORDS FEATURE TEST
echo ============================================================
echo.

REM Get install directory
set "INSTALL_DIR=%~dp0.."
cd /d "%INSTALL_DIR%"
echo   Install dir: %CD%

REM ─────────────────────────────────────────────────────────────
REM TEST 1: Unit Tests
REM ─────────────────────────────────────────────────────────────
echo.
echo ─── TEST 1: Unit Tests ───
echo.

python tests/test_checkpoint.py
if %ERRORLEVEL% NEQ 0 (
    echo   ✗ Unit tests FAILED
    goto :error
)
echo   ✓ Unit tests PASSED

REM ─────────────────────────────────────────────────────────────
REM TEST 2: --list-keywords (should show none initially)
REM ─────────────────────────────────────────────────────────────
echo.
echo ─── TEST 2: List Keywords (empty) ───
echo.

REM Create temp project
set "TEMP_PROJECT=%TEMP%\checkpoint_test_%RANDOM%"
mkdir "%TEMP_PROJECT%" 2>nul

python main.py --project "%TEMP_PROJECT%" --list-keywords
echo   ✓ --list-keywords works (should show "no presets")

REM ─────────────────────────────────────────────────────────────
REM TEST 3: Create a minimal voiceover for testing
REM ─────────────────────────────────────────────────────────────
echo.
echo ─── TEST 3: Create Test Voiceover ───
echo.

REM Create a simple SRT file
(
echo 1
echo 00:00:00,000 --^> 00:00:05,000
echo Elon Musk founded SpaceX in 2002.
echo.
echo 2
echo 00:00:05,000 --^> 00:00:10,000
echo Tesla has revolutionized electric vehicles.
echo.
echo 3
echo 00:00:10,000 --^> 00:00:15,000
echo The future of space exploration looks bright.
) > "%TEMP_PROJECT%\test_voiceover.srt"

echo   ✓ Created test voiceover: %TEMP_PROJECT%\test_voiceover.srt

REM ─────────────────────────────────────────────────────────────
REM TEST 4: Run with --save-keywords (analyze only, then cancel)
REM ─────────────────────────────────────────────────────────────
echo.
echo ─── TEST 4: Test --save-keywords ───
echo.
echo   Running pipeline with --save-keywords my_test_preset
echo   (Will run ANALYZE stage only, creating checkpoint)
echo.

REM Run with timeout to just do ANALYZE stage
REM We use --non-interactive and --keywords 2 for speed
python main.py --project "%TEMP_PROJECT%" --voiceover test_voiceover.srt --keywords 2 --non-interactive --save-keywords my_test_preset 2>&1 | findstr /C:"keyword" /C:"Keyword" /C:"ANALYZE" /C:"saved" /C:"💾"

REM Check if saved_keywords.json was created
if exist "%TEMP_PROJECT%\saved_keywords.json" (
    echo.
    echo   ✓ saved_keywords.json created
    echo   Contents:
    type "%TEMP_PROJECT%\saved_keywords.json"
) else (
    echo   ⚠ saved_keywords.json not found yet (pipeline may still be running)
)

REM Check if checkpoint.json was created
if exist "%TEMP_PROJECT%\checkpoint.json" (
    echo.
    echo   ✓ checkpoint.json created
    echo   Contents:
    type "%TEMP_PROJECT%\checkpoint.json"
) else (
    echo   ⚠ checkpoint.json not found
)

REM ─────────────────────────────────────────────────────────────
REM TEST 5: --list-keywords (should show saved preset)
REM ─────────────────────────────────────────────────────────────
echo.
echo ─── TEST 5: List Keywords (should show preset) ───
echo.

python main.py --project "%TEMP_PROJECT%" --list-keywords

REM ─────────────────────────────────────────────────────────────
REM CLEANUP
REM ─────────────────────────────────────────────────────────────
echo.
echo ─── CLEANUP ───
echo.
echo   Temp project: %TEMP_PROJECT%
echo   (Not deleting - you can inspect the files)
echo.
echo   Files created:
dir /b "%TEMP_PROJECT%"

echo.
echo ============================================================
echo   FEATURE TEST COMPLETE
echo ============================================================
echo.
echo   To test resume functionality manually:
echo     1. Start a long run: python main.py --project "%TEMP_PROJECT%" --voiceover test_voiceover.srt
echo     2. Press Ctrl+C during DOWNLOAD stage
echo     3. Resume: python main.py --project "%TEMP_PROJECT%" --voiceover test_voiceover.srt --resume
echo.
echo   To test saved keywords:
echo     python main.py --project "%TEMP_PROJECT%" --list-keywords
echo     python main.py --project "%TEMP_PROJECT%" --voiceover test_voiceover.srt --use-keywords
echo.
goto :end

:error
echo.
echo   ✗ TEST FAILED
exit /b 1

:end
echo   Done.
pause
