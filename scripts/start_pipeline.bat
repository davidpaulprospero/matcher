@echo off
setlocal

REM ============================================================================
REM START PIPELINE
REM Place this .bat at the root of a project folder (alongside voiceover/ or
REM with the voiceover SRT/MP3 directly in the project root).
REM Edit MATCHER_ROOT below to point at your voiceover-matcher repo, then
REM double-click to launch the pipeline.
REM
REM The bat delegates voiceover discovery to main.py (which checks
REM voiceover/ first, then the project root for any .srt/.mp3/.wav/.mp4).
REM ============================================================================

REM ===== EDIT THIS LINE =====
SET MATCHER_ROOT=D:\_Projects\voiceover-matcher-dev
REM ==========================

REM ===== OPTIONAL MODE =====
REM MODE=full   = resume from checkpoint, run remaining stages (default)
REM MODE=fresh  = wipe checkpoint, re-run everything from ANALYZE
REM MODE=match  = skip download/transcribe, run matching only
REM MODE=output = regenerate OTIO/EDL/XML only (fastest, needs checkpoint)
SET MODE=full
REM ==========================

SET PROJECT_DIR=%~dp0

if not exist "%MATCHER_ROOT%\main.py" (
    echo.
    echo ============================================================
    echo   START PIPELINE - ERROR
    echo ============================================================
    echo.
    echo   main.py not found at: %MATCHER_ROOT%\main.py
    echo   Edit MATCHER_ROOT at the top of this file.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   START PIPELINE
echo ============================================================
echo   Project:  %PROJECT_DIR%
echo   Matcher:  %MATCHER_ROOT%
echo   Mode:     %MODE%
echo.

cd /d "%MATCHER_ROOT%"

if /i "%MODE%"=="fresh" (
    echo   WARNING: MODE=fresh will wipe the existing checkpoint.
    echo   Press Ctrl+C within 5 seconds to abort, otherwise continuing...
    timeout /t 5 /nobreak >nul
    python main.py --project "%PROJECT_DIR%" --fresh --non-interactive
) else if /i "%MODE%"=="match" (
    python main.py --project "%PROJECT_DIR%" --match-only --non-interactive
) else if /i "%MODE%"=="output" (
    python main.py --project "%PROJECT_DIR%" --output-only --non-interactive
) else (
    python main.py --project "%PROJECT_DIR%" --non-interactive
)

echo.
echo ============================================================
echo   PIPELINE EXITED (code %ERRORLEVEL%)
echo ============================================================
echo.
pause