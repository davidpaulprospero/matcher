@echo off
setlocal

REM ============================================================================
REM PIPELINE WATCH (5-min polling)
REM Place this .bat at the root of a project folder (alongside voiceover/).
REM Edit MATCHER_ROOT below to point at your voiceover-matcher repo, then
REM double-click to start an infinite watch loop.
REM
REM Each cycle: runs watch_auto.py for 4m50s, then sleeps 5 min, then repeats.
REM Press Ctrl+C in this window to stop.
REM ============================================================================

REM ===== EDIT THIS LINE =====
SET MATCHER_ROOT=D:\_Projects\voiceover-matcher-dev
REM ==========================

SET PROJECT_DIR=%~dp0

if not exist "%MATCHER_ROOT%\scripts\watch_auto.py" (
    echo.
    echo ERROR: watch_auto.py not found at %MATCHER_ROOT%\scripts\watch_auto.py
    echo Edit MATCHER_ROOT at the top of this file.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   PIPELINE WATCH STARTED
echo ============================================================
echo   Project:    %PROJECT_DIR%
echo   Matcher:    %MATCHER_ROOT%
echo   Polling:    every ~5 minutes
echo   Stop with:  Ctrl+C
echo.

cd /d "%MATCHER_ROOT%"

:loop
echo.
echo [%date% %time%] Watch cycle starting...
python "%MATCHER_ROOT%\scripts\watch_auto.py" --project-path "%PROJECT_DIR%" --duration 290
echo [%date% %time%] Watch cycle ended (exit %ERRORLEVEL%). Sleeping 5 min...
timeout /t 300 /nobreak >nul
goto loop