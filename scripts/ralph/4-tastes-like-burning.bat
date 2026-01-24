@echo off
REM ============================================================================
REM   "It tastes like burning!" - Ralph Wiggum
REM
REM   View logs when something went wrong
REM ============================================================================

cd /d "D:\_Projects\voiceover-matcher-subtitle"

echo.
echo   =====================================================
echo      "It tastes like burning!" - Ralph
echo      Log Viewer
echo   =====================================================
echo.

REM Find latest log directory (first result from reverse-sorted list)
set LATEST_LOG=
for /f "delims=" %%i in ('dir /b /ad /o-n scripts\ralph\logs 2^>nul') do if not defined LATEST_LOG set LATEST_LOG=%%i

if "%LATEST_LOG%"=="" (
    echo   No logs found yet. Run Ralph first!
    pause
    exit /b
)

echo   Latest session: %LATEST_LOG%
echo.

echo   -----------------------------------------------------
echo   Session Report:
echo   -----------------------------------------------------
if exist "scripts\ralph\logs\%LATEST_LOG%\REPORT.md" (
    type "scripts\ralph\logs\%LATEST_LOG%\REPORT.md"
) else (
    echo   No report yet (session may still be running)
)

echo.
echo   -----------------------------------------------------
echo   Recent session log:
echo   -----------------------------------------------------
if exist "scripts\ralph\logs\%LATEST_LOG%\session.log" (
    powershell -Command "Get-Content 'scripts\ralph\logs\%LATEST_LOG%\session.log' | Select-Object -Last 30"
)

echo.
echo   -----------------------------------------------------
echo   BLOCKED status:
echo   -----------------------------------------------------
if exist "scripts\ralph\BLOCKED.md" (
    echo   !! BLOCKED !!
    type "scripts\ralph\BLOCKED.md"
) else (
    echo   Not blocked.
)

echo.
echo   -----------------------------------------------------
echo   Options:
echo   -----------------------------------------------------
echo     1. View full session log
echo     2. View latest iteration log
echo     3. View metrics
echo     4. Open logs folder
echo     5. Exit
echo.

set /p CHOICE="   Choose (1-5): "

if "%CHOICE%"=="1" (
    if exist "scripts\ralph\logs\%LATEST_LOG%\session.log" (
        notepad "scripts\ralph\logs\%LATEST_LOG%\session.log"
    )
)
if "%CHOICE%"=="2" (
    set LATEST_ITER=
    for /f "delims=" %%i in ('dir /b /o-n "scripts\ralph\logs\%LATEST_LOG%\S*.log" 2^>nul') do if not defined LATEST_ITER set LATEST_ITER=%%i
    if defined LATEST_ITER notepad "scripts\ralph\logs\%LATEST_LOG%\%LATEST_ITER%"
)
if "%CHOICE%"=="3" (
    if exist "scripts\ralph\metrics.csv" (
        start "" "scripts\ralph\metrics.csv"
    ) else (
        echo   No metrics yet.
        pause
    )
)
if "%CHOICE%"=="4" (
    explorer "scripts\ralph\logs\%LATEST_LOG%"
)
