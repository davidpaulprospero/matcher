@echo off
REM ============================================================================
REM   "I'm learnding!" - Ralph Wiggum
REM
REM   Morning check-in: See what Ralph learned overnight
REM ============================================================================

cd /d "D:\_Projects\voiceover-matcher-subtitle"

echo.
echo   =====================================================
echo      "I'm learnding!" - Ralph
echo      Morning Check-in
echo   =====================================================
echo.

REM Find latest log directory (first result from reverse-sorted list)
set LATEST_LOG=
for /f "delims=" %%i in ('dir /b /ad /o-n scripts\ralph\logs 2^>nul') do if not defined LATEST_LOG set LATEST_LOG=%%i

echo   Latest session: %LATEST_LOG%
echo.

echo   -----------------------------------------------------
echo   Progress Summary:
echo   -----------------------------------------------------
powershell -Command "& {.\scripts\ralph\status.ps1}"

echo.
echo   -----------------------------------------------------
echo   What Ralph Did (commits):
echo   -----------------------------------------------------
git log --oneline --since="12 hours ago" 2>nul || git log --oneline -10

echo.
echo   -----------------------------------------------------
echo   Metrics Summary:
echo   -----------------------------------------------------
if exist "scripts\ralph\metrics.csv" (
    powershell -Command "Import-Csv 'scripts\ralph\metrics.csv' | Select-Object -Last 10 | Format-Table -AutoSize"
) else (
    echo   No metrics yet.
)

echo.
echo   -----------------------------------------------------
echo   Any Blockers?
echo   -----------------------------------------------------
if exist "scripts\ralph\BLOCKED.md" (
    echo   !! YES - Ralph needs help !!
    echo.
    type "scripts\ralph\BLOCKED.md"
    echo.
    set /p CLEAR="   Clear blocker and continue? (y/n): "
    if /i "%CLEAR%"=="y" (
        del "scripts\ralph\BLOCKED.md"
        echo   Blocker cleared.
    )
) else (
    echo   No blockers. Ralph is good to go!
)

echo.
echo   -----------------------------------------------------
echo   What's next?
echo   -----------------------------------------------------
echo     1. Continue Ralph (interactive)
echo     2. Start TrueAuto mode
echo     3. Interview mode (give specific direction)
echo     4. View detailed logs
echo     5. Just exit
echo.

set /p NEXT="   Choose (1-5): "

if "%NEXT%"=="1" (
    start "Ralph Loop" powershell -NoExit -Command "& {Set-Location 'D:\_Projects\voiceover-matcher-subtitle'; .\scripts\ralph\ralph.ps1}"
    start "Ralph Watch" powershell -NoExit -Command "& {Set-Location 'D:\_Projects\voiceover-matcher-subtitle'; .\scripts\ralph\watch.ps1}"
)
if "%NEXT%"=="2" (
    start "Ralph Loop" powershell -NoExit -Command "& {Set-Location 'D:\_Projects\voiceover-matcher-subtitle'; .\scripts\ralph\ralph.ps1 -TrueAuto -SkipPlanApproval}"
    start "Ralph Watch" powershell -NoExit -Command "& {Set-Location 'D:\_Projects\voiceover-matcher-subtitle'; .\scripts\ralph\watch.ps1}"
)
if "%NEXT%"=="3" (
    call "%~dp07-hi-super-nintendo-chalmers.bat"
)
if "%NEXT%"=="4" (
    call "%~dp04-tastes-like-burning.bat"
)
