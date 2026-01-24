@echo off
REM ============================================================================
REM   "I bent my Wookie!" - Ralph Wiggum
REM
REM   Emergency stop and recovery - when Ralph broke something
REM ============================================================================

cd /d "D:\_Projects\voiceover-matcher-subtitle"

echo.
echo   =====================================================
echo      "I bent my Wookie!" - Ralph
echo      Emergency Recovery Mode
echo   =====================================================
echo.

echo   Step 1: Checking for running Ralph processes...
tasklist /FI "WINDOWTITLE eq Ralph*" 2>nul | find "powershell" >nul
if %errorlevel%==0 (
    echo   Found running Ralph processes.
    echo.
    set /p KILL="   Kill them? (y/n): "
    if /i "%KILL%"=="y" (
        taskkill /FI "WINDOWTITLE eq Ralph*" /F 2>nul
        echo   Processes terminated.
    )
) else (
    echo   No running Ralph processes found.
)

echo.
echo   Step 2: Current status...
powershell -Command "& {.\scripts\ralph\status.ps1}"

echo.
echo   Step 3: Recent commits (potential rollback targets)...
echo   -----------------------------------------------------
git log --oneline -10
echo   -----------------------------------------------------

echo.
echo   Step 4: Uncommitted changes...
git status --short

echo.
echo   -----------------------------------------------------
echo   Recovery options:
echo   -----------------------------------------------------
echo     git revert HEAD                    Undo last commit
echo     git reset --soft HEAD~1            Undo commit, keep changes
echo     git checkout -- .                  Discard all changes (DANGER)
echo     git stash                          Stash changes for later
echo.
echo     del scripts\ralph\BLOCKED.md       Clear blocked state
echo.

set /p ACTION="   Enter command to run (or press Enter to exit): "
if not "%ACTION%"=="" (
    echo.
    echo   Running: %ACTION%
    %ACTION%
)

echo.
pause
