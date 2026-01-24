@echo off
REM ============================================================================
REM   "Me fail English? That's unpossible!" - Ralph Wiggum
REM
REM   Quick status check - see what Ralph has been up to
REM ============================================================================

cd /d "D:\_Projects\voiceover-matcher-subtitle"

echo.
echo   =====================================================
echo      "Me fail English? That's unpossible!" - Ralph
echo   =====================================================
echo.

powershell -Command "& {.\scripts\ralph\status.ps1}"

echo.
echo   -----------------------------------------------------
echo   Recent git activity:
echo   -----------------------------------------------------
git log --oneline -8

echo.
echo   -----------------------------------------------------
echo   Quick commands:
echo   -----------------------------------------------------
echo     1-im-learnding.bat                            Morning check-in
echo     2-sleep-thats-where-im-a-viking.bat           Launch Ralph + Watch
echo     2-sleep-thats-where-im-a-viking.bat overnight Overnight mode
echo     2-sleep-thats-where-im-a-viking.bat yolo      Full auto
echo     3-me-fail-english.bat                         Quick status (this)
echo     4-tastes-like-burning.bat                     View logs
echo     5-i-bent-my-wookie.bat                        Emergency recovery
echo.

pause
