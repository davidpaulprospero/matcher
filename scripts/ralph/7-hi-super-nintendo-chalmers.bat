@echo off
REM ============================================================================
REM   "Hi, Super Nintendo Chalmers!" - Ralph Wiggum
REM
REM   Interview Mode: Tell Ralph what you want, he'll make it happen
REM ============================================================================

cd /d "%~dp0..\.."

echo.
echo   =====================================================
echo      "Hi, Super Nintendo Chalmers!" - Ralph
echo      Interview Mode
echo   =====================================================
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0interview.ps1" %*
