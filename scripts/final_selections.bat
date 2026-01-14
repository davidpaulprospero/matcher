@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Final Selections Analyzer
REM Drag and drop an exported OTIO file onto this batch file to analyze
REM which clips you selected in your final edit.
REM ============================================================================

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"

REM Check if OTIO file was provided
if "%~1"=="" (
    echo.
    echo ============================================================
    echo   FINAL SELECTIONS ANALYZER
    echo ============================================================
    echo.
    echo   Usage: Drag and drop an exported OTIO file onto this batch file
    echo.
    echo   This tool analyzes your final edit to:
    echo     - Track which clips you selected for each segment
    echo     - Update the global cache for future matching improvement
    echo     - Generate selection reports
    echo.
    echo   Press any key to exit...
    pause >nul
    exit /b 1
)

REM Get the OTIO file path
set "OTIO_FILE=%~1"

REM Check if file exists
if not exist "%OTIO_FILE%" (
    echo.
    echo ERROR: File not found: %OTIO_FILE%
    echo.
    pause
    exit /b 1
)

REM Check if it's an OTIO file
set "EXT=%~x1"
if /i not "%EXT%"==".otio" (
    echo.
    echo WARNING: File does not have .otio extension: %EXT%
    echo Proceeding anyway...
    echo.
)

echo.
echo ============================================================
echo   FINAL SELECTIONS ANALYZER
echo ============================================================
echo.
echo   OTIO: %~nx1
echo   Path: %~dp1
echo.

REM Change to script directory
cd /d "%SCRIPT_DIR%"

REM Run the Python analyzer
python "%SCRIPT_DIR%analyze_final_edit.py" "%OTIO_FILE%"

REM Check result
if errorlevel 1 (
    echo.
    echo ERROR: Analysis failed. Check the error messages above.
    echo.
) else (
    echo.
    echo ============================================================
    echo   ANALYSIS COMPLETE
    echo ============================================================
    echo.
)

echo Press any key to exit...
pause >nul
