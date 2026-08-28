@echo off
setlocal

REM ============================================================================
REM LOWER THIRDS
REM Drag and drop a .otio file onto this batch file. It locates the project
REM (walks up to find voiceover/), then renders name/place/date/info overlays
REM via local Ollama + Pillow (no paid software, no AI image generation).
REM ============================================================================

REM ===== EDIT THIS LINE =====
SET MATCHER_ROOT=D:\_Projects\voiceover-matcher-dev
REM ==========================

if "%~1"=="" (
    echo.
    echo ============================================================
    echo   LOWER THIRDS
    echo ============================================================
    echo.
    echo   Usage: Drag and drop a .otio file onto this batch file.
    echo.
    echo   Generates broadcast-style name/place/date/info overlays
    echo   from the project's voiceover SRT and writes lowerthirds.otio.
    echo   Uses local Ollama for entity extraction (free, offline).
    echo.
    pause
    exit /b 1
)

SET OTIO_FILE=%~1

if not exist "%OTIO_FILE%" (
    echo.
    echo ERROR: File not found: %OTIO_FILE%
    echo.
    pause
    exit /b 1
)

SET EXT=%~x1
if /i not "%EXT%"==".otio" (
    echo.
    echo WARNING: File does not have .otio extension: %EXT%
    echo Proceeding anyway...
    echo.
)

if not exist "%MATCHER_ROOT%\scripts\lower_thirds.py" (
    echo.
    echo ERROR: lower_thirds.py not found at %MATCHER_ROOT%\scripts\
    echo Edit MATCHER_ROOT at the top of this file.
    echo.
    pause
    exit /b 1
)

cd /d "%MATCHER_ROOT%"

REM Resolve project root from OTIO path (walks up looking for voiceover/)
for /f "delims=" %%P in ('python "%MATCHER_ROOT%\scripts\_find_project_root.py" "%OTIO_FILE%"') do set "PROJECT_DIR=%%P"

if "%PROJECT_DIR%"=="" (
    echo.
    echo ERROR: Could not locate a voiceover\ folder above:
    echo   %OTIO_FILE%
    echo.
    echo Searched up to 5 ancestor directories for:
    echo   voiceover\voiceover_trimmed.srt
    echo   voiceover\voiceover.srt
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   LOWER THIRDS
echo ============================================================
echo   OTIO:        %OTIO_FILE%
echo   Project:     %PROJECT_DIR%
echo   Output OTIO: %PROJECT_DIR%\lowerthirds.otio
echo.

python "%MATCHER_ROOT%\scripts\lower_thirds.py" "%PROJECT_DIR%" --output-otio "%PROJECT_DIR%\lowerthirds.otio"

echo.
echo ============================================================
echo   EXITED (code %ERRORLEVEL%)
echo ============================================================
echo.
pause