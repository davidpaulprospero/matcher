@echo off
setlocal

REM ============================================================================
REM DISTRIBUTE REFERENCE PICTURES
REM Drag and drop a .otio file onto this batch file. It sprays reference images
REM (downloaded from the web via pyimagedl) as a new V-Ref track.
REM
REM This bat intentionally does NOT use the minimax image provider — that
REM subscription is cancelled. The default pyimagedl path keeps working.
REM ============================================================================

REM ===== EDIT THIS LINE =====
SET MATCHER_ROOT=D:\_Projects\voiceover-matcher-dev
REM ==========================

if "%~1"=="" (
    echo.
    echo ============================================================
    echo   DISTRIBUTE REFERENCE PICTURES
    echo ============================================================
    echo.
    echo   Usage: Drag and drop a .otio file onto this batch file.
    echo.
    echo   This sprays reference pictures as a V-Ref track on top of
    echo   an existing timeline. Image source: web download (pyimagedl).
    echo   No AI image generation is used.
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

if not exist "%MATCHER_ROOT%\scripts\distribute_ref_pictures.py" (
    echo.
    echo ERROR: distribute_ref_pictures.py not found at %MATCHER_ROOT%\scripts\
    echo Edit MATCHER_ROOT at the top of this file.
    echo.
    pause
    exit /b 1
)

cd /d "%MATCHER_ROOT%"

REM Resolve project root + SRT from OTIO path (walks up looking for voiceover/ or SRT)
set "PROJECT_DIR="
set "SRT_FILE="
for /f "delims=" %%P in ('python "%MATCHER_ROOT%\scripts\_find_project_root.py" "%OTIO_FILE%"') do (
    if not defined PROJECT_DIR (
        set "PROJECT_DIR=%%P"
    ) else if not defined SRT_FILE (
        set "SRT_FILE=%%P"
    )
)

if "%PROJECT_DIR%"=="" (
    echo.
    echo ERROR: Could not locate a project (voiceover/ folder or SRT/MP3 file)
    echo above: %OTIO_FILE%
    echo.
    echo Searched up to 5 ancestor directories.
    echo.
    pause
    exit /b 1
)

if "%SRT_FILE%"=="" (
    echo.
    echo ERROR: Found project at %PROJECT_DIR% but no .srt file inside.
    echo distribute_ref_pictures.py requires an SRT.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   DISTRIBUTE REFERENCE PICTURES
echo ============================================================
echo   OTIO:        %OTIO_FILE%
echo   Project:     %PROJECT_DIR%
echo   SRT:         %SRT_FILE%
echo   Provider:    pyimagedl  ^(web download — no AI image gen^)
echo.

python "%MATCHER_ROOT%\scripts\distribute_ref_pictures.py" --otio "%OTIO_FILE%" --srt "%SRT_FILE%"

echo.
echo ============================================================
echo   EXITED (code %ERRORLEVEL%)
echo ============================================================
echo.
pause