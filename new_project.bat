@echo off
REM ============================================
REM  VOICEOVER-MATCHER: Create New Project
REM ============================================
REM
REM Double-click to run, or use command line:
REM   new_project.bat                              (prompts for input)
REM   new_project.bat "E:\Projects\MyDoc"          (full path)
REM   new_project.bat "E:\Projects" "Japan Quake"  (base + name)
REM

setlocal enabledelayedexpansion

REM Get install directory
set "INSTALL_DIR=%~dp0"
if "%INSTALL_DIR:~-1%"=="\" set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"

cd /d "%INSTALL_DIR%"

echo.
echo ========================================
echo   VOICEOVER-MATCHER: New Project
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM If arguments provided, use setup_project.py directly
if not "%~1"=="" (
    if not "%~2"=="" (
        REM Two args: base + name
        call :create_with_date "%~1" "%~2"
    ) else (
        REM One arg: full path
        python setup_project.py "%~1"
    )
    goto :end
)

REM ============================================
REM  Interactive Mode
REM ============================================

REM Default base path
set "DEFAULT_BASE=E:\Edit Job"

echo   Default base: %DEFAULT_BASE%
echo.
set /p "BASE_PATH=  Base path (Enter for default): "

if "%BASE_PATH%"=="" set "BASE_PATH=%DEFAULT_BASE%"

REM Remove quotes if user added them
set "BASE_PATH=%BASE_PATH:"=%"

echo.
echo   Selected: %BASE_PATH%
echo.

REM Check if base exists
if not exist "%BASE_PATH%" (
    echo   Base path doesn't exist. Create it? [Y/n]
    set /p "CREATE_BASE="
    if /i "!CREATE_BASE!"=="n" (
        echo   Cancelled.
        goto :end
    )
    mkdir "%BASE_PATH%" 2>nul
    echo   Created: %BASE_PATH%
)

REM List existing folders
echo.
echo   Existing folders in %BASE_PATH%:
echo   ----------------------------------------
set "COUNT=0"
for /d %%D in ("%BASE_PATH%\*") do (
    set /a COUNT+=1
    echo   !COUNT!. %%~nxD
)
if %COUNT%==0 echo   (none)
echo   ----------------------------------------
echo.

REM Ask for client/category folder
set /p "CLIENT_FOLDER=  Client/category folder (or Enter to skip): "
if not "%CLIENT_FOLDER%"=="" (
    set "BASE_PATH=%BASE_PATH%\%CLIENT_FOLDER%"
    if not exist "!BASE_PATH!" mkdir "!BASE_PATH!"
)

REM Ask for series folder
echo.
set /p "SERIES_FOLDER=  Series/show folder (or Enter to skip): "
if not "%SERIES_FOLDER%"=="" (
    set "BASE_PATH=%BASE_PATH%\%SERIES_FOLDER%"
    if not exist "!BASE_PATH!" mkdir "!BASE_PATH!"
)

REM Ask for project name
echo.
echo   Project will be created in: %BASE_PATH%
echo.
set /p "PROJECT_NAME=  Project name (e.g., Japan Earthquake): "

if "%PROJECT_NAME%"=="" (
    echo   No name provided. Cancelled.
    goto :end
)

REM Create with date
call :create_with_date "%BASE_PATH%" "%PROJECT_NAME%"
goto :end

REM ============================================
REM  Function: Create project with date suffix
REM ============================================
:create_with_date
set "BASE=%~1"
set "NAME=%~2"

REM Get current date (YYYY-MM-DD format)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DT=%%I"
set "TODAY=%DT:~0,4%-%DT:~4,2%-%DT:~6,2%"

REM Sanitize project name (remove invalid chars)
set "NAME=%NAME:<=%"
set "NAME=%NAME:>=%"
set "NAME=%NAME::=%"
set "NAME=%NAME:/=%"
set "NAME=%NAME:\=%"
set "NAME=%NAME:|=%"
set "NAME=%NAME:?=%"
set "NAME=%NAME:*=%"

set "FULL_PATH=%BASE%\%NAME%__%TODAY%"

echo.
echo   Creating: %FULL_PATH%
echo.

REM Create directories
mkdir "%FULL_PATH%\voiceover" 2>nul
mkdir "%FULL_PATH%\downloaded_videos" 2>nul
mkdir "%FULL_PATH%\otio_output" 2>nul
mkdir "%FULL_PATH%\.cache" 2>nul
mkdir "%FULL_PATH%\logs" 2>nul

REM Create run.bat
(
echo @echo off
echo REM Voiceover-Matcher Runner
echo REM Project: %NAME%
echo.
echo set "INSTALL_DIR=%INSTALL_DIR%"
echo set "PROJECT_DIR=%%~dp0"
echo if "%%PROJECT_DIR:~-1%%"=="\" set "PROJECT_DIR=%%PROJECT_DIR:~0,-1%%"
echo.
echo cd /d "%%INSTALL_DIR%%"
echo python main.py --project "%%PROJECT_DIR%%" %%*
echo.
echo if %%ERRORLEVEL%% NEQ 0 pause
) > "%FULL_PATH%\run.bat"

REM Create README files
echo Place your voiceover .srt or .mp3 files here> "%FULL_PATH%\voiceover\README.txt"
echo Downloaded stock footage will be saved here> "%FULL_PATH%\downloaded_videos\README.txt"
echo Generated OTIO timeline files> "%FULL_PATH%\otio_output\README.txt"

REM Create project_config.yaml
(
echo # Project-Specific Configuration Overrides
echo # Uncomment settings to override global config.yaml
echo.
echo # voiceover_path: "./voiceover/script.srt"
echo.
echo # matching:
echo #   confidence_threshold: 0.4
echo.
echo # output:
echo #   num_alternatives: 3
) > "%FULL_PATH%\project_config.yaml"

echo   ========================================
echo     Project Created Successfully!
echo   ========================================
echo.
echo   Folders:
echo     [OK] voiceover/
echo     [OK] downloaded_videos/
echo     [OK] otio_output/
echo     [OK] .cache/
echo     [OK] logs/
echo.
echo   Files:
echo     [OK] run.bat
echo     [OK] project_config.yaml
echo.
echo   ----------------------------------------
echo   Next steps:
echo   1. Put voiceover in: %FULL_PATH%\voiceover\
echo   2. Double-click run.bat to start
echo   ----------------------------------------
echo.

exit /b 0

:end
echo.
pause
