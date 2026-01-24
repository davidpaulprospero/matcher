@echo off
REM ============================================================================
REM   "Sleep! That's where I'm a Viking!" - Ralph Wiggum
REM
REM   Launches Ralph Loop + Watch Dashboard side by side
REM   Perfect for overnight runs or AFK sessions
REM ============================================================================

title Ralph Loop Launcher

echo.
echo   =====================================================
echo      "Sleep! That's where I'm a Viking!" - Ralph
echo   =====================================================
echo.
echo   This will open two windows:
echo     1. Ralph Loop (main worker)
echo     2. Watch Dashboard (monitoring)
echo.
echo   Press Ctrl+C in the Ralph window to stop.
echo.

cd /d "D:\_Projects\voiceover-matcher-subtitle"

REM Parse arguments
set FOCUS=
set TRUEAUTO=
set SKIP_APPROVAL=
set TASK=
set QUEUE=

:parse_args
if "%~1"=="" goto done_parsing
if /i "%~1"=="-f" set FOCUS=%~2& shift & shift & goto parse_args
if /i "%~1"=="--focus" set FOCUS=%~2& shift & shift & goto parse_args
if /i "%~1"=="-t" set TRUEAUTO=-TrueAuto& shift & goto parse_args
if /i "%~1"=="--trueauto" set TRUEAUTO=-TrueAuto& shift & goto parse_args
if /i "%~1"=="-s" set SKIP_APPROVAL=-SkipPlanApproval& shift & goto parse_args
if /i "%~1"=="--skip" set SKIP_APPROVAL=-SkipPlanApproval& shift & goto parse_args
if /i "%~1"=="--task" set TASK=-Task "%~2"& shift & shift & goto parse_args
if /i "%~1"=="-q" set QUEUE=-Queue& shift & goto parse_args
if /i "%~1"=="--queue" set QUEUE=-Queue& shift & goto parse_args
if /i "%~1"=="overnight" (
    set TRUEAUTO=-TrueAuto
    set SKIP_APPROVAL=-SkipPlanApproval
    set FOCUS=testing
    shift
    goto parse_args
)
if /i "%~1"=="yolo" (
    set TRUEAUTO=-TrueAuto
    set SKIP_APPROVAL=-SkipPlanApproval
    shift
    goto parse_args
)
shift
goto parse_args
:done_parsing

REM Build the command
set RALPH_CMD=powershell -NoExit -Command "& {Set-Location 'D:\_Projects\voiceover-matcher-subtitle'; .\scripts\ralph\ralph.ps1 %TRUEAUTO% %SKIP_APPROVAL% %QUEUE%"
if defined FOCUS set RALPH_CMD=%RALPH_CMD% -FocusArea '%FOCUS%'
if defined TASK set RALPH_CMD=%RALPH_CMD% %TASK%
set RALPH_CMD=%RALPH_CMD%}"

echo   Starting Watch Dashboard...
start "Ralph Watch" powershell -NoExit -Command "& {Set-Location 'D:\_Projects\voiceover-matcher-subtitle'; .\scripts\ralph\watch.ps1}"

REM Give watch a moment to start
timeout /t 2 /nobreak > nul

echo   Starting Ralph Loop...
echo.
echo   Command: %RALPH_CMD%
echo.

start "Ralph Loop" %RALPH_CMD%

echo.
echo   =====================================================
echo   Both windows launched!
echo.
echo   Tips:
echo     - Watch window shows live progress
echo     - Ctrl+C in Ralph window to stop
echo     - Check scripts\ralph\logs\ for history
echo   =====================================================
echo.
echo   This window will close in 5 seconds...
timeout /t 5
