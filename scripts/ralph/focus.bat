@echo off
REM ============================================================================
REM   Ralph Focus Mode - Full auto on a single area
REM   Usage: focus.bat [area]
REM ============================================================================

cd /d "D:\_Projects\voiceover-matcher-subtitle"

if not "%~1"=="" (
    set "AREA=%~1"
    goto :run
)

:menu
cls
echo.
echo   =====================================================
echo      Ralph Focus Mode - Full Auto
echo   =====================================================
echo.
echo   Choose a focus area:
echo.
echo     [1] pipeline        - Core pipeline improvements
echo     [2] testing         - Add/fix tests
echo     [3] speed           - Performance optimization
echo     [4] quality         - Code quality, linting, docs
echo     [5] rate-limiting   - API rate limit handling
echo     [6] otio            - Timeline output fixes
echo     [7] caption         - Caption/transcription
echo     [8] config          - Configuration system
echo     [9] client-learning - Cross-project learning
echo     [A] agents          - Self-healing agents
echo     [B] compilation     - Python compilation/syntax
echo.
echo     [Q] Quit
echo.
set /p CHOICE="   Enter choice: "

if /i "%CHOICE%"=="1" set "AREA=pipeline" & goto :run
if /i "%CHOICE%"=="2" set "AREA=testing" & goto :run
if /i "%CHOICE%"=="3" set "AREA=speed" & goto :run
if /i "%CHOICE%"=="4" set "AREA=quality" & goto :run
if /i "%CHOICE%"=="5" set "AREA=rate-limiting" & goto :run
if /i "%CHOICE%"=="6" set "AREA=otio" & goto :run
if /i "%CHOICE%"=="7" set "AREA=caption" & goto :run
if /i "%CHOICE%"=="8" set "AREA=config" & goto :run
if /i "%CHOICE%"=="9" set "AREA=client-learning" & goto :run
if /i "%CHOICE%"=="A" set "AREA=agents" & goto :run
if /i "%CHOICE%"=="B" set "AREA=compilation" & goto :run
if /i "%CHOICE%"=="Q" exit /b 0

echo   Invalid choice, try again...
timeout /t 2 >nul
goto :menu

:run
echo.
echo   Starting Ralph Loop - Full Auto Mode
echo   Focus Area: %AREA%
echo.
echo   Clearing existing PRD and queue...

REM Clear PRD
echo {"branchName":null,"sprintNumber":null,"focusArea":null,"projectContext":null,"userStories":[]} > "%~dp0prd.json"

REM Clear queue
echo {"focusAreas":[],"sessionId":null,"createdAt":null,"interviewContext":null,"interviewDetails":null} > "%~dp0queue.json"

echo   Done. Starting fresh on %AREA%...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0ralph.ps1" -FocusArea "%AREA%" -TrueAuto
