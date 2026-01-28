@echo off
REM ============================================================================
REM   "Sleep! That's where I'm a Viking!" - Ralph Wiggum
REM
REM   Unified Ralph Loop Launcher
REM   Combines all launcher functionality into one entry point
REM
REM   Usage:
REM     sleep-thats-where-im-a-viking.bat              (interactive)
REM     sleep-thats-where-im-a-viking.bat -Mode trueauto -FocusArea testing
REM     sleep-thats-where-im-a-viking.bat overnight    (shortcut: trueauto + testing)
REM ============================================================================

title Ralph Loop - Unified Launcher

cd /d "D:\_Projects\voiceover-matcher-subtitle"

REM Handle shortcuts
if /i "%~1"=="overnight" (
    powershell -ExecutionPolicy Bypass -File "scripts\ralph\launcher.ps1" -Mode trueauto -FocusArea testing
    goto :eof
)

if /i "%~1"=="yolo" (
    powershell -ExecutionPolicy Bypass -File "scripts\ralph\launcher.ps1" -Mode trueauto
    goto :eof
)

REM Pass all arguments to launcher.ps1
powershell -ExecutionPolicy Bypass -File "scripts\ralph\launcher.ps1" %*
