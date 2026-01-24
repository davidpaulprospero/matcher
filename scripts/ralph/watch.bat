@echo off
REM Ralph Loop Watcher - Monitor progress in real-time
REM "I'm watching you... always watching" - Ralph probably

cd /d "%~dp0..\.."
powershell -ExecutionPolicy Bypass -File "%~dp0watch.ps1" %*
