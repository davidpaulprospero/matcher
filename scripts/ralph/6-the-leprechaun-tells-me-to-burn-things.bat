@echo off
REM ================================================
REM   "The leprechaun tells me to burn things"
REM          - Ralph Wiggum
REM
REM   Watch Ralph Loop progress (he's always watching)
REM ================================================

cd /d "%~dp0..\.."
powershell -ExecutionPolicy Bypass -File "%~dp0watch.ps1" %*
