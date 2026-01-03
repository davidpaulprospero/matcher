@echo off
REM Voiceover-Matcher Runner
REM Runs main.py and closes automatically when done

cd /d "%~dp0"
python main.py %*
exit
