@echo off
REM Start the PO Token server for YouTube subtitle/video access
REM This server must be running when using the pipeline with caption-first mode

echo Starting PO Token Server...
echo Server will run on http://127.0.0.1:4416
echo.
echo Press Ctrl+C to stop the server
echo.

node "%USERPROFILE%\bgutil-ytdlp-pot-provider\server\build\main.js"
