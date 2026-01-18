@echo off
REM Start Ollama with custom models directory
REM Models are stored in D:\ollama\models

set OLLAMA_MODELS=D:\ollama\models

echo Starting Ollama with models from: %OLLAMA_MODELS%
echo.

REM Check if already running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Ollama is already running. Killing existing instance...
    taskkill /IM "ollama app.exe" /F 2>NUL
    taskkill /IM "ollama.exe" /F 2>NUL
    timeout /t 2 /nobreak >NUL
)

REM Start ollama serve in background
start /B "" ollama serve

REM Wait for server to start
timeout /t 3 /nobreak >NUL

REM List available models
echo.
echo Available models:
ollama list

echo.
echo Ollama is ready for script generation.
echo Recommended: ollama pull mistral:7b
