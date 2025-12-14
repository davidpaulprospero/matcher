@echo off
REM Voiceover-Matcher Installation Helper
REM Run this from the folder where you extracted the zip

echo.
echo ========================================
echo  VOICEOVER-MATCHER INSTALLATION
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo [1/4] Installing Python dependencies...
pip install -r requirements.txt --break-system-packages 2>nul || pip install -r requirements.txt

echo.
echo [2/4] Checking .env file...
if not exist ".env" (
    echo Creating .env template...
    (
        echo # Voiceover-Matcher API Keys
        echo # Fill in your API keys below
        echo.
        echo # Required: At least one embedding provider
        echo GEMINI_API_KEY=your_gemini_key_here
        echo # VOYAGE_API_KEY=your_voyage_key_here
        echo.
        echo # Required: LLM for keyword extraction and matching
        echo ANTHROPIC_API_KEY=your_anthropic_key_here
        echo.
        echo # Optional: For video downloading
        echo # YOUTUBE_API_KEY=your_youtube_key_here
    ) > .env
    echo.
    echo IMPORTANT: Edit .env and add your API keys!
    echo   - GEMINI_API_KEY (for embeddings)
    echo   - ANTHROPIC_API_KEY (for LLM matching)
    echo.
) else (
    echo .env file already exists
)

echo [3/4] Testing installation...
python -c "from src.config import Config; print('  Config module OK')"
python -c "import faster_whisper; print('  Faster-whisper OK')" 2>nul || echo   Faster-whisper not installed (optional)
python -c "import torch; print(f'  PyTorch OK (CUDA: {torch.cuda.is_available()})')"

echo.
echo [4/4] Installation complete!
echo.
echo ========================================
echo  NEXT STEPS
echo ========================================
echo.
echo 1. Edit .env and add your API keys
echo.
echo 2. Create a new project:
echo    python setup_project.py "E:\Your\Project\Path"
echo.
echo 3. Or run directly:
echo    python main.py -v voiceover.srt
echo.
echo ========================================
pause
