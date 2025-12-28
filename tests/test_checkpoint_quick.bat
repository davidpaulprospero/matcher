@echo off
REM ============================================================
REM QUICK VALIDATION: Checkpoint & Saved Keywords
REM ============================================================
REM Run this first to make sure the code works before testing
REM ============================================================

echo.
echo ============================================================
echo   QUICK VALIDATION: Checkpoint ^& Saved Keywords
echo ============================================================
echo.

cd /d "%~dp0.."
echo   Install dir: %CD%

REM ─────────────────────────────────────────────────────────────
REM Check 1: Module imports
REM ─────────────────────────────────────────────────────────────
echo.
echo [1/5] Checking module imports...

python -c "from src.checkpoint import CheckpointManager, KeywordManager, SavedKeywords, STAGE_ORDER; print('      OK: checkpoint module imports')"
if %ERRORLEVEL% NEQ 0 (
    echo      FAIL: Cannot import checkpoint module
    goto :error
)

REM ─────────────────────────────────────────────────────────────
REM Check 2: Main.py imports
REM ─────────────────────────────────────────────────────────────
echo [2/5] Checking main.py imports...

python -c "import main; print('      OK: main.py imports')"
if %ERRORLEVEL% NEQ 0 (
    echo      FAIL: Cannot import main.py
    goto :error
)

REM ─────────────────────────────────────────────────────────────
REM Check 3: New arguments exist
REM ─────────────────────────────────────────────────────────────
echo [3/5] Checking new command line arguments...

python main.py --help 2>&1 | findstr /C:"--resume" >nul
if %ERRORLEVEL% NEQ 0 (
    echo      FAIL: --resume argument not found
    goto :error
)
echo       OK: --resume

python main.py --help 2>&1 | findstr /C:"--fresh" >nul
if %ERRORLEVEL% NEQ 0 (
    echo      FAIL: --fresh argument not found
    goto :error
)
echo       OK: --fresh

python main.py --help 2>&1 | findstr /C:"--use-keywords" >nul
if %ERRORLEVEL% NEQ 0 (
    echo      FAIL: --use-keywords argument not found
    goto :error
)
echo       OK: --use-keywords

python main.py --help 2>&1 | findstr /C:"--save-keywords" >nul
if %ERRORLEVEL% NEQ 0 (
    echo      FAIL: --save-keywords argument not found
    goto :error
)
echo       OK: --save-keywords

python main.py --help 2>&1 | findstr /C:"--list-keywords" >nul
if %ERRORLEVEL% NEQ 0 (
    echo      FAIL: --list-keywords argument not found
    goto :error
)
echo       OK: --list-keywords

REM ─────────────────────────────────────────────────────────────
REM Check 4: CheckpointManager basic test
REM ─────────────────────────────────────────────────────────────
echo [4/5] Testing CheckpointManager...

python -c "import tempfile; from src.checkpoint import CheckpointManager; td=tempfile.mkdtemp(); cm=CheckpointManager(td,'test'); cm.save('ANALYZE',{'keywords':['a','b']}); assert cm.exists(); cm2=CheckpointManager(td); cm2.load(); assert cm2.should_skip_stage('ANALYZE'); assert not cm2.should_skip_stage('DOWNLOAD'); cm2.clear(); assert not cm2.exists(); print('      OK: CheckpointManager works')"
if %ERRORLEVEL% NEQ 0 (
    echo      FAIL: CheckpointManager test failed
    goto :error
)

REM ─────────────────────────────────────────────────────────────
REM Check 5: KeywordManager basic test
REM ─────────────────────────────────────────────────────────────
echo [5/5] Testing KeywordManager...

python -c "import tempfile; from src.checkpoint import KeywordManager; td=tempfile.mkdtemp(); km=KeywordManager(td); km.save_keywords(['kw1','kw2'],topic_context='test',name='test'); assert km.has_presets(); p=km.get_preset('test'); assert p.keywords==['kw1','kw2']; print('      OK: KeywordManager works')"
if %ERRORLEVEL% NEQ 0 (
    echo      FAIL: KeywordManager test failed
    goto :error
)

REM ─────────────────────────────────────────────────────────────
REM Success
REM ─────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   All CHECKS PASSED
echo ============================================================
echo.
echo   The checkpoint and saved keywords features are working.
echo.
echo   Next steps:
echo     1. Run: python tests\test_checkpoint.py  (full unit tests)
echo     2. Or test manually with a real project
echo.
goto :end

:error
echo.
echo ============================================================
echo   VALIDATION FAILED
echo ============================================================
echo.
echo   Please check the error above and fix before proceeding.
echo.
exit /b 1

:end
pause
