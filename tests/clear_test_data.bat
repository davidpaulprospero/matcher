@echo off
REM ============================================================
REM Clear Test Data
REM Removes cached test videos so fresh ones are downloaded
REM ============================================================

cd /d "%~dp0"

echo.
echo   Clearing test data...
echo.

if exist "test_data" (
    rmdir /s /q "test_data"
    echo   ✓ Cleared test_data folder
) else (
    echo   No test_data folder found
)

echo.
echo   Done! Run tests again to download fresh videos.
echo.
pause
