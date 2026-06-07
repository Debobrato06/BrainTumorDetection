@echo off
TITLE Brain Tumor Detection App - Starter
echo ==========================================
echo    BRAIN TUMOR DETECTION APP STARTER
echo ==========================================
echo.

:: Check for Flask (and other core dependencies)
python -c "import flask" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Dependencies not found. Installing from requirements.txt...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [INFO] requirements.txt failed, installing core packages manually...
        pip install flask torch numpy matplotlib Pillow torchvision scikit-image
    )
)

echo.
echo Starting Flask server on http://localhost:5000...
echo Close this window to stop the server (or use stop.bat).
echo.

python app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application failed to start.
    echo Please check if Python is in your PATH and dependencies are installed.
    pause
)
