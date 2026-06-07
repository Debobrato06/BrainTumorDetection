@echo off
TITLE Brain Tumor Detection App - Stopper
echo ==========================================
echo    BRAIN TUMOR DETECTION APP STOPPER
echo ==========================================
echo.

setlocal enabledelayedexpansion
set SEARCH_PORT=5000
echo Searching for process on port %SEARCH_PORT%...

set PID_FOUND=0
for /f "tokens=5" %%a in ('netstat -aon ^| findstr LISTENING ^| findstr :%SEARCH_PORT%') do (
    set TARGET_PID=%%a
    if not "!TARGET_PID!"=="" (
        echo.
        echo [INFO] Found process with PID: !TARGET_PID!
        echo Killing process tree...
        taskkill /F /T /PID !TARGET_PID! >nul 2>&1
        set PID_FOUND=1
    )
)

if "%PID_FOUND%"=="0" (
    echo.
    echo [INFO] No process found running on port %SEARCH_PORT%.
) else (
    echo.
    echo [SUCCESS] Application processes on port %SEARCH_PORT% have been terminated.
)

echo.
pause
