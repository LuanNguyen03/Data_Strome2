@echo off
REM ============================================
REM Setup All Dependencies - Dry Eye Assessment
REM ============================================
echo.
echo ================================================
echo   Dry Eye Assessment - Complete Setup
echo ================================================
echo.

REM Check Python
echo [STEP 1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo.

REM Check Node.js
echo [STEP 2/5] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found! Please install Node.js 18+ first.
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
)
node --version
npm --version
echo.

REM Install Backend Dependencies
echo [STEP 3/5] Installing Backend Dependencies...
echo ------------------------------------------------

REM Check if UV is available
uv --version >nul 2>&1
if errorlevel 1 (
    echo UV not found, using pip instead...
    pip install -r requirements.txt
) else (
    echo Using UV (faster)...
    uv sync
)

if errorlevel 1 (
    echo [ERROR] Failed to install backend dependencies!
    pause
    exit /b 1
)
echo [SUCCESS] Backend dependencies installed!
echo.

REM Install Frontend Dependencies
echo [STEP 4/5] Installing Frontend Dependencies...
echo ------------------------------------------------
cd frontend
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install frontend dependencies!
    pause
    exit /b 1
)
cd ..
echo [SUCCESS] Frontend dependencies installed!
echo.

REM Setup Gemini AI (Optional)
echo [STEP 5/5] Gemini AI Setup (Optional)...
echo ------------------------------------------------
set /p setup_gemini="Do you want to setup Gemini AI now? (y/n): "
if /i "%setup_gemini%"=="y" (
    echo.
    call setup_gemini.bat
) else (
    echo Skipping Gemini AI setup. You can run 'setup_gemini.bat' later.
)
echo.

REM Summary
echo ================================================
echo   SETUP COMPLETE!
echo ================================================
echo.
echo [INSTALLED]
echo   - Backend dependencies (FastAPI, ML libraries, etc.)
echo   - Frontend dependencies (React, Vite, etc.)
if /i "%setup_gemini%"=="y" (
    echo   - Gemini AI configured
) else (
    echo   - Gemini AI: Not configured (optional)
)
echo.
echo [NEXT STEPS]
echo   1. Start backend:  uv run python backend/run.py
echo      or:             python backend/run.py
echo.
echo   2. Start frontend (in new terminal):
echo      cd frontend
echo      npm run dev
echo.
echo   3. Open browser: http://localhost:5173
echo.
echo [OPTIONAL]
if /i not "%setup_gemini%"=="y" (
    echo   - Setup Gemini AI: setup_gemini.bat
)
echo   - Check Gemini:    python check_gemini.py
echo   - View help:       make help  (if you have make)
echo.
echo ================================================
pause
