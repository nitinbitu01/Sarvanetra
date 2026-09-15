@echo off
TITLE SARVANETRA — Permanent Live Hackathon Launcher (.dev)
COLOR 0A
CLS

echo ===============================================================================
echo            SARVANETRA — PERMANENT .DEV LIVE HACKATHON LAUNCHER
echo ===============================================================================
echo.
echo [*] Checking prerequisites...

:: 1. Verify Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH.
    pause
    exit /b 1
)

:: 2. Verify Ngrok
where ngrok >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] ngrok is not found in PATH.
    pause
    exit /b 1
)

:: Set Environment for 30-camera AI Fleet
set SENTINEL_FORCE_CLIPS=1
set SENTINEL_STRICT_LIVE=0
set SENTINEL_ANPR_CAMERAS=ALL
set SENTINEL_GLOBALID_CAMERAS=ALL
set SENTINEL_FLEET_WORKERS=8

:: Save Permanent Domain to File
if not exist "%~dp0output" mkdir "%~dp0output"
echo https://backer-thicket-denim.ngrok-free.dev > "%~dp0output\live_hackathon_url.txt"

echo [*] Starting SARVANETRA FastAPI Backend on port 8000...
start "SARVANETRA - Backend (Port 8000)" cmd /k "python -m uvicorn backend.main:app --port 8000 --host 127.0.0.1 --reload"

echo [*] Starting SARVANETRA AI Video Analytics Fleet Supervisor...
start "SARVANETRA - AI Video Analytics Fleet" cmd /k "python -m backend.scripts.fleet_supervisor"

echo [*] Starting SARVANETRA Frontend UI on port 5173 with Tunnel Support...
cd /d "%~dp0frontend"
start "SARVANETRA - Frontend UI (Port 5173)" cmd /k "npm run dev -- --host"

cd /d "%~dp0"
echo.
echo [*] Launching Permanent .DEV Tunnel: https://backer-thicket-denim.ngrok-free.dev ...
echo.
start "SARVANETRA - Permanent .dev Live Website" cmd /k "ngrok http --url=backer-thicket-denim.ngrok-free.dev 5173"

echo ===============================================================================
echo [SUCCESS] ALL SERVICES STARTED! YOUR PERMANENT WEBSITE IS LIVE!
echo.
echo  YOUR PERMANENT .DEV WEBSITE URL (NEVER CHANGES):
echo  ---------------------------------------------------------
echo    https://backer-thicket-denim.ngrok-free.dev
echo  ---------------------------------------------------------
echo.
echo  SUBMIT THIS EXACT URL TO YOUR HACKATHON PORTAL!
echo.
echo  HACKATHON JUDGE LOGIN CREDENTIALS:
echo    Username:  admin
echo    Password:  admin
echo.
echo  LOCAL MIRROR:
echo    Frontend UI:  http://localhost:5173
echo    API Docs:     http://127.0.0.1:8000/docs
echo ===============================================================================
echo.
pause
