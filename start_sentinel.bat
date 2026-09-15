@echo off
TITLE Sentinel Gujarat — 24/7 AI Video Surveillance System
COLOR 0A
CLS

echo ===============================================================================
echo                SENTINEL GUJARAT — 24/7 AI VIDEO SURVEILLANCE
echo          Gujarat Police Command, Control and Emergency Response Centre
echo ===============================================================================
echo.

:: 1. Verify Python Environment
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    pause
    exit /b 1
)

:: Set Environment Configuration for Full 30-Camera Live AI Fleet
set SENTINEL_FORCE_CLIPS=1
set SENTINEL_STRICT_LIVE=0
set SENTINEL_ANPR_CAMERAS=ALL
set SENTINEL_GLOBALID_CAMERAS=ALL
set SENTINEL_FLEET_WORKERS=8

:: 2. Start FastAPI Backend Server
echo [*] Starting Sentinel Gujarat FastAPI Backend Server (Port 8000)...
start "Sentinel Gujarat - Backend Server" cmd /k "python -m uvicorn backend.main:app --port 8000 --host 127.0.0.1 --reload"

:: 3. Start Live 24/7 AI CCTV Surveillance Pipeline Multi-Process Fleet
echo [*] Starting Live 24/7 AI Multi-Process Fleet Supervisor (30 Live Cameras)...
start "Sentinel Gujarat - Multi-Process Fleet Supervisor" cmd /k "python -m backend.scripts.fleet_supervisor"

:: 4. Start Vite Frontend Server
echo [*] Starting Sentinel Gujarat Command Room Frontend (Port 5173)...
cd /d "%~dp0frontend"
start "Sentinel Gujarat - Control Room UI" cmd /k "npm run dev -- --host"

echo.
echo ===============================================================================
echo [SUCCESS] ALL 3 SERVICES STARTED IN PARALLEL!
echo  - Control Room UI:    http://localhost:5173
echo  - FastAPI Backend:    http://127.0.0.1:8000
echo  - API Swagger Docs:   http://127.0.0.1:8000/docs
echo  - 24/7 AI Pipeline:  Active on 30 CCTV Cameras
echo ===============================================================================
echo.
pause
