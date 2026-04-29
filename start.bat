@echo off
echo Starting Radiology AI System...
echo.

:: Start backend
echo [1/2] Starting FastAPI backend on http://localhost:8000
start "Backend" cmd /k "cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

:: Wait a moment
timeout /t 3 /nobreak >nul

:: Start frontend
echo [2/2] Starting Next.js frontend on http://localhost:3000
start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo System starting...
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API docs: http://localhost:8000/docs
echo.
pause
