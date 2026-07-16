@echo off
setlocal
cd /d "%~dp0" || (echo [LOI] Khong the chuyen den thu muc project. & pause & exit /b 1)
set PYTHONIOENCODING=utf-8

where py >nul 2>nul
if errorlevel 1 (
    echo [LOI] Khong tim thay Python Launcher ^(py^). Hay cai Python 3.11.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Dang tao moi truong ao Python 3.11...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo [LOI] Khong tao duoc .venv bang Python 3.11.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [LOI] Cai thu vien that bai.
    pause
    exit /b 1
)

start "AI House Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
timeout /t 6 /nobreak >nul
start "AI House Frontend" cmd /k "set API_URL=http://127.0.0.1:8000&& .venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501"

echo.
echo Backend: http://127.0.0.1:8000/docs
echo Frontend: http://127.0.0.1:8501
echo Giu hai cua so dich vu dang mo trong luc demo.
endlocal
