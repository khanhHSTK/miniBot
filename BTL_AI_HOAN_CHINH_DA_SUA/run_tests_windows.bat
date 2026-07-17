@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [LOI] Chua co .venv. Hay chay start_windows.bat truoc.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pytest -q
pause
endlocal
