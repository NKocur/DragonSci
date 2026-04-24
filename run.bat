@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run start.bat first to create it and build dragonsci.
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python demo.py
if errorlevel 1 (
    echo.
    echo Launch failed.
    pause
    exit /b 1
)

endlocal
