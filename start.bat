@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        py -3 -m venv .venv
    )
    if errorlevel 1 (
        echo.
        echo Could not create the virtual environment. Make sure Python is installed.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

python -c "import maturin" >nul 2>nul
if errorlevel 1 (
    echo Installing maturin...
    python -m pip install maturin
    if errorlevel 1 (
        echo.
        echo Could not install maturin.
        pause
        exit /b 1
    )
)

python -c "import numpy" >nul 2>nul
if errorlevel 1 (
    echo Installing numpy...
    python -m pip install numpy
    if errorlevel 1 (
        echo.
        echo Could not install numpy.
        pause
        exit /b 1
    )
)

for %%F in (python\dragonsci\_dragonsci*.pyd) do (
    if exist "%%~fF" (
        del /f /q "%%~fF" >nul 2>nul
        if exist "%%~fF" (
            echo.
            echo Could not replace %%~nxF because it is in use.
            echo Close any running DragonSci demo or Python windows, then run start.bat again.
            pause
            exit /b 1
        )
    )
)

echo Building dragonsci...
maturin develop --target x86_64-pc-windows-msvc
if errorlevel 1 (
    echo.
    echo Build failed.
    pause
    exit /b 1
)

echo.
echo Launching demo...
python demo.py
if errorlevel 1 (
    echo.
    echo Launch failed.
    pause
    exit /b 1
)

endlocal
