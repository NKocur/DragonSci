@echo off
set "PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%"
cd /d "%~dp0"
C:\msys64\usr\bin\bash.exe run.sh
if errorlevel 1 ( echo. & echo Launch failed. & pause )
