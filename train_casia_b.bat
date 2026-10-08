@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Install the environment with: powershell -File setup.ps1
    exit /b 1
)
".venv\Scripts\python.exe" run.py train --config configs/safegait_er/casia_b.yaml %*
exit /b %ERRORLEVEL%
