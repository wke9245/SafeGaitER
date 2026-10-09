@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set "CHECKPOINT_PYTHON=.venv\Scripts\python.exe"
) else (
    set "CHECKPOINT_PYTHON=python"
)
"%CHECKPOINT_PYTHON%" -X utf8 -m scripts.inspect_checkpoint --device cuda %*
set "CHECKPOINT_EXIT=%ERRORLEVEL%"
if not "%CHECKPOINT_EXIT%"=="0" (
    echo Checkpoint inspection failed. Read the error above.
    echo Install the CUDA environment with powershell -NoProfile -File setup.ps1
)
pause
exit /b %CHECKPOINT_EXIT%
