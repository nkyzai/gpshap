@echo off
setlocal
cd /d "H:\gapshap"
set "HF_HOME=H:\gapshap\models"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
if not exist ".venv\Scripts\activate.bat" (
  echo [ERROR] Virtual environment missing. Run install.bat first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
title gapshap local AI server
start "" "http://127.0.0.1:8787"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
pause
