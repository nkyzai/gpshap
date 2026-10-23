@echo off
setlocal
cd /d "%~dp0"
set "HF_HOME=%~dp0models"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
if not exist ".venv312\Scripts\python.exe" (
  echo [ERROR] Python 3.12 environment missing. Run install.bat first.
  pause
  exit /b 1
)
call ".venv312\Scripts\activate.bat"
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize.bit_length() == 63 else 1)"
if errorlevel 1 (
  echo [ERROR] The active environment is not Python 3.12. Run install.bat.
  pause
  exit /b 1
)
title gapshap local AI server
start "" "http://127.0.0.1:8787"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
pause
