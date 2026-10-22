@echo off
setlocal EnableExtensions DisableDelayedExpansion

REM ============================================================
REM  gapshap - Full Automated Installer (v2: Hub + Vision + Diffusion + SSE)
REM  Run ONCE. Afterwards use run.bat
REM  Options:  install.bat [--no-models] [--vision] [--imagegen] [--cpu]
REM ============================================================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "GAPSHAP_MODEL=Qwen/Qwen2.5-1.5B-Instruct"
set "GAPSHAP_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2"
set "GAPSHAP_VISION_MODEL=microsoft/Florence-2-base"
set "GAPSHAP_IMAGE_MODEL=stabilityai/sd-turbo"

set "DL_MODELS=1"
set "DL_VISION=0"
set "DL_IMAGEGEN=0"
set "FORCE_CPU=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--no-models" set "DL_MODELS=0"
if /I "%~1"=="--vision"    set "DL_VISION=1"
if /I "%~1"=="--imagegen"  set "DL_IMAGEGEN=1"
if /I "%~1"=="--all"       ( set "DL_VISION=1" & set "DL_IMAGEGEN=1" )
if /I "%~1"=="--cpu"       set "FORCE_CPU=1"
shift
goto parse_args
:args_done

REM All model weights live INSIDE the project folder
set "HF_HOME=%PROJECT_ROOT%\models"
set "HUGGINGFACE_HUB_CACHE=%PROJECT_ROOT%\models\hub"
set "HF_HUB_ENABLE_HF_TRANSFER=0"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "TRANSFORMERS_NO_ADVISORY_WARNINGS=1"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

echo ============================================================
echo  gapshap Installer v2
echo ============================================================
echo Project root : %PROJECT_ROOT%
echo Model store  : %HF_HOME%
echo.

if not exist "%PROJECT_ROOT%" mkdir "%PROJECT_ROOT%"
cd /d "%PROJECT_ROOT%"

echo ------------------------------------------------------------
echo [1/9] Checking Python...
echo ------------------------------------------------------------
set "PYTHON_FOUND="
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 --version >nul 2>nul
    if %errorlevel%==0 set "PYTHON_FOUND=py"
)

if not defined PYTHON_FOUND (
    echo Python not found. Installing Python 3.12 via winget...
    winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERROR] Could not install Python automatically.
        echo Install Python 3.10+ manually, then run install.bat again.
        pause
        exit /b 1
    )
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
    where py >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python installed but not visible to this shell.
        echo Close this window and run install.bat again.
        pause
        exit /b 1
    )
    set "PYTHON_FOUND=py"
)
py -3 --version

echo ------------------------------------------------------------
echo [2/9] Creating virtual environment...
echo ------------------------------------------------------------
if not exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    py -3 -m venv "%PROJECT_ROOT%\.venv"
    if errorlevel 1 (
        echo [ERROR] venv creation failed.
        pause
        exit /b 1
    )
)
call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel

echo ------------------------------------------------------------
echo [3/9] Creating project folders...
echo ------------------------------------------------------------
for %%D in (app static data uploads indexes scripts models "data\generated") do (
    if not exist "%PROJECT_ROOT%\%%~D" mkdir "%PROJECT_ROOT%\%%~D"
)
if not exist "%PROJECT_ROOT%\app\__init__.py" type nul > "%PROJECT_ROOT%\app\__init__.py"

echo ------------------------------------------------------------
echo [4/9] Writing requirements.txt...
echo ------------------------------------------------------------
>  "%PROJECT_ROOT%\requirements.txt" echo fastapi
>> "%PROJECT_ROOT%\requirements.txt" echo uvicorn[standard]
>> "%PROJECT_ROOT%\requirements.txt" echo transformers
>> "%PROJECT_ROOT%\requirements.txt" echo accelerate
>> "%PROJECT_ROOT%\requirements.txt" echo safetensors
>> "%PROJECT_ROOT%\requirements.txt" echo sentence-transformers
>> "%PROJECT_ROOT%\requirements.txt" echo faiss-cpu
>> "%PROJECT_ROOT%\requirements.txt" echo numpy
>> "%PROJECT_ROOT%\requirements.txt" echo tqdm
>> "%PROJECT_ROOT%\requirements.txt" echo diffusers
>> "%PROJECT_ROOT%\requirements.txt" echo ftfy
>> "%PROJECT_ROOT%\requirements.txt" echo einops
>> "%PROJECT_ROOT%\requirements.txt" echo timm
>> "%PROJECT_ROOT%\requirements.txt" echo pypdf
>> "%PROJECT_ROOT%\requirements.txt" echo python-docx
>> "%PROJECT_ROOT%\requirements.txt" echo pillow
>> "%PROJECT_ROOT%\requirements.txt" echo requests
>> "%PROJECT_ROOT%\requirements.txt" echo beautifulsoup4
>> "%PROJECT_ROOT%\requirements.txt" echo python-multipart
>> "%PROJECT_ROOT%\requirements.txt" echo httpx
>> "%PROJECT_ROOT%\requirements.txt" echo huggingface_hub
>> "%PROJECT_ROOT%\requirements.txt" echo psutil

echo ------------------------------------------------------------
echo [5/9] Installing PyTorch...
echo ------------------------------------------------------------
set "GPU_AVAILABLE=0"
if "%FORCE_CPU%"=="0" (
    where nvidia-smi >nul 2>nul && set "GPU_AVAILABLE=1"
)

if "%GPU_AVAILABLE%"=="1" (
    echo NVIDIA GPU detected - installing CUDA build...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo CUDA build failed - falling back to CPU build...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
) else (
    echo No NVIDIA GPU ^(or --cpu^) - installing CPU build...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo ------------------------------------------------------------
echo [6/8] Installing gapshap Python dependencies + Llama.cpp (CUDA)...
echo ------------------------------------------------------------

REM Install standard deps first
python -m pip install -r "%PROJECT_ROOT%\requirements.txt"

REM Install llama-cpp-python with CUDA support
REM This compiles it against your GPU. It takes 5-10 minutes on first run.
echo Installing llama-cpp-python with CUDA support (this may take 5-10 mins)...
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
python -m pip install llama-cpp-python --force-reinstall --no-cache-dir --verbose

if errorlevel 1 (
    echo [WARNING] CUDA compilation failed. Falling back to CPU-only llama-cpp...
    python -m pip install llama-cpp-python --force-reinstall --no-cache-dir
)

echo Dependencies installed.

echo ------------------------------------------------------------
echo [7/9] Writing model bootstrap script...
echo ------------------------------------------------------------
>  "%PROJECT_ROOT%\scripts\download_models.py" echo import os
>> "%PROJECT_ROOT%\scripts\download_models.py" echo from huggingface_hub import snapshot_download
>> "%PROJECT_ROOT%\scripts\download_models.py" echo targets = []
>> "%PROJECT_ROOT%\scripts\download_models.py" echo targets.append(os.environ["GAPSHAP_MODEL"])
>> "%PROJECT_ROOT%\scripts\download_models.py" echo targets.append(os.environ["GAPSHAP_EMBED_MODEL"])
>> "%PROJECT_ROOT%\scripts\download_models.py" echo if os.environ.get("GAPSHAP_DL_VISION") == "1":
>> "%PROJECT_ROOT%\scripts\download_models.py" echo     targets.append(os.environ["GAPSHAP_VISION_MODEL"])
>> "%PROJECT_ROOT%\scripts\download_models.py" echo if os.environ.get("GAPSHAP_DL_IMAGEGEN") == "1":
>> "%PROJECT_ROOT%\scripts\download_models.py" echo     targets.append(os.environ["GAPSHAP_IMAGE_MODEL"])
>> "%PROJECT_ROOT%\scripts\download_models.py" echo for repo in targets:
>> "%PROJECT_ROOT%\scripts\download_models.py" echo     print("Downloading:", repo, flush=True)
>> "%PROJECT_ROOT%\scripts\download_models.py" echo     try:
>> "%PROJECT_ROOT%\scripts\download_models.py" echo         snapshot_download(repo_id=repo, ignore_patterns=["*.msgpack","*.h5","*.ot","*.onnx","*.tflite","original/*"])
>> "%PROJECT_ROOT%\scripts\download_models.py" echo         print("  OK:", repo, flush=True)
>> "%PROJECT_ROOT%\scripts\download_models.py" echo     except Exception as exc:
>> "%PROJECT_ROOT%\scripts\download_models.py" echo         print("  FAILED:", repo, exc, flush=True)
>> "%PROJECT_ROOT%\scripts\download_models.py" echo print("Bootstrap complete.", flush=True)

echo ------------------------------------------------------------
echo [8/9] Downloading models into %PROJECT_ROOT%\models ...
echo ------------------------------------------------------------
if "%DL_MODELS%"=="1" (
    set "GAPSHAP_DL_VISION=%DL_VISION%"
    set "GAPSHAP_DL_IMAGEGEN=%DL_IMAGEGEN%"
    echo This may take a while. You can also download models later from the
    echo "Models" tab inside the web UI ^(recommended - shows sizes first^).
    python "%PROJECT_ROOT%\scripts\download_models.py"
) else (
    echo Skipped model download ^(--no-models^). Use the Models tab in the UI.
)

echo ------------------------------------------------------------
echo [9/9] Creating run.bat launcher...
echo ------------------------------------------------------------
>  "%PROJECT_ROOT%\run.bat" echo @echo off
>> "%PROJECT_ROOT%\run.bat" echo setlocal
>> "%PROJECT_ROOT%\run.bat" echo cd /d "%PROJECT_ROOT%"
>> "%PROJECT_ROOT%\run.bat" echo set "HF_HOME=%PROJECT_ROOT%\models"
>> "%PROJECT_ROOT%\run.bat" echo set "PYTHONUTF8=1"
>> "%PROJECT_ROOT%\run.bat" echo set "PYTHONUNBUFFERED=1"
>> "%PROJECT_ROOT%\run.bat" echo if not exist ".venv\Scripts\activate.bat" (
>> "%PROJECT_ROOT%\run.bat" echo   echo [ERROR] Virtual environment missing. Run install.bat first.
>> "%PROJECT_ROOT%\run.bat" echo   pause
>> "%PROJECT_ROOT%\run.bat" echo   exit /b 1
>> "%PROJECT_ROOT%\run.bat" echo )
>> "%PROJECT_ROOT%\run.bat" echo call ".venv\Scripts\activate.bat"
>> "%PROJECT_ROOT%\run.bat" echo title gapshap local AI server
>> "%PROJECT_ROOT%\run.bat" echo start "" "http://127.0.0.1:8787"
>> "%PROJECT_ROOT%\run.bat" echo python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
>> "%PROJECT_ROOT%\run.bat" echo pause

echo.
if not exist "%PROJECT_ROOT%\app\main.py" (
    echo [WARNING] app\main.py not found.
    echo Save the gapshap source files into %PROJECT_ROOT% before running run.bat.
) else (
    echo [OK] app\main.py detected.
)

echo.
echo ============================================================
echo  gapshap installation complete
echo ============================================================
echo  Start with : %PROJECT_ROOT%\run.bat
echo  Open       : http://127.0.0.1:8787
echo  Models dir : %PROJECT_ROOT%\models
echo ============================================================
pause