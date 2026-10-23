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
echo [1/8] Installing Python 3.12...
echo ------------------------------------------------------------
set "PYTHON_EXE=%PROJECT_ROOT%\python312\python.exe"
curl.exe --fail --location --silent --show-error --output "%TEMP%\gapshap-python-3.12.10-amd64.exe" https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
if errorlevel 1 (
    echo [ERROR] Could not download the official Python installer.
    pause
    exit /b 1
)
"%TEMP%\gapshap-python-3.12.10-amd64.exe" /passive /log "%TEMP%\gapshap-python-install.log" InstallAllUsers=0 TargetDir="%PROJECT_ROOT%\python312" DefaultJustForMeTargetDir="%PROJECT_ROOT%\python312" Include_exe=1 Include_lib=1 Include_pip=1 Include_test=0 Include_launcher=0 PrependPath=0 Shortcuts=0
set "PYTHON_INSTALL_EXIT=%ERRORLEVEL%"
if not "%PYTHON_INSTALL_EXIT%"=="0" (
    echo [ERROR] Python 3.12.10 installation failed with exit code %PYTHON_INSTALL_EXIT%.
    echo Installer log: %TEMP%\gapshap-python-install.log
    del /q "%TEMP%\gapshap-python-3.12.10-amd64.exe" >nul 2>nul
    pause
    exit /b 1
)
del /q "%TEMP%\gapshap-python-3.12.10-amd64.exe" >nul 2>nul
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE="
    for /r "%PROJECT_ROOT%\python312" %%P in (python.exe) do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%~fP"
    )
)
if not defined PYTHON_EXE (
    echo [ERROR] Python installer returned success, but python.exe was not found under:
    echo %PROJECT_ROOT%\python312
    echo Installer log: %TEMP%\gapshap-python-install.log
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import sys; print('Using Python', sys.version); raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize.bit_length() == 63 else 1)"
if errorlevel 1 (
    echo [ERROR] The selected interpreter must be 64-bit Python 3.12.
    pause
    exit /b 1
)

echo ------------------------------------------------------------
echo [2/8] Creating Python 3.12 virtual environment...
echo ------------------------------------------------------------
"%PYTHON_EXE%" -m venv --clear "%PROJECT_ROOT%\.venv312"
if errorlevel 1 (
    echo [ERROR] Python 3.12 virtual environment creation failed.
    pause
    exit /b 1
)
set "VENV_PYTHON=%PROJECT_ROOT%\.venv312\Scripts\python.exe"
"%VENV_PYTHON%" -c "import sys; print('Environment Python', sys.version); raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize.bit_length() == 63 else 1)"
if errorlevel 1 (
    echo [ERROR] .venv312 is not running 64-bit Python 3.12. Re-run install.bat after checking the environment.
    pause
    exit /b 1
)
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Could not prepare pip in the Python 3.12 environment.
    pause
    exit /b 1
)

echo ------------------------------------------------------------
echo [3/8] Creating project folders...
echo ------------------------------------------------------------
for %%D in (app static data uploads indexes scripts models "data\generated") do (
    if not exist "%PROJECT_ROOT%\%%~D" mkdir "%PROJECT_ROOT%\%%~D"
)
if not exist "%PROJECT_ROOT%\app\__init__.py" type nul > "%PROJECT_ROOT%\app\__init__.py"

echo ------------------------------------------------------------
echo [4/8] Installing PyTorch...
echo ------------------------------------------------------------
set "GPU_AVAILABLE=0"
if "%FORCE_CPU%"=="0" (
    where nvidia-smi >nul 2>nul && set "GPU_AVAILABLE=1"
)
set "TORCH_CUDA_INSTALL=0"

if "%GPU_AVAILABLE%"=="1" (
    echo NVIDIA GPU detected - installing PyTorch CUDA 12.8 wheels for Python 3.12...
    "%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo CUDA wheel install failed - falling back to PyTorch CPU wheels...
        "%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        if errorlevel 1 (
            echo [ERROR] PyTorch could not be installed for Python 3.12.
            pause
            exit /b 1
        )
    ) else (
        set "TORCH_CUDA_INSTALL=1"
    )
) else (
    set "TORCH_CUDA_INSTALL=0"
    echo No NVIDIA GPU ^(or --cpu^) - installing CPU build...
    "%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo [ERROR] PyTorch could not be installed for Python 3.12.
        pause
        exit /b 1
    )
)
if "%TORCH_CUDA_INSTALL%"=="1" (
    "%VENV_PYTHON%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
    if errorlevel 1 (
        echo PyTorch CUDA cannot access the GPU or driver. Reinstalling CPU PyTorch...
        "%VENV_PYTHON%" -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        if errorlevel 1 (
            echo [ERROR] Could not install the PyTorch CPU fallback.
            pause
            exit /b 1
        )
        set "TORCH_CUDA_INSTALL=0"
    )
)
"%VENV_PYTHON%" -c "import torch, sys; print('PyTorch', torch.__version__, 'on Python', sys.version.split()[0]); raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize.bit_length() == 63 else 1)"
if errorlevel 1 (
    echo [ERROR] PyTorch verification failed in the Python 3.12 environment.
    pause
    exit /b 1
)

echo ------------------------------------------------------------
echo [5/8] Installing gapshap Python dependencies + Llama.cpp...
echo ------------------------------------------------------------

REM Install dependencies using a prebuilt CPU llama.cpp wheel; never compile it with CMake here.
"%VENV_PYTHON%" -m pip install --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r "%PROJECT_ROOT%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Dependencies failed to install. Confirm Python 3.12 x64 and internet access.
    echo If the error mentions llama-cpp-python, check that its prebuilt wheel index is reachable.
    pause
    exit /b 1
)

if "%GPU_AVAILABLE%"=="1" (
    echo Installing prebuilt llama.cpp CUDA 12.4 wheel...
    "%VENV_PYTHON%" -m pip install --upgrade --no-deps --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 "llama-cpp-python>=0.3,<0.4"
    if errorlevel 1 (
        echo [WARNING] CUDA wheel unavailable. Restoring the prebuilt CPU llama.cpp wheel...
        "%VENV_PYTHON%" -m pip install --force-reinstall --no-deps --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu "llama-cpp-python>=0.3,<0.4"
        if errorlevel 1 (
            echo [ERROR] Could not install or restore the prebuilt CPU llama.cpp wheel.
            pause
            exit /b 1
        )
    ) else (
        "%VENV_PYTHON%" -c "import llama_cpp" >nul 2>nul
        if errorlevel 1 (
            echo [WARNING] CUDA llama.cpp wheel failed import. Restoring the CPU wheel...
            "%VENV_PYTHON%" -m pip install --force-reinstall --no-deps --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu "llama-cpp-python>=0.3,<0.4"
            if errorlevel 1 (
                echo [ERROR] Could not restore the prebuilt CPU llama.cpp wheel.
                pause
                exit /b 1
            )
        )
    )
)

"%VENV_PYTHON%" -c "import torch, llama_cpp; print('Verified torch', torch.__version__, 'and llama.cpp', llama_cpp.__version__)"
if errorlevel 1 (
    echo [ERROR] Required PyTorch or llama.cpp module failed verification.
    pause
    exit /b 1
)

echo Dependencies installed and verified.

echo ------------------------------------------------------------
echo [6/8] Writing model bootstrap script...
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
if errorlevel 1 (
    echo [ERROR] Could not create the model bootstrap script.
    pause
    exit /b 1
)

echo ------------------------------------------------------------
echo [7/8] Downloading models into %PROJECT_ROOT%\models ...
echo ------------------------------------------------------------
if "%DL_MODELS%"=="1" (
    set "GAPSHAP_DL_VISION=%DL_VISION%"
    set "GAPSHAP_DL_IMAGEGEN=%DL_IMAGEGEN%"
    echo This may take a while. You can also download models later from the
    echo "Models" tab inside the web UI ^(recommended - shows sizes first^).
    "%VENV_PYTHON%" "%PROJECT_ROOT%\scripts\download_models.py"
    if errorlevel 1 (
        echo [ERROR] Model bootstrap script failed.
        pause
        exit /b 1
    )
) else (
    echo Skipped model download ^(--no-models^). Use the Models tab in the UI.
)

echo ------------------------------------------------------------
echo [8/8] Creating run.bat launcher...
echo ------------------------------------------------------------
>  "%PROJECT_ROOT%\run.bat" echo @echo off
>> "%PROJECT_ROOT%\run.bat" echo setlocal
>> "%PROJECT_ROOT%\run.bat" echo cd /d "%%~dp0"
>> "%PROJECT_ROOT%\run.bat" echo set "HF_HOME=%%~dp0models"
>> "%PROJECT_ROOT%\run.bat" echo set "PYTHONUTF8=1"
>> "%PROJECT_ROOT%\run.bat" echo set "PYTHONUNBUFFERED=1"
>> "%PROJECT_ROOT%\run.bat" echo if not exist ".venv312\Scripts\python.exe" (
>> "%PROJECT_ROOT%\run.bat" echo   echo [ERROR] Python 3.12 environment missing. Run install.bat first.
>> "%PROJECT_ROOT%\run.bat" echo   pause
>> "%PROJECT_ROOT%\run.bat" echo   exit /b 1
>> "%PROJECT_ROOT%\run.bat" echo )
>> "%PROJECT_ROOT%\run.bat" echo call ".venv312\Scripts\activate.bat"
>> "%PROJECT_ROOT%\run.bat" echo python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize.bit_length() == 63 else 1)"
>> "%PROJECT_ROOT%\run.bat" echo if errorlevel 1 ^(
>> "%PROJECT_ROOT%\run.bat" echo   echo [ERROR] The active environment is not Python 3.12. Run install.bat.
>> "%PROJECT_ROOT%\run.bat" echo   pause
>> "%PROJECT_ROOT%\run.bat" echo   exit /b 1
>> "%PROJECT_ROOT%\run.bat" echo ^)
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