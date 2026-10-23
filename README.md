# Gpshap

Local Windows LLM server and UI.

## Structure

- `app/` FastAPI backend
- `static/` frontend UI
- `models/` local model cache, not committed
- `data/` local config/database, not committed
- `uploads/` user uploads, not committed
- `indexes/` RAG indexes, not committed

## Security

This repository intentionally ignores:

- virtual environments
- model weights
- local databases
- uploaded files
- Hugging Face tokens
- runtime config

Do not commit `.env`, `data/config.json`, or model files.

## Local privacy

Chat history is stored in the local SQLite database under `data/`; Gapshap does not sync, upload, or email chat transcripts. Chat-triggered web search and browser speech-to-text are disabled so chat prompts and voice input are not sent to those services. Model catalog browsing and model downloads do contact Hugging Face, but do not send chat content. The Windows launcher binds the server to `127.0.0.1`; do not change it to a public or LAN interface if you want to keep it local.

Keep your Windows account and project data directory private. Anyone with access to your Windows account or files may be able to access local chat data; this app does not provide database encryption at rest.

The installer downloads and extracts Python 3.12.10 x64 from the official NuGet package into the project directory, then recreates `.venv312` using that interpreter. This avoids conflicts with other installed Python versions. PyTorch is installed from its Python 3.12 CUDA 12.8 wheel index on NVIDIA systems, with CPU wheels as fallback. `llama-cpp-python` is installed from prebuilt wheel indexes so the installer does not attempt a local CMake/C++ build. Run the updated `install.bat` and `run.bat`; the old `.venv` is intentionally not used.
