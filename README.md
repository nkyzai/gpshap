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
