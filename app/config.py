import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
UPLOAD_DIR = ROOT / "uploads"
INDEX_DIR = ROOT / "indexes"
STATIC_DIR = ROOT / "static"
MODELS_DIR = ROOT / "models"
GENERATED_DIR = DATA_DIR / "generated"

for _d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, STATIC_DIR, MODELS_DIR, GENERATED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Keep every byte of model weight inside the project folder -------------
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR / "hub")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


CONFIG_PATH = DATA_DIR / "config.json"

DEFAULT_CONFIG = {
    # server
    "host": "127.0.0.1",
    "port": 8787,

    # models
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vision_model": "microsoft/Florence-2-base",
    "image_model": "stabilityai/sd-turbo",

    # vision
    "vision_mode": "auto",          # auto | vlm | caption | off
    "vision_auto_analyze": True,

    # diffusion
    "image_steps": 4,
    "image_width": 512,
    "image_height": 512,
    "image_guidance": 0.0,
    "image_negative": "",

    # generation
    "max_new_tokens": 768,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.08,
    "device": "auto",               # auto | cpu | cuda
    "dtype": "auto",                # auto | float16 | bfloat16 | float32

    # rag
    "chunk_size": 900,
    "chunk_overlap": 120,
    "top_k_rag": 4,
    "history_messages": 12,
    "context_budget": 6000,

    # agent
    "agent_router": True,
    "max_tool_rounds": 3,
    "stream": True,

    # hub
    "hf_token": "",
    
    # GGUF / Llama.cpp backend settings (CRITICAL FOR ROUTING)
    "active_backend": "transformers",
    "active_gguf_model": "",
    "active_model_path": "",
}


# In app/config.py, inside load_config(), add this check after loading stored config:

def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                config.update(stored)
        except Exception:
            pass
    
    # --- SANITIZATION LOGIC ---
    # If user has a GGUF path in 'model' but backend is 'transformers', reset to default
    current_model = config.get("model", "")
    current_backend = config.get("active_backend", "transformers")
    
    if ".gguf" in str(current_model).lower() and current_backend == "transformers":
        print(f"[gapshap] Warning: Detected GGUF model '{current_model}' with 'transformers' backend.", flush=True)
        print("[gapshap] Auto-resetting to safe default model to prevent crash.", flush=True)
        config["model"] = DEFAULT_CONFIG["model"] # Reset to Qwen2.5-1.5B
        save_config(config)
        
    token = config.get("hf_token") or ""
    if token:
        os.environ["HF_TOKEN"] = token

    return config


def save_config(config: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def update_config(patch: dict) -> dict:
    """Merge patch into config, persist, and return the fresh config."""
    allowed_keys = set(DEFAULT_CONFIG.keys())
    filtered_patch = {k: v for k, v in patch.items() if k in allowed_keys}
    
    CONFIG.update(filtered_patch)
    save_config(CONFIG)

    token = CONFIG.get("hf_token") or ""
    if token:
        os.environ["HF_TOKEN"] = token

    return CONFIG


CONFIG = load_config()