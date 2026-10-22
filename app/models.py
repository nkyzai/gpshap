import json
import threading
from pathlib import Path
import requests
from huggingface_hub import snapshot_download
from .config import ROOT

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

_download_status = {}

def search_models(query: str, limit: int = 20):
    url = "https://huggingface.co/api/models"
    params = {
        "search": query,
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
        "full": "true"
    }
    try:
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        models = res.json()
        
        results = []
        for m in models:
            results.append({
                "id": m.get("id"),
                "pipeline_tag": m.get("pipeline_tag", "unknown"),
                "tags": m.get("tags", []),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
            })
        return results
    except Exception as e:
        print(f"[gapshap] HF search failed: {e}")
        return []

def list_local_models():
    local = []
    for p in MODELS_DIR.iterdir():
        if p.is_dir():
            config_path = p / "config.json"
            info = {"id": p.name.replace("--", "/"), "path": str(p), "dir_name": p.name}
            if config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text())
                    info["model_type"] = cfg.get("model_type", "unknown")
                    info["architectures"] = cfg.get("architectures", [])
                except: pass
            local.append(info)
    return local

def download_model_background(model_id: str):
    if model_id in _download_status and _download_status[model_id] == "downloading":
        return
    
    _download_status[model_id] = "downloading"
    
    def _download():
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=MODELS_DIR / model_id.replace("/", "--"),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.bin", "*.pt", "*.msgpack", "*.h5", "*.ot"] 
            )
            _download_status[model_id] = "completed"
        except Exception as e:
            _download_status[model_id] = f"failed: {str(e)}"
            
    threading.Thread(target=_download, daemon=True).start()

def get_download_status(model_id: str):
    return _download_status.get(model_id, "not_started")