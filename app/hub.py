import json
import os
import threading
import time
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import CONFIG, DEFAULT_CONFIG, MODELS_DIR, update_config
from . import llm
from .model_runtime import MODEL_SESSION_LOCK

# Job Management
JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()
CANCEL_EVENTS: dict[str, threading.Event] = {}

def local_repo_path(repo_id: str) -> Path:
    return MODELS_DIR / repo_id.replace("/", "--")

def disk_free() -> dict:
    total, used, free = shutil.disk_usage(MODELS_DIR)
    models_size = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    return {
        "total_gb": round(total / 1024**3, 2),
        "free_gb": round(free / 1024**3, 2),
        "models_gb": round(models_size / 1024**3, 2),
    }

def jobs() -> list[dict]:
    with JOB_LOCK:
        return [
            {
                **job,
                "status": (
                    "cancelling"
                    if repo_id in CANCEL_EVENTS and CANCEL_EVENTS[repo_id].is_set()
                    else job["status"]
                ),
            }
            for repo_id, job in JOBS.items()
        ]

def active_job() -> Optional[dict]:
    with JOB_LOCK:
        for repo_id, job in JOBS.items():
            if (job["status"] in ("downloading", "queued")
                    or repo_id in CANCEL_EVENTS and CANCEL_EVENTS[repo_id].is_set()):
                return {**job, "status": "cancelling" if repo_id in CANCEL_EVENTS
                        and CANCEL_EVENTS[repo_id].is_set() else job["status"]}
    return None

def cancel(repo_id: str) -> dict:
    with JOB_LOCK:
        if repo_id in CANCEL_EVENTS and JOBS.get(repo_id, {}).get("status") in (
            "queued", "downloading", "cancelling"
        ):
            CANCEL_EVENTS[repo_id].set()
            JOBS[repo_id]["status"] = "cancelling"
            return {"ok": True, "repo_id": repo_id}
    return {"ok": False, "error": "No active download was found for this model."}


def delete_job(repo_id: str) -> dict:
    with JOB_LOCK:
        if repo_id in CANCEL_EVENTS:
            raise RuntimeError("The download is still stopping; try deleting the job again shortly.")
        JOBS.pop(repo_id, None)
    return {"ok": True, "repo_id": repo_id}


def is_cached(repo_id: str) -> bool:
    path = local_repo_path(repo_id)
    if path.is_dir():
        if (path / ".gpshap-download-complete").is_file():
            return True
        has_weights = any(
            file.is_file() and file.stat().st_size > 0
            and file.name.lower().endswith((".safetensors", ".bin", ".gguf"))
            and ".cache" not in file.parts
            for file in path.rglob("*")
        )
        return has_weights and (
            (path / "config.json").is_file()
            or (path / "model_index.json").is_file()
            or repo_id.lower().endswith(".gguf")
        )
    if repo_id.endswith(".gguf"):
        return (MODELS_DIR / repo_id).exists()
    return False

def local_models() -> list[dict]:
    models = []
    # Scan for HF-style folders
    for item in MODELS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            repo_id = item.name.replace("--", "/")
            is_image = (item / "model_index.json").is_file()
            has_transformer_config = (
                (item / "config.json").is_file()
                and any(
                    file.is_file() and file.stat().st_size > 0
                    and file.name.lower().endswith((".safetensors", ".bin"))
                    and ".cache" not in file.parts
                    for file in item.rglob("*")
                )
            )
            if is_image and is_cached(repo_id) or has_transformer_config:
                local_bytes = sum(
                    file.stat().st_size for file in item.rglob("*")
                    if file.is_file() and ".cache" not in file.parts
                    and file.name != ".gpshap-download-complete"
                )
                models.append({
                    "id": repo_id,
                    "kind": "image" if is_image else "llm",
                    "path": str(item),
                    "backend": "diffusers" if is_image else "transformers",
                    "format": "diffusers" if is_image else "transformers",
                    "downloaded": True,
                    "status": "ready",
                    "local_bytes": local_bytes,
                    "size_estimated": False,
                })
            for gguf_file in item.rglob("*.gguf"):
                models.append({
                    "id": gguf_file.relative_to(MODELS_DIR).as_posix(),
                    "kind": "llm",
                    "path": str(gguf_file),
                    "backend": "llama.cpp",
                    "format": "gguf",
                    "downloaded": True,
                    "status": "ready",
                    "local_bytes": gguf_file.stat().st_size,
                    "size_estimated": False,
                })
        elif item.is_file() and item.name.endswith(".gguf"):
             models.append({
                "id": item.name,
                "kind": "llm",
                "path": str(item),
                "backend": "llama.cpp",
                "format": "gguf",
                "downloaded": True,
               "status": "ready",
               "local_bytes": item.stat().st_size,
               "size_estimated": False,
            })
    return models

def browse(search: str = "", task: str = "all", sort: str = "downloads", 
           limit: int = 40, deep: bool = False, force: bool = False) -> dict:
    from . import hub_api
    try:
        raw = hub_api.search_models(search, task, sort, limit)
        enriched = hub_api.enrich_many(raw, deep=deep)
        
        # Mark downloaded status
        for m in enriched:
            m["downloaded"] = is_cached(m["id"])
            if m["format"] == "gguf":
                m["backend"] = "llama.cpp"
            else:
                m["backend"] = "transformers"
                
        return {
            "models": enriched,
            "disk": disk_free(),
            "jobs": jobs()
        }
    except Exception as e:
        raise e

def details(repo_id: str) -> dict:
    from . import hub_api
    return hub_api.describe(repo_id)

def download(repo_id: str, kind: str = "llm") -> dict:
    job_id = repo_id
    with JOB_LOCK:
        if job_id in CANCEL_EVENTS or (
            job_id in JOBS and JOBS[job_id]["status"] in ("downloading", "queued")
        ):
            return {"ok": False, "error": "Already downloading or cancellation is still pending"}

        cancel_event = threading.Event()
        JOBS[job_id] = {
            "repo_id": repo_id,
            "status": "queued",
            "progress": 0,
            "total": 0,
            "downloaded": 0,
            "started_at": time.time(),
            "kind": kind
        }
        CANCEL_EVENTS[job_id] = cancel_event

    def _worker():
        process = None
        try:
            local_dir = local_repo_path(repo_id)
            local_dir.mkdir(parents=True, exist_ok=True)
            if cancel_event.is_set():
                with JOB_LOCK:
                    JOBS[job_id]["status"] = "cancelled"
                return

            environment = os.environ.copy()
            if CONFIG.get("hf_token"):
                environment["HF_TOKEN"] = str(CONFIG["hf_token"])
            process = subprocess.Popen(
                [sys.executable, "-m", "app.download_worker", repo_id, str(local_dir)],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=environment,
            )
            with JOB_LOCK:
                JOBS[job_id]["status"] = "downloading"
                JOBS[job_id]["stage"] = "Downloading files from Hugging Face"

            next_size_update = 0.0
            while process.poll() is None:
                if cancel_event.wait(0.25):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                if time.monotonic() >= next_size_update:
                    downloaded = sum(
                        file.stat().st_size for file in local_dir.rglob("*")
                        if file.is_file() and ".cache" not in file.parts
                    )
                    with JOB_LOCK:
                        JOBS[job_id]["downloaded"] = downloaded
                    next_size_update = time.monotonic() + 1.0

            with JOB_LOCK:
                if cancel_event.is_set():
                    JOBS[job_id]["status"] = "cancelled"
                elif process.returncode == 0:
                    (local_dir / ".gpshap-download-complete").write_text(
                        repo_id, encoding="utf-8"
                    )
                    JOBS[job_id]["status"] = "ready"
                    JOBS[job_id]["progress"] = 1.0
                else:
                    JOBS[job_id]["status"] = "error"
                    JOBS[job_id]["error"] = (
                        f"Download process exited with code {process.returncode}; "
                        "see the server log for details."
                    )
        except Exception as e:
            with JOB_LOCK:
                if cancel_event.is_set():
                    JOBS[job_id]["status"] = "cancelled"
                else:
                    JOBS[job_id]["status"] = "error"
                    JOBS[job_id]["error"] = str(e)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            with JOB_LOCK:
                CANCEL_EVENTS.pop(job_id, None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return {"ok": True, "job_id": job_id}

def use_model(
    model_id: str,
    kind: str = "llm",
    gguf_n_ctx: int | None = None,
    gguf_n_gpu_layers: int | None = None,
) -> dict:
    """
    Routes the model to the correct backend.
    If it's a GGUF file, switches to llama.cpp.
    If it's a HF repo, switches to transformers.
    """
    is_gguf = model_id.lower().endswith(".gguf")
    repo_dir = local_repo_path(model_id)
    if not is_gguf and repo_dir.is_dir():
        is_gguf = any(repo_dir.rglob("*.gguf"))

    if is_gguf:
        from . import imagegen

        local_entry = next(
            (item for item in local_models() if item["id"] == model_id),
            None,
        )
        if local_entry:
            candidate = Path(local_entry["path"])
        elif repo_dir.is_dir():
            gguf_files = sorted(repo_dir.rglob("*.gguf"))
            if len(gguf_files) != 1:
                raise ValueError(
                    f"Repository {model_id} contains {len(gguf_files)} GGUF files; "
                    "select a specific GGUF file from the local model inventory."
                )
            candidate = gguf_files[0]
        else:
            candidate = Path(model_id).expanduser()
            if not candidate.is_file() and not candidate.is_absolute():
                candidate = MODELS_DIR / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"GGUF model file not found: {model_id}")
        gguf_path = str(candidate.resolve())

        previous_gguf_config = {
            "gguf_n_ctx": CONFIG.get("gguf_n_ctx", 4096),
            "gguf_n_gpu_layers": CONFIG.get("gguf_n_gpu_layers", -1),
        }
        requested_gguf_config = {
            key: value for key, value in {
                "gguf_n_ctx": gguf_n_ctx,
                "gguf_n_gpu_layers": gguf_n_gpu_layers,
            }.items() if value is not None
        }
        if requested_gguf_config:
            update_config(requested_gguf_config)

        with MODEL_SESSION_LOCK:
            imagegen.unload()
            try:
                llm.load_gguf_model(gguf_path, force=True)
            except Exception as exc:
                llm.unload()
                update_config(previous_gguf_config)
                fallback_model = CONFIG.get("model", DEFAULT_CONFIG["model"])
                if str(fallback_model).lower().endswith(".gguf"):
                    fallback_model = DEFAULT_CONFIG["model"]
                update_config({
                    "active_backend": "transformers",
                    "active_gguf_model": "",
                    "active_model_path": "",
                    "model": fallback_model,
                })
                raise RuntimeError(f"Failed to load GGUF: {exc}") from exc

            update_config({
                "active_backend": "llama",
                "active_gguf_model": model_id,
                "active_model_path": gguf_path,
            })
        return {"ok": True, "field": "llm", "model": model_id, "backend": "llama.cpp"}
            
    else:
        from . import imagegen

        with MODEL_SESSION_LOCK:
            imagegen.unload()
            llm.unload()
            update_config({
                "active_backend": "transformers",
                "model": model_id,
                "active_gguf_model": "",
                "active_model_path": ""
            })
            threading.Thread(target=_warmup_llm_exclusive, daemon=True).start()
        return {"ok": True, "field": "llm", "model": model_id, "backend": "transformers"}


def _warmup_llm_exclusive() -> None:
    from . import imagegen

    with MODEL_SESSION_LOCK:
        imagegen.unload()
        llm.warmup()


def remove_model(repo_id: str) -> dict:
    path = local_repo_path(repo_id)
    if path.exists():
        shutil.rmtree(path)
        return {"ok": True}
    
    # Check for direct GGUF
    gguf_path = MODELS_DIR / repo_id
    if gguf_path.exists():
        gguf_path.unlink()
        return {"ok": True}
        
    raise FileNotFoundError(f"Model {repo_id} not found on disk")