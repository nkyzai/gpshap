import fnmatch
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from queue import Queue
from urllib.parse import quote

import requests

from . import db
from .config import CONFIG, MODELS_DIR

CACHE_ROOT = MODELS_DIR / "hub"

BASE_IGNORE = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.tflite", "*.pb", "flax_model*", "*.mlindex", "original/*", "consolidated*", "*.pth", "*.ckpt", "*training_args*", "*.ipynb"]

KIND_BY_TASK = {"text-generation": "llm", "image-text-to-text": "vision", "text-to-image": "image", "feature-extraction": "embedding", "sentence-similarity": "embedding", "image-to-text": "vision", "visual-question-answering": "vision", "text-classification": "other", "automatic-speech-recognition": "audio"}

PRESETS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "llm", 0.49, 1.0, 32768, ["chat", "tools", "cpu-friendly", "fast"], "Tiny"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "llm", 1.54, 3.1, 32768, ["chat", "tools", "cpu-friendly"], "Default"),
    ("Qwen/Qwen2.5-7B-Instruct", "llm", 7.62, 15.2, 131072, ["chat", "tools", "code", "long-context"], "Recommended"),
    ("NousResearch/Hermes-2-Pro-Llama-3-8B", "llm", 8.03, 16.1, 8192, ["chat", "tools", "function-calling"], "Best open tool-caller"),
    ("microsoft/Florence-2-base", "vision", 0.23, 0.5, 1024, ["vision", "ocr", "captioning", "cpu-friendly", "fast"], "Analyser"),
    ("sentence-transformers/all-MiniLM-L6-v2", "embedding", 0.02, 0.09, 256, ["embeddings", "rag", "cpu-friendly", "fast"], "RAG Embedder"),
    ("stabilityai/sd-turbo", "image", 0.93, 4.9, 77, ["image-generation", "fast"], "Diffusion"),
]

def _endpoint() -> str: return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
def _token() -> str | None: return (CONFIG.get("hf_token") or "").strip() or None
def _headers() -> dict:
    h = {"Accept": "application/json", "User-Agent": "gapshap/2.1"}
    t = _token()
    if t: h["Authorization"] = f"Bearer {t}"
    return h

def _get_json(path: str, params: dict | None = None, timeout=(5, 25)) -> any:
    url = _endpoint() + path
    try:
        r = requests.get(url, params=params, headers=_headers(), timeout=timeout)
        if r.status_code >= 400: raise Exception(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()
    except Exception as e:
        print(f"[gapshap] Hub GET failed {path}: {e}", flush=True)
        return []

def cache_repo_path(repo_id: str) -> Path: return CACHE_ROOT / f"models--{repo_id.replace('/', '--')}"
def is_cached(repo_id: str) -> bool:
    path = cache_repo_path(repo_id)
    if not path.exists(): return False
    snapshots = path / "snapshots"
    if not snapshots.exists(): return False
    for snap in snapshots.iterdir():
        if snap.is_dir() and any(snap.iterdir()): return True
    return False

def dir_size(path: Path) -> int:
    total = 0
    if not path.exists(): return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                if not fp.is_symlink(): total += fp.stat().st_size
            except Exception: pass
    return total

def classify_kind(pipeline_tag: str, tags: set, model_id: str) -> str:
    low = model_id.lower()
    tagstr = " ".join(tags).lower()
    if pipeline_tag in KIND_BY_TASK: base = KIND_BY_TASK[pipeline_tag]
    elif "diffusers" in tagstr or "text-to-image" in tagstr: base = "image"
    elif "sentence-transformers" in tagstr or "embeddings" in tagstr: base = "embedding"
    else: base = "llm"
    if base == "llm" and any(k in low for k in ("vl", "llava", "vision", "qwen2-vl", "minicpm-v", "idefics", "pixtral", "florence", "kosmos")): base = "vision"
    return base

def detect_capabilities(model_id: str, tags: set, pipeline_tag: str, context: int, params_b: float | None, kind: str) -> list[str]:
    low = model_id.lower()
    tagstr = " ".join(t.lower() for t in tags)
    caps = set()
    if kind == "embedding" or "sentence-transformers" in tagstr: caps.add("embeddings")
    if kind == "image" or "text-to-image" in tagstr: caps.add("image-generation")
    if kind == "vision" or pipeline_tag == "image-text-to-text": caps.add("vision")
    if any(k in low for k in ("instruct", "-it", "chat", "hermes", "wizard")) or "conversational" in tagstr or "chatml" in tagstr: caps.add("chat")
    if "code" in low or "coder" in low or "code" in tagstr: caps.add("code")
    if any(k in tagstr for k in ("function-calling", "tool", "tools")) or any(k in low for k in ("hermes", "qwen2.5", "qwen3", "llama-3.1", "llama-3.2", "firefunction", "mistral-nemo")): caps.add("tools")
    if context and context >= 32768: caps.add("long-context")
    if any(k in low for k in ("gguf", "awq", "gptq", "bnb", "4bit", "8bit")): caps.add("quantized")
    if params_b:
        if params_b <= 2: caps.update({"cpu-friendly", "fast"})
        elif params_b <= 8: caps.add("balanced")
        else: caps.add("gpu-required")
    if "gated" in tagstr: caps.add("gated")
    order = ["chat", "tools", "vision", "ocr", "image-generation", "embeddings", "code", "reasoning", "long-context", "multilingual", "quantized", "captioning", "cpu-friendly", "fast", "balanced", "gpu-required", "gated"]
    return [c for c in order if c in caps] + sorted(caps - set(order))

def estimate_bytes(params_b: float | None, tags: set, model_id: str) -> int:
    if not params_b: return 0
    low = model_id.lower()
    tagstr = " ".join(t.lower() for t in tags)
    blob = low + " " + tagstr
    per_param = 2.0
    if any(k in blob for k in ("q4", "4bit", "awq", "gptq", "int4")): per_param = 0.55
    elif any(k in blob for k in ("q8", "8bit", "int8", "fp8")): per_param = 1.05
    elif "fp32" in blob or "float32" in blob: per_param = 4.0
    return int(params_b * 1e9 * per_param)

def _preset_map() -> dict:
    out = {}
    for pid, kind, params, size, ctx, caps, note in PRESETS:
        out[pid] = {"id": pid, "kind": kind, "params_billion": params, "size_bytes": int(size * 1024 ** 3), "context_tokens": ctx, "capabilities": caps, "note": note, "source": "preset", "likes": 0, "downloads": 0, "library": ""}
    return out
PRESET_MAP = _preset_map()

def enrich_from_config(repo_id: str, record: dict) -> dict:
    try:
        txt = requests.get(f"{_endpoint()}/{quote(repo_id, safe='/')}/resolve/main/config.json", headers=_headers(), timeout=(5, 10)).text
        cfg = json.loads(txt)
        ctx = 0
        for key in ("max_position_embeddings", "model_max_length", "seq_length", "n_positions", "max_sequence_length", "sliding_window"):
            val = cfg.get(key)
            if isinstance(val, int) and val > ctx: ctx = val
        if ctx: record["context_tokens"] = ctx
        arch = cfg.get("architectures") or []
        if arch:
            record["architecture"] = arch[0]
            low = arch[0].lower()
            caps = set(record.get("capabilities") or [])
            if any(k in low for k in ("vl", "vision", "image", "llava", "idefics", "florence")):
                caps.add("vision")
                if record.get("kind") == "llm": record["kind"] = "vision"
            if "diffusion" in low or "stable" in low or "unet" in low: caps.add("image-generation")
            record["capabilities"] = sorted(caps)
    except Exception: pass
    return record

def exact_size(repo_id: str) -> tuple[int, bool]:
    try:
        data = _get_json(f"/api/models/{quote(repo_id, safe='/')}", params={"blobs": "true"})
        if not isinstance(data, dict): return 0, False
        total = 0; has_st = False
        for sib in data.get("siblings") or []:
            name = sib.get("rfilename", "") or ""
            size = sib.get("size", 0) or 0
            if name.endswith(".safetensors"): has_st = True
            if any(fnmatch.fnmatch(name, pat) for pat in BASE_IGNORE): continue
            if name.endswith((".safetensors", ".bin", ".gguf", ".pt", ".json", ".txt", ".model")): total += size
        return total, has_st
    except Exception: return 0, False

_BROWSE_CACHE: dict = {}
_BROWSE_LOCK = threading.Lock()
_BROWSE_TTL = 180

def _norm(text: str) -> str: return re.sub(r"[^a-z0-9]+", "", (text or "").lower())
def _query_matches(repo_id: str, query: str) -> bool:
    if not (query or "").strip(): return True
    rid = _norm(repo_id)
    if _norm(query) in rid: return True
    tokens = [_norm(t) for t in query.split() if _norm(t)]
    return bool(tokens) and all(t in rid for t in tokens)
def _preset_matches_task(kind: str, task: str) -> bool:
    if task in ("all", "", None): return True
    mapping = {"text-generation": {"llm"}, "llm": {"llm"}, "image-text-to-text": {"vision"}, "vision": {"vision"}, "feature-extraction": {"embedding"}, "embedding": {"embedding"}, "text-to-image": {"image"}, "image": {"image"}}
    return kind in mapping.get(task, {task})

def _search_live(query: str, task: str, sort: str, limit: int) -> list[dict]:
    params = {"limit": max(1, min(int(limit), 100)), "sort": sort if sort in ("downloads", "likes", "lastModified", "createdAt", "trendingScore") else "downloads", "direction": -1, "full": "true"}
    if query.strip(): params["search"] = query.strip()
    real_tasks = {"llm": "text-generation", "vision": "image-text-to-text", "embedding": "feature-extraction", "image": "text-to-image", "audio": "automatic-speech-recognition"}
    filter_val = None
    if task != "all":
        if task in real_tasks: filter_val = real_tasks[task]
        elif task in ("text-generation", "image-text-to-text", "feature-extraction", "text-to-image"): filter_val = task
    if filter_val: params["filter"] = filter_val
    data = _get_json("/api/models", params=params)
    if isinstance(data, dict): data = data.get("items") or data.get("models") or []
    if not isinstance(data, list): return []
    return [x for x in data if isinstance(x, dict) and (x.get("id") or x.get("modelId"))]

def _enrich_one(raw: dict, deep: bool = False) -> dict:
    repo_id = raw.get("id") or raw.get("modelId") or ""
    tags = set(raw.get("tags") or [])
    pipeline = raw.get("pipeline_tag") or ""
    library = raw.get("library_name") or ""
    safetensors = raw.get("safetensors") or {}
    fmt = "transformers"
    if "gguf" in " ".join(tags).lower() or any(str(f).endswith(".gguf") for f in (raw.get("files") or [])): fmt = "gguf"
    elif "diffusers" in " ".join(tags).lower(): fmt = "diffusers"
    elif "onnx" in " ".join(tags).lower(): fmt = "onnx"
    kind = classify_kind(pipeline, tags, repo_id)
    params_b = None
    st_total = safetensors.get("total") if isinstance(safetensors, dict) else None
    if st_total and st_total > 0: params_b = round(st_total / 1e9, 3)
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*([bm])\b", repo_id.lower())
        if m:
            v = float(m.group(1))
            if m.group(2) == "m": v /= 1000.0
            if 0.01 <= v <= 2000: params_b = round(v, 3)
    context = 0
    if deep:
        rec = {"id": repo_id, "kind": kind, "capabilities": []}
        rec = enrich_from_config(repo_id, rec)
        context = rec.get("context_tokens", 0)
    else:
        if "qwen2.5" in repo_id.lower(): context = 32768
        elif "llama-3" in repo_id.lower(): context = 8192
        elif "phi-3" in repo_id.lower(): context = 128000
    size_bytes = 0; estimated = True
    if deep:
        sz, _ = exact_size(repo_id)
        if sz > 0: size_bytes = sz; estimated = False
    if not size_bytes: size_bytes = estimate_bytes(params_b, tags, repo_id)
    caps = detect_capabilities(repo_id, tags, pipeline, context, params_b, kind)
    note = ""
    if fmt == "gguf": note = "GGUF - Uses Llama.cpp backend"
    elif raw.get("gated"): note = "Gated - Accept license on HF website first"
    elif kind == "image": note = "Diffusion Image Model"
    elif kind == "vision": note = "Vision/Multimodal Model"
    elif kind == "embedding": note = "Embedding/RAG Model"
    return {"id": repo_id, "kind": kind, "format": fmt, "runnable": True, "params_billion": params_b, "size_bytes": size_bytes, "size_estimated": estimated, "size_exact": not estimated, "context_tokens": context, "capabilities": caps, "pipeline_tag": pipeline, "architecture": "", "library": library, "likes": int(raw.get("likes") or 0), "downloads": int(raw.get("downloads") or 0), "last_modified": str(raw.get("lastModified") or "")[:10], "created_at": str(raw.get("createdAt") or "")[:10], "gated": bool(raw.get("gated")), "source": "hub", "downloaded": False, "note": note, "url": f"{_endpoint()}/{repo_id}", "files": []}

def browse(search: str = "", task: str = "text-generation", sort: str = "downloads", limit: int = 40, deep: bool = False, force: bool = False) -> dict:
    query = (search or "").strip()
    limit = max(1, min(int(limit), 100))
    key = (query.lower(), task, sort, limit, bool(deep))
    if not force:
        with _BROWSE_LOCK:
            hit = _BROWSE_CACHE.get(key)
            if hit and time.time() - hit["ts"] < _BROWSE_TTL:
                out = dict(hit["value"]); out["cached"] = True; return out
    started = time.time(); live: list[dict] = []; warning = ""; source = "live"
    try:
        raws = _search_live(query, task, sort, limit)
        enrich_limit = 20 if not deep else len(raws)
        live = [_enrich_one(r, deep=deep) for r in raws[:enrich_limit]]
    except Exception as exc:
        warning = f"{type(exc).__name__}: {exc}"; source = "offline"; traceback.print_exc()
    presets: dict[str, dict] = {}
    for pid, preset in PRESET_MAP.items():
        if not _preset_matches_task(preset.get("kind", "llm"), task): continue
        if not _query_matches(pid, query): continue
        presets[pid] = {**preset, "source": "preset", "downloaded": is_cached(pid), "size_estimated": True, "runnable": True, "format": "transformers", "url": f"{_endpoint()}/{pid}"}
    if warning:
        models = [] if query else list(presets.values()); source = "offline"
    else:
        merged: dict[str, dict] = {}
        for item in live: merged[item["id"]] = item
        for pid, preset in presets.items():
            if pid in merged:
                cur = merged[pid]
                for field in ("context_tokens", "size_bytes", "params_billion"):
                    if not cur.get(field) and preset.get(field): cur[field] = preset[field]
                if not cur.get("note"): cur["note"] = preset.get("note", "")
                cur["source"] = "hub+preset"
            else: merged[pid] = preset
        models = list(merged.values())
        if not live: source = "preset-only" if models else "empty"
    for model in models:
        model.setdefault("capabilities", []); model.setdefault("downloaded", False)
        downloaded = bool(model.get("downloaded")) or is_cached(model["id"])
        model["downloaded"] = downloaded; model["local_bytes"] = dir_size(cache_repo_path(model["id"])) if downloaded else 0
        model["status"] = "ready" if downloaded else "not_installed"
    if sort == "size": models.sort(key=lambda m: m.get("size_bytes") or 10 ** 15)
    elif sort == "likes": models.sort(key=lambda m: m.get("likes") or 0, reverse=True)
    result = {"models": models, "count": len(models), "live_count": len(live), "query": query, "task": task, "sort": sort, "source": source, "warning": warning, "endpoint": _endpoint(), "elapsed": round(time.time() - started, 2), "cached": False, "disk": disk_free(), "hint": "No repository matched." if source == "empty" else ""}
    if not warning:
        with _BROWSE_LOCK: _BROWSE_CACHE[key] = {"ts": time.time(), "value": result}
    return result

def details(repo_id: str) -> dict:
    record = dict(PRESET_MAP.get(repo_id, {"id": repo_id, "source": "hub"}))
    try:
        raw = _get_json(f"/api/models/{quote(repo_id, safe='/')}")
        if isinstance(raw, dict):
            enriched = _enrich_one(raw, deep=True)
            record.update({k: v for k, v in enriched.items() if v not in (None, "", [], False)})
    except Exception as exc: record["error"] = f"{type(exc).__name__}: {exc}"
    record.setdefault("kind", PRESET_MAP.get(repo_id, {}).get("kind", "llm"))
    record.setdefault("capabilities", PRESET_MAP.get(repo_id, {}).get("capabilities", []))
    record["downloaded"] = is_cached(repo_id); record["local_bytes"] = dir_size(cache_repo_path(repo_id))
    return record

class _Job:
    def __init__(self, repo_id: str, kind: str):
        self.id = uuid.uuid4().hex[:8]; self.repo_id = repo_id; self.kind = kind
        self.status = "queued"; self.progress = 0.0; self.downloaded = 0; self.total = 0
        self.error = ""; self.cancel = threading.Event(); self.started = time.time()
        self.finished = None; self.stage = "queued"
    def to_dict(self) -> dict:
        return {"id": self.id, "repo_id": self.repo_id, "kind": self.kind, "status": self.status, "progress": round(self.progress, 3), "downloaded": self.downloaded, "total": self.total, "error": self.error, "stage": self.stage, "elapsed": round((self.finished or time.time()) - self.started, 1)}

_JOBS: dict[str, _Job] = {}
_QUEUE: "Queue[str]" = Queue()
_JOBS_LOCK = threading.Lock()
_WORKER = None

def _worker():
    while True:
        repo_id = _QUEUE.get()
        with _JOBS_LOCK: job = _JOBS.get(repo_id)
        if job is None: continue
        try: _run_download(job)
        except Exception as exc:
            job.status = "error"; job.error = f"{type(exc).__name__}: {exc}"; job.finished = time.time()
            db.upsert_model(repo_id, kind=job.kind, status="error", progress=0.0, error=job.error)
            print(f"[gapshap] download failed {repo_id}: {exc}", flush=True)
        finally: _QUEUE.task_done()

def _ensure_worker():
    global _WORKER
    if _WORKER is None or not _WORKER.is_alive():
        _WORKER = threading.Thread(target=_worker, daemon=True, name="hub-downloader"); _WORKER.start()

def _monitor(job: _Job, stop: threading.Event):
    path = cache_repo_path(job.repo_id)
    while not stop.is_set():
        try:
            job.downloaded = dir_size(path)
            if job.total > 0: job.progress = min(0.995, job.downloaded / job.total)
        except Exception: pass
        stop.wait(1.0)

def _find_gguf_file(repo_id: str) -> str | None:
    repo_path = cache_repo_path(repo_id)
    snapshots_dir = repo_path / "snapshots"
    if not snapshots_dir.exists(): return None
    for snap in snapshots_dir.iterdir():
        if snap.is_dir():
            for f in snap.iterdir():
                if f.suffix.lower() == ".gguf": return str(f.resolve())
    return None

def _run_download(job: _Job):
    from huggingface_hub import snapshot_download
    job.status = "downloading"; job.stage = "resolving"
    db.upsert_model(job.repo_id, kind=job.kind, status="downloading", progress=0.0)
    ignore = list(BASE_IGNORE); total_size = 0
    try:
        _size, has_st = exact_size(job.repo_id)
        total_size = _size
        if has_st: ignore.append("*.bin")
    except Exception: pass
    if not total_size: total_size = PRESET_MAP.get(job.repo_id, {}).get("size_bytes", 0)
    job.total = total_size
    stop = threading.Event()
    monitor = threading.Thread(target=_monitor, args=(job, stop), daemon=True); monitor.start()
    class _Tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kw):
            self.total = total; self.desc = desc or ""; self.n = 0
            if desc: job.stage = str(desc)[:60]
        def update(self, n=1):
            if job.cancel.is_set(): raise KeyboardInterrupt("cancelled by user")
            self.n += n
        def set_description(self, desc): job.stage = str(desc)[:60]
        def close(self): pass
        def refresh(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    try:
        snapshot_download(repo_id=job.repo_id, ignore_patterns=ignore, token=(CONFIG.get("hf_token") or "").strip() or None, max_workers=4, tqdm_class=_Tqdm)
    except KeyboardInterrupt:
        stop.set(); job.status = "cancelled"; job.stage = "cancelled"; db.set_model_status(job.repo_id, "error", error="cancelled"); return
    except Exception as exc:
        stop.set(); msg = str(exc)
        if "gated" in msg.lower() or "401" in msg or "403" in msg: msg = "This model is gated. Accept its licence on huggingface.co, then paste a read token into gapshap Settings -> HuggingFace token."
        job.status = "error"; job.error = msg; job.finished = time.time()
        db.upsert_model(job.repo_id, kind=job.kind, status="error", progress=0.0, error=msg); raise
    stop.set()
    final_size = dir_size(cache_repo_path(job.repo_id))
    job.downloaded = final_size; job.total = job.total or final_size; job.progress = 1.0; job.status = "ready"; job.stage = "ready"; job.finished = time.time()
    gguf_file = _find_gguf_file(job.repo_id)
    record = details(job.repo_id)
    db.upsert_model(job.repo_id, kind=record.get("kind", job.kind), size_bytes=record.get("size_bytes") or final_size, context_tokens=record.get("context_tokens") or 0, params=str(record.get("params_billion") or ""), pipeline_tag=record.get("pipeline_tag", ""), capabilities=json.dumps(record.get("capabilities") or []), status="ready", progress=1.0, error="", gguf_path=gguf_file or "")
    print(f"[gapshap] download ready: {job.repo_id} (GGUF: {bool(gguf_file)})", flush=True)

def download(repo_id: str, kind: str = "llm") -> dict:
    with _JOBS_LOCK:
        existing = _JOBS.get(repo_id)
        if existing and existing.status in ("queued", "downloading"): return existing.to_dict()
        job = _Job(repo_id, kind); _JOBS[repo_id] = job
    preset = PRESET_MAP.get(repo_id, {})
    db.upsert_model(repo_id, kind=kind or preset.get("kind", "llm"), size_bytes=preset.get("size_bytes", 0), context_tokens=preset.get("context_tokens", 0), params=str(preset.get("params_billion") or ""), capabilities=json.dumps(preset.get("capabilities") or []), status="queued", progress=0.0, error="")
    _ensure_worker(); _QUEUE.put(repo_id)
    return job.to_dict()

def cancel(repo_id: str) -> dict:
    with _JOBS_LOCK: job = _JOBS.get(repo_id)
    if not job: return {"ok": False, "error": "no active job"}
    job.cancel.set()
    return {"ok": True, "job": job.to_dict()}

def jobs() -> list[dict]:
    with _JOBS_LOCK: return [j.to_dict() for j in _JOBS.values()]

def active_job() -> dict | None:
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if job.status in ("queued", "downloading"): return job.to_dict()
    return None

def local_models() -> list[dict]:
    out: dict[str, dict] = {}
    for row in db.list_model_registry():
        caps = []
        try: caps = json.loads(row["capabilities"] or "[]")
        except Exception: pass
        gguf_path = ""
        try: gguf_path = row["gguf_path"] or ""
        except Exception: pass
        out[row["id"]] = {"id": row["id"], "kind": row["kind"] or "llm", "size_bytes": row["size_bytes"] or 0, "local_bytes": row["size_bytes"] or 0, "context_tokens": row["context_tokens"] or 0, "params_billion": float(row["params"]) if row["params"] else None, "pipeline_tag": row["pipeline_tag"] or "", "capabilities": caps, "status": row["status"] or "ready", "progress": row["progress"] or 0, "error": row["error"] or "", "downloaded": True, "last_used": row["last_used"] or 0, "gguf_path": gguf_path, "backend": "llama.cpp" if gguf_path else "transformers"}
    try:
        from huggingface_hub import scan_cache_dir
        scan = scan_cache_dir(str(CACHE_ROOT))
        for repo in scan.repos:
            if getattr(repo, "repo_type", "model") != "model": continue
            repo_id = repo.repo_id; preset = PRESET_MAP.get(repo_id, {})
            if repo_id in out: entry = out[repo_id]
            else: entry = {"id": repo_id, "kind": preset.get("kind") or "llm", "size_bytes": int(repo.size_on_disk or 0), "context_tokens": preset.get("context_tokens", 0), "params_billion": preset.get("params_billion"), "pipeline_tag": "", "capabilities": preset.get("capabilities") or [], "status": "ready", "progress": 1.0, "error": "", "last_used": 0, "gguf_path": _find_gguf_file(repo_id) or "", "backend": "llama.cpp" if _find_gguf_file(repo_id) else "transformers"}
            entry["local_bytes"] = int(repo.size_on_disk or 0); entry["size_bytes"] = entry.get("size_bytes") or entry["local_bytes"]; entry["downloaded"] = True
            if entry.get("status") in (None, "", "unknown"): entry["status"] = "ready"
            out[repo_id] = entry
    except Exception as exc: print(f"[gapshap] cache scan failed: {exc}", flush=True)
    with _JOBS_LOCK:
        for repo_id, job in _JOBS.items():
            if repo_id in out and job.status in ("queued", "downloading"):
                out[repo_id]["status"] = job.status; out[repo_id]["progress"] = job.progress; out[repo_id]["downloaded"] = out[repo_id].get("local_bytes", 0) > 0
    for repo_id, entry in out.items():
        if entry.get("status") not in ("downloading", "queued") and not is_cached(repo_id): entry["status"] = "incomplete"; entry["downloaded"] = False
    roles = {"llm": CONFIG.get("model"), "vision": CONFIG.get("vision_model"), "embedding": CONFIG.get("embedding_model"), "image": CONFIG.get("image_model")}
    if CONFIG.get("active_gguf_model"): roles["llm"] = CONFIG.get("active_gguf_model")
    for entry in out.values():
        entry["active_role"] = next((k for k, v in roles.items() if v == entry["id"]), None)
        entry["note"] = PRESET_MAP.get(entry["id"], {}).get("note", "")
    return sorted(out.values(), key=lambda r: (r.get("active_role") is None, r.get("kind", ""), r["id"].lower()))

def use_model(repo_id: str, kind: str | None = None) -> dict:
    entry = next((m for m in local_models() if m["id"] == repo_id), None)
    kind = kind or (entry or {}).get("kind") or PRESET_MAP.get(repo_id, {}).get("kind") or "llm"

    gguf_path = (entry or {}).get("gguf_path", "")
    
    if gguf_path and Path(gguf_path).exists():
        from . import llama_loader
        from .config import update_config
        
        # STRICT ROUTING: Set backend to llama AND keep model safe
        update_config({
            "active_backend": "llama", 
            "active_gguf_model": repo_id,
            "active_model_path": gguf_path,
            "model": "Qwen/Qwen2.5-1.5B-Instruct" 
        })
        
        llama_loader.load(gguf_path, force=True)
        
        db.touch_model(repo_id)
        return {"ok": True, "field": "model", "model": repo_id, "backend": "llama.cpp", "path": gguf_path}
    else:
        field = {"llm": "model", "vision": "vision_model", "embedding": "embedding_model", "image": "image_model"}.get(kind)
        if not field: raise ValueError(f"Unsupported model kind: {kind}")

        from . import config as config_module
        if field == "model":
            config_module.update_config({"active_backend": "transformers", "active_gguf_model": ""})
            
        config_module.update_config({field: repo_id})

        changed = {}
        if field == "model":
            from . import llm
            llm.unload()
            llm.warmup()
            changed["llm"] = llm.info()
        if field == "embedding":
            from . import rag
            rag.unload_embedder()
        if field == "vision":
            from . import vision
            vision.unload()
        if field == "image":
            from . import imagegen
            imagegen.unload()

        db.touch_model(repo_id)
        return {"ok": True, "field": field, "model": repo_id, **changed}

def remove_model(repo_id: str) -> dict:
    active = {CONFIG.get("model"), CONFIG.get("vision_model"), CONFIG.get("embedding_model"), CONFIG.get("image_model")}
    if repo_id in active: raise ValueError("Cannot delete active model. Switch first.")
    path = cache_repo_path(repo_id)
    removed = 0
    if path.exists():
        removed = dir_size(path)
        shutil.rmtree(path, ignore_errors=True)
    with _JOBS_LOCK:
        job = _JOBS.get(repo_id)
        if job and job.status in ("queued", "downloading"): job.cancel.set()
    db.delete_model_row(repo_id)
    return {"ok": True, "repo_id": repo_id, "freed_bytes": removed}

def disk_free() -> dict:
    try:
        import psutil
        usage = psutil.disk_usage(str(MODELS_DIR))
        return {"free_gb": round(usage.free / 1024 ** 3, 2), "total_gb": round(usage.total / 1024 ** 3, 2), "models_gb": round(dir_size(CACHE_ROOT) / 1024 ** 3, 2)}
    except Exception: return {}