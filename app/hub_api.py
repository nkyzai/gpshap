"""
Direct Hugging Face Hub REST client for gapshap.

Deliberately uses requests instead of huggingface_hub so the browser keeps
working regardless of hub library version. Supports HF_ENDPOINT mirrors.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

from .config import CONFIG, DATA_DIR


DEFAULT_ENDPOINT = "https://huggingface.co"
CACHE_PATH = DATA_DIR / "hub_metadata_cache.json"

IGNORE_PATTERNS = [
    "*.msgpack", "*.h5", "*.ot", "*.onnx", "*.tflite", "*.pb",
    "flax_model*", "original/*", "*.pth", "*.ckpt", "*.ipynb",
]
WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".onnx", ".tflite")

_lock = threading.RLock()
_cache: dict = {}
_cache_loaded = False
_cache_dirty = False


class HubError(RuntimeError):
    """Actionable Hub failure. Never silently swallowed by the UI."""


# --------------------------------------------------------------------------
# configuration / transport
# --------------------------------------------------------------------------
def endpoint() -> str:
    return (
        os.environ.get("HF_ENDPOINT")
        or os.environ.get("GAPSHAP_HF_ENDPOINT")
        or DEFAULT_ENDPOINT
    ).rstrip("/")


def token() -> str | None:
    return (CONFIG.get("hf_token") or os.environ.get("HF_TOKEN") or "").strip() or None


def headers() -> dict:
    h = {
        "Accept": "application/json",
        "User-Agent": "gapshap/2.1 (local-windows-agent)",
    }
    t = token()
    if t:
        h["Authorization"] = f"Bearer {t}"
    return h


def _get(url: str, params: dict | None = None, timeout=(8, 30)) -> requests.Response:
    try:
        r = requests.get(url, params=params, headers=headers(), timeout=timeout)
    except requests.exceptions.SSLError as exc:
        raise HubError(
            f"TLS error contacting {endpoint()}. Check your clock/certificates, or set "
            f"HF_ENDPOINT=https://hf-mirror.com in run.bat. ({exc})"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise HubError(
            f"Timed out contacting {endpoint()}. Your network may block Hugging Face; "
            f"set HF_ENDPOINT=https://hf-mirror.com in run.bat."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise HubError(
            f"Cannot reach {endpoint()}. Check internet/fireline/proxy, or use "
            f"HF_ENDPOINT=https://hf-mirror.com in run.bat. ({exc})"
        ) from exc

    if r.status_code in (401, 403):
        raise HubError(
            f"Hub returned {r.status_code}. The repo may be gated/private, or your token "
            f"is missing/invalid. Accept the licence on huggingface.co and paste a read "
            f"token in gapshap Settings -> HuggingFace."
        )
    if r.status_code == 429:
        raise HubError("Hugging Face rate-limited this request. Wait ~60s and press Refresh.")
    if r.status_code >= 400:
        raise HubError(f"Hub returned HTTP {r.status_code} for {url}")
    return r


def _json(path: str, params: dict | None = None) -> dict | list:
    r = _get(endpoint() + path, params)
    try:
        return r.json()
    except ValueError as exc:
        raise HubError(f"Hub returned non-JSON data from {path}") from exc


def _text(path: str, timeout=(6, 20)) -> str:
    r = _get(endpoint() + path, timeout=timeout)
    return r.text


# --------------------------------------------------------------------------
# metadata cache (24h) - avoids hammering the Hub on every keystroke
# --------------------------------------------------------------------------
def _load_cache() -> None:
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    with _lock:
        if _cache_loaded:
            return
        try:
            if CACHE_PATH.exists():
                _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                if not isinstance(_cache, dict):
                    _cache = {}
        except Exception:
            _cache = {}
        _cache_loaded = True


def _cached(repo_id: str) -> dict | None:
    _load_cache()
    with _lock:
        entry = _cache.get(repo_id)
        if not entry:
            return None
        if time.time() - float(entry.get("ts", 0)) > 86400:
            return None
        return entry


def _store(repo_id: str, data: dict) -> None:
    global _cache_dirty
    _load_cache()
    with _lock:
        _cache[repo_id] = {"ts": time.time(), **data}
        _cache_dirty = True


def flush_cache() -> None:
    global _cache_dirty
    with _lock:
        if not _cache_dirty:
            return
        try:
            tmp = CACHE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")
            tmp.replace(CACHE_PATH)
            _cache_dirty = False
        except Exception:
            pass


# --------------------------------------------------------------------------
# classification helpers
# --------------------------------------------------------------------------
def _param_billions(repo_id: str, tags: set, safetensors: dict | None) -> float | None:
    if safetensors:
        total = safetensors.get("total")
        if isinstance(total, (int, float)) and total > 0:
            return round(total / 1e9, 3)

    low = repo_id.lower()
    patterns = [
        r"-(\d+(?:\.\d+)?)([bm])(?:-|_|$|\.|/)",
        r"[_ ](\d+(?:\.\d+)?)([bm])(?:-|_|$|\.|/)",
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            value = float(m.group(1))
            if m.group(2) == "m":
                value /= 1000.0
            if 0.01 <= value <= 2000:
                return round(value, 3)
    return None


def _context_from(cfg: dict, tags: set) -> int:
    if isinstance(cfg, dict):
        for key in ("max_position_embeddings", "model_max_length", "n_positions",
                    "seq_length", "max_sequence_length", "sliding_window"):
            val = cfg.get(key)
            if isinstance(val, int) and 0 < val <= 10_000_000:
                return val
        text_cfg = cfg.get("text_config")
        if isinstance(text_cfg, dict):
            val = text_cfg.get("max_position_embeddings")
            if isinstance(val, int) and val > 0:
                return val

    for tag in tags:
        m = re.search(r"context[:=\s-]*(\d+)", str(tag), re.I)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)k[-_ ]?context", str(tag), re.I)
        if m:
            return int(m.group(1)) * 1000
    return 0


def _kind(pipeline: str, tags: set, repo_id: str, arch: str) -> str:
    low = repo_id.lower()
    tagstr = " ".join(t.lower() for t in tags)

    if "diffusers" in tagstr or pipeline in ("text-to-image", "image-to-image", "inpainting"):
        return "image"
    if pipeline in ("feature-extraction", "sentence-similarity") or \
       "sentence-transformers" in tagstr or \
       any(k in low for k in ("bge-", "e5-", "minilm", "gte-", "embeddings", "nomic-embed")):
        return "embedding"
    if pipeline in ("image-text-to-text", "visual-question-answering", "image-to-text") or \
       any(k in low for k in ("-vl", "vl-", "llava", "minicpm-v", "idefics", "pixtral",
                              "florence", "kosmos", "cogvlm", "internvl")) or \
       any(k in arch.lower() for k in ("vision", "llava", "idefics", "florence")):
        return "vision"
    if pipeline in ("automatic-speech-recognition",):
        return "audio"
    return "llm"


def _format_of(tags: set, safetensors: dict | None, arch: str,
               files: list[str]) -> str:
    tagstr = " ".join(t.lower() for t in tags)
    if "gguf" in tagstr or any(f.lower().endswith(".gguf") for f in files):
        return "gguf"
    if "diffusers" in tagstr or any("unet/" in f or "vae/" in f for f in files):
        return "diffusers"
    if "onnx" in tagstr:
        return "onnx"
    if safetensors or arch or any(f.endswith(".safetensors") for f in files):
        return "transformers"
    return "other"


def _capabilities(repo_id: str, tags: set, pipeline: str, kind: str,
                  context: int, params: float | None, fmt: str) -> list[str]:
    low = repo_id.lower()
    tagstr = " ".join(t.lower() for t in tags)
    caps = set()

    if kind == "embedding":
        caps.add("embeddings")
    if kind == "image":
        caps.add("image-generation")
    if kind == "vision":
        caps.add("vision")
    if "florence" in low or "ocr" in tagstr or "docling" in low:
        caps.update({"ocr", "captioning"})
    if any(k in low for k in ("instruct", "-it", "chat", "hermes", "wizard", "zephyr")) or \
       "conversational" in tagstr or "chatml" in tagstr:
        caps.add("chat")
    if any(k in low for k in ("coder", "code-", "starcoder", "deepseek-coder")) or \
       "code" in tagstr:
        caps.add("code")
    if any(k in tagstr for k in ("function-calling", "tool-use", "tools")) or \
       any(k in low for k in ("hermes", "firefunction", "qwen2.5", "qwen3", "llama-3.1",
                              "llama-3.2", "mistral-nemo")):
        caps.add("tools")
    if context and context >= 32768:
        caps.add("long-context")
    if any(k in low for k in ("reasoning", "thinker", "r1-", "qwq", "deepseek-r")):
        caps.add("reasoning")
    if "multilingual" in tagstr or "multilingual" in low:
        caps.add("multilingual")
    if fmt == "gguf" or any(k in low for k in ("awq", "gptq", "bnb", "4bit", "8bit", "int4")):
        caps.add("quantized")
    if fmt == "gguf":
        caps.add("gguf")
    if params:
        if params <= 2:
            caps.update({"cpu-friendly", "fast"})
        elif params >= 8:
            caps.add("gpu-required")
        else:
            caps.add("balanced")
    if "gated" in tagstr:
        caps.add("gated")

    order = ["chat", "tools", "vision", "ocr", "image-generation", "embeddings", "code",
             "reasoning", "long-context", "multilingual", "quantized", "gguf", "captioning",
             "cpu-friendly", "fast", "balanced", "gpu-required", "gated"]
    return [c for c in order if c in caps]


def _estimate_bytes(params: float | None, kind: str, tags: set, repo_id: str) -> int:
    if params:
        blob = (repo_id + " " + " ".join(tags)).lower()
        per_param = 2.0
        if any(k in blob for k in ("q4", "4bit", "awq", "gptq", "int4", "q3")):
            per_param = 0.55
        elif any(k in blob for k in ("q5", "q6")):
            per_param = 0.8
        elif any(k in blob for k in ("q8", "8bit", "int8", "fp8")):
            per_param = 1.05
        elif "fp32" in blob or "float32" in blob:
            per_param = 4.0
        return int(params * 1e9 * per_param)
    if kind == "embedding":
        return int(0.13 * 1024 ** 3)
    if kind == "image":
        return int(4.9 * 1024 ** 3)
    return 0


# --------------------------------------------------------------------------
# public: live search
# --------------------------------------------------------------------------
def search_models(search: str = "", task: str = "all", sort: str = "downloads",
                  limit: int = 40) -> list[dict]:
    params: dict = {"limit": max(1, min(int(limit), 100)), "full": "true"}

    if search.strip():
        params["search"] = search.strip()

    real_tasks = {"text-generation", "image-text-to-text", "feature-extraction",
                  "sentence-similarity", "text-to-image", "image-to-image",
                  "automatic-speech-recognition", "zero-shot-image-classification"}
    if task and task != "all" and task in real_tasks:
        params["filter"] = task

    # "size" is sorted locally after metadata enrichment (Hub has no size sort)
    hub_sort = sort if sort in ("downloads", "likes", "lastModified", "createdAt",
                                "trendingScore") else "downloads"
    params["sort"] = hub_sort
    params["direction"] = -1

    data = _json("/api/models", params)
    if isinstance(data, dict):
        data = data.get("items") or data.get("models") or []
    if not isinstance(data, list):
        raise HubError("Unexpected Hugging Face response format (expected a list of models).")

    return [d for d in data if isinstance(d, dict) and (d.get("id") or d.get("modelId"))]


def _siblings(repo_id: str) -> list[dict]:
    data = _json(f"/api/models/{quote(repo_id, safe='/')}", {"blobs": "true"})
    if not isinstance(data, dict):
        return []
    out = []
    for sib in data.get("siblings") or []:
        name = sib.get("rfilename") or sib.get("name") or ""
        if name:
            out.append({"name": name, "size": int(sib.get("size") or 0)})
    return out


def _one(raw: dict, deep: bool = False) -> dict:
    repo_id = raw.get("id") or raw.get("modelId") or ""
    tags = set(raw.get("tags") or [])
    pipeline = raw.get("pipeline_tag") or ""
    library = raw.get("library_name") or ""
    safetensors = raw.get("safetensors") or None
    api_config = raw.get("config") or {}
    arch = ""
    architectures = api_config.get("architectures") or []
    if architectures:
        arch = architectures[0]

    cached = None if deep else _cached(repo_id)

    config = {}
    context = _context_from(api_config, tags)
    files = []
    exact_bytes = 0

    if cached:
        config = cached.get("config") or {}
        context = cached.get("context") or context
        files = cached.get("files") or []
        exact_bytes = cached.get("bytes") or 0
    else:
        # config.json is tiny; gives the REAL context window for new models
        try:
            text = _text(f"/{quote(repo_id, safe='/')}/resolve/main/config.json")
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                config = parsed
                if not arch and parsed.get("architectures"):
                    arch = parsed["architectures"][0]
                context = _context_from(parsed, tags) or context
        except Exception:
            pass  # GGUF / diffusers repos often have no root config.json

        if deep:
            try:
                siblings = _siblings(repo_id)
                files = [s["name"] for s in siblings]
                exact_bytes = sum(
                    s["size"] for s in siblings
                    if s["name"].lower().endswith(WEIGHT_SUFFIXES)
                    and not any(fnmatch.fnmatch(s["name"], p) for p in IGNORE_PATTERNS)
                )
            except HubError:
                pass

        _store(repo_id, {"config": config, "context": context,
                         "files": files[:120], "bytes": exact_bytes})

    if not context and config:
        context = _context_from(config, tags)

    params_b = _param_billions(repo_id, tags, safetensors)
    kind = _kind(pipeline, tags, repo_id, arch)
    fmt = _format_of(tags, safetensors, arch, files)

    size = exact_bytes if exact_bytes else int(raw.get("usedStorage") or 0)
    estimated = not bool(exact_bytes)
    if not size:
        size = _estimate_bytes(params_b, kind, tags, repo_id)

    capabilities = _capabilities(repo_id, tags, pipeline, kind, context, params_b, fmt)
    if raw.get("gated"):
        if "gated" not in capabilities:
            capabilities.append("gated")

    runnable = fmt in ("transformers", "diffusers")

    return {
        "id": repo_id,
        "kind": kind,
        "format": fmt,
        "runnable": runnable,
        "params_billion": params_b,
        "size_bytes": size,
        "size_estimated": estimated,
        "size_exact": not estimated,
        "context_tokens": context,
        "capabilities": capabilities,
        "pipeline_tag": pipeline,
        "architecture": arch,
        "library": library,
        "likes": int(raw.get("likes") or 0),
        "downloads": int(raw.get("downloads") or 0),
        "last_modified": str(raw.get("lastModified") or raw.get("last_modified") or "")[:10],
        "created_at": str(raw.get("createdAt") or "")[:10],
        "gated": bool(raw.get("gated")),
        "source": "hub",
        "downloaded": False,
        "note": (
            "GGUF - needs a llama.cpp runtime, not transformers"
            if fmt == "gguf" else
            "Diffusion pipeline" if kind == "image" else ""
        ),
        "url": f"{endpoint()}/{repo_id}",
        "files": files[:60],
    }


def enrich_many(raw_models: list[dict], deep: bool = False,
                workers: int = 4) -> list[dict]:
    if not raw_models:
        return []

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        futures = {pool.submit(_one, raw, deep): raw for raw in raw_models}
        for fut in as_completed(futures):
            raw = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                # One bad repo must never wipe out the whole search
                results.append({
                    "id": raw.get("id") or raw.get("modelId") or "unknown",
                    "kind": "llm", "format": "other", "runnable": False,
                    "params_billion": None, "size_bytes": 0, "size_estimated": True,
                    "context_tokens": 0, "capabilities": [], "pipeline_tag": "",
                    "likes": int(raw.get("likes") or 0),
                    "downloads": int(raw.get("downloads") or 0),
                    "gated": bool(raw.get("gated")), "source": "hub",
                    "downloaded": False, "note": f"Metadata incomplete: {exc}",
                    "files": [],
                })

    flush_cache()

    order = {r["id"]: i for i, r in enumerate(raw_models)}
    results.sort(key=lambda r: order.get(r["id"], 10 ** 6))
    return results


def describe(repo_id: str, deep: bool = True) -> dict:
    raw = _json(f"/api/models/{quote(repo_id, safe='/')}")
    if not isinstance(raw, dict):
        raise HubError(f"Unexpected metadata response for {repo_id}")
    record = _one(raw, deep=deep)
    record["source"] = "hub"
    return record


def diagnostics() -> dict:
    import platform
    checks = []

    for label, kwargs in (
        ("connectivity", {"search": "", "task": "all", "limit": 1}),
        ("search: qwen", {"search": "qwen", "task": "all", "limit": 5}),
        ("search: qwen3", {"search": "qwen3", "task": "all", "limit": 5}),
    ):
        t0 = time.time()
        try:
            rows = search_models(**kwargs)
            checks.append({
                "check": label, "ok": True, "count": len(rows),
                "seconds": round(time.time() - t0, 2),
                "samples": [r.get("id") or r.get("modelId") for r in rows[:5]],
            })
        except Exception as exc:
            checks.append({
                "check": label, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    versions = {}
    for name in ("transformers", "huggingface_hub", "tokenizers", "diffusers"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"IMPORT FAILED: {exc}"

    return {
        "endpoint": endpoint(),
        "token_set": bool(token()),
        "python": platform.python_version(),
        "versions": versions,
        "checks": checks,
        "cache_entries": len(_cached.__globals__.get("_cache", {})),
        "advice": [
            "Search matches repository IDs. Try: qwen, llama, gemma, deepseek, phi, mistral, hermes.",
            "Set Task = All and sort = Recently updated to find brand-new releases.",
            "GGUF entries need a llama.cpp runtime; gapshap's current loader uses transformers/safetensors.",
            "If connectivity fails, add: set \"HF_ENDPOINT=https://hf-mirror.com\" to run.bat.",
            "huggingface_hub must be >=1.5,<2.0 for transformers 5.x.",
        ],
    }