import os
import threading
import time
from pathlib import Path
from typing import Iterator

from .config import CONFIG


_state = {
    "model_id": None,
    "llm": None,
    "loaded_at": None,
}
_lock = threading.RLock()
_gen_lock = threading.Lock()


def _get_n_gpu_layers() -> int:
    """Auto-detect VRAM usage for offloading layers."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        
        # Get total VRAM in MB
        vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        
        # Heuristic for RTX 3090 (24GB): Offload all
        if vram_mb > 20000:
            return 999
        elif vram_mb > 12000:
            return 40
        elif vram_mb > 8000:
            return 20
        else:
            return 0
            
    except Exception:
        return 0


def _get_n_ctx() -> int:
    """Determine optimal context length based on config or model size."""
    # Try explicit config first
    ctx = CONFIG.get("context_length")
    if isinstance(ctx, int) and ctx > 0:
        return ctx
    
    # Default to 8192 for modern models, which balances quality and speed
    return 8192


def _get_n_threads() -> int:
    """Optimize thread count for CPU inference fallback."""
    try:
        cpu_count = os.cpu_count() or 4
        # Use half the logical cores, minimum 2
        return max(2, cpu_count // 2)
    except Exception:
        return 4


def load(model_path: str, force: bool = False):
    """Load a GGUF model from disk."""
    with _lock:
        # Safety check: Does the file exist?
        path_obj = Path(model_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"GGUF model not found at: {model_path}")
        
        if not force and _state["llm"] is not None and _state["model_id"] == str(path_obj.resolve()):
            return _state["llm"]

        unload()

        from llama_cpp import Llama

        print(f"[gapshap] Loading GGUF: {path_obj.name}", flush=True)
        
        n_ctx = _get_n_ctx()
        n_batch = 512
        n_threads = _get_n_threads()
        n_gpu_layers = _get_n_gpu_layers()
        
        device = "cuda" if n_gpu_layers > 0 else "cpu"

        try:
            llm = Llama(
                model_path=str(path_obj),
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                verbose=False,
                chat_format=None,  # We handle templating manually for safety
            )
        except Exception as exc:
            print(f"[gapshap] Failed to load GGUF {path_obj.name}: {exc}", flush=True)
            raise RuntimeError(f"Failed to load GGUF model: {exc}") from exc

        resolved_path = str(path_obj.resolve())
        
        _state.update({
            "model_id": resolved_path,
            "llm": llm,
            "loaded_at": time.time(),
        })

        print(f"[gapshap] GGUF loaded on {device} ({n_gpu_layers} layers offloaded, ctx={n_ctx})", flush=True)
        return llm


def unload() -> None:
    with _lock:
        if _state["llm"] is not None:
            del _state["llm"]
        _state["llm"] = None
        _state["model_id"] = None
        
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def generate_stream(
    messages: list[dict],
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stop: list[str] | None = None,
    tools: list[dict] | None = None,  # Accepted for signature compatibility, ignored for GGUF
) -> Iterator[str]:
    """Yield text deltas from a GGUF model."""
    llm = _state["llm"]
    if llm is None:
        raise RuntimeError("GGUF model not loaded")

    max_new_tokens = max_new_tokens or int(CONFIG.get("max_new_tokens", 512))
    temperature = temperature if temperature is not None else float(CONFIG.get("temperature", 0.7))
    top_p = top_p if top_p is not None else float(CONFIG.get("top_p", 0.9))
    
    # Simple ChatML-like prompt construction for generic compatibility
    # Most modern GGUFs expect this or similar. 
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"<|{role}|>\n{content}\n")
    prompt_parts.append("<|assistant|>\n")
    prompt = "".join(prompt_parts)

    stop_seq = stop or ["</s>", "<|user|>", "<|end|>"]

    with _gen_lock:
        stream = llm(
            prompt,
            max_tokens=max_new_tokens,
            temp=temperature,
            top_p=top_p,
            repeat_penalty=1.1,
            stream=True,
            stop=stop_seq,
        )

        for chunk in stream:
            token_text = chunk["choices"][0].get("text", "")
            if token_text:
                yield token_text


def chat(messages: list[dict], **kwargs) -> str:
    return "".join(generate_stream(messages, **kwargs))