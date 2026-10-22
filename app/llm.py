import gc
import threading
import time
from typing import Iterator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from .config import CONFIG


_state = {
    "model_id": None,
    "model": None,
    "tokenizer": None,
    "loaded_at": None,
    "params": 0,
}
_lock = threading.RLock()
_gen_lock = threading.Lock()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _tf_version() -> tuple[int, int]:
    import transformers
    try:
        major, minor = transformers.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return 0, 0


def dtype_kwargs(dtype) -> dict:
    """transformers >= 4.56 renamed torch_dtype -> dtype."""
    major, minor = _tf_version()
    if (major, minor) >= (4, 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def resolve_device() -> str:
    pref = str(CONFIG.get("device", "auto")).lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str):
    pref = str(CONFIG.get("dtype", "auto")).lower()
    if pref == "float16":
        return torch.float16
    if pref == "bfloat16":
        return torch.bfloat16
    if pref == "float32":
        return torch.float32
    if device.startswith("cuda"):
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16
    return torch.float32


class _StopFlag(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


# --------------------------------------------------------------------------
# load / unload
# --------------------------------------------------------------------------
def unload() -> None:
    with _lock:
        if _state["model"] is not None:
            del _state["model"]
        _state["model"] = None
        _state["tokenizer"] = None
        _state["model_id"] = None
        _state["params"] = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load(model_id: str | None = None, force: bool = False):
    model_id = model_id or CONFIG["model"]

    with _lock:
        if not force and _state["model"] is not None and _state["model_id"] == model_id:
            return _state["model"], _state["tokenizer"]

        if _state["model"] is not None:
            unload()

        device = resolve_device()
        dtype = resolve_dtype(device)

        print(f"[gapshap] loading {model_id} on {device} ({dtype})", flush=True)
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        load_kwargs.update(dtype_kwargs(dtype))
        if device.startswith("cuda"):
            load_kwargs["device_map"] = "auto"

        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except TypeError:
            load_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            model = model.to(device)

        if not device.startswith("cuda"):
            model = model.to(device)

        model.eval()

        params = sum(p.numel() for p in model.parameters())
        _state.update({
            "model_id": model_id,
            "model": model,
            "tokenizer": tokenizer,
            "loaded_at": time.time(),
            "params": params,
        })

        print(f"[gapshap] loaded {model_id} in {time.time() - t0:.1f}s "
              f"({params / 1e9:.2f}B params)", flush=True)
        return model, tokenizer


def ensure():
    return load(CONFIG["model"])


def warmup() -> None:
    try:
        load(CONFIG["model"])
    except Exception as exc:
        print(f"[gapshap] LLM warmup failed: {exc}", flush=True)


def info() -> dict:
    model = _state["model"]
    data = {
        "loaded": model is not None,
        "model_id": _state["model_id"] or CONFIG["model"],
        "configured_model": CONFIG["model"],
        "params_billion": round(_state["params"] / 1e9, 3) if _state["params"] else None,
        "device": resolve_device(),
        "cuda_available": torch.cuda.is_available(),
    }
    if model is not None:
        data["dtype"] = str(next(model.parameters()).dtype).replace("torch.", "")
        data["context_tokens"] = context_length()
    if torch.cuda.is_available():
        data["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        data["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
        try:
            data["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return data


def context_length() -> int:
    model = _state["model"]
    if model is None:
        return 0
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "model_max_length", "max_sequence_length",
                 "seq_length", "n_positions"):
        val = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(val, int) and val > 0:
            return val
    return 0


# --------------------------------------------------------------------------
# prompt building
# --------------------------------------------------------------------------
def build_prompt(messages: list[dict], tools: list[dict] | None = None) -> str:
    _model, tokenizer = ensure()

    if tools:
        try:
            return tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = []
        for message in messages:
            parts.append(f"<|im_start|>{message.get('role', 'user')}\n"
                         f"{message.get('content', '')}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def _trim(ids: torch.Tensor, max_input: int) -> torch.Tensor:
    if ids.dim() == 2 and ids.shape[1] > max_input > 0:
        return ids[:, -max_input:]
    return ids


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def generate_stream(
    messages: list[dict],
    tools: list[dict] | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stop: list[str] | None = None,
    images: list | None = None,
) -> Iterator[str]:
    """Yield text deltas. Blocking; call from a worker thread."""
    model, tokenizer = ensure()

    prompt = build_prompt(messages, tools=tools)
    max_new_tokens = int(max_new_tokens or CONFIG.get("max_new_tokens", 768))
    temperature = float(CONFIG.get("temperature", 0.7)) if temperature is None else temperature
    top_p = float(CONFIG.get("top_p", 0.9)) if top_p is None else top_p
    top_k = int(CONFIG.get("top_k", 50))
    rep = float(CONFIG.get("repetition_penalty", 1.08))

    inputs = tokenizer(prompt, return_tensors="pt")
    ctx = context_length() or 8192
    inputs["input_ids"] = _trim(inputs["input_ids"], max(512, ctx - max_new_tokens - 8))
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"][:, -inputs["input_ids"].shape[-1]:]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    stop_event = threading.Event()
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(inputs)
    gen_kwargs.update({
        "max_new_tokens": max_new_tokens,
        "streamer": streamer,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "stopping_criteria": StoppingCriteriaList([_StopFlag(stop_event)]),
    })
    if temperature > 0:
        gen_kwargs.update({"temperature": max(temperature, 1e-3), "top_p": top_p, "top_k": top_k})
    if rep and rep != 1.0:
        gen_kwargs["repetition_penalty"] = rep

    def _run():
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except Exception as exc:
            print(f"[gapshap] generate error: {exc}", flush=True)
        finally:
            stop_event.set()

    with _gen_lock:
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        stop_strings = [s for s in (stop or []) if s]
        acc = ""
        for piece in streamer:
            if not piece:
                continue
            acc += piece

            hit = None
            for token in stop_strings:
                idx = acc.find(token)
                if idx != -1:
                    hit = idx
                    break

            if hit is not None:
                remaining = acc[:hit]
                if remaining:
                    yield remaining
                stop_event.set()
                break

            yield piece
            acc = ""

        thread.join(timeout=5)


def chat(messages: list[dict], tools: list[dict] | None = None, stop: list[str] | None = None,
         max_new_tokens: int | None = None) -> str:
    return "".join(generate_stream(messages, tools=tools, stop=stop,
                                   max_new_tokens=max_new_tokens))