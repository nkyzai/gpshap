import random
import threading
import time
import uuid
from pathlib import Path

import torch

from .config import CONFIG, GENERATED_DIR
from .llm import dtype_kwargs, resolve_device


_lock = threading.RLock()
_gen_lock = threading.Lock()
_state = {"model_id": None, "pipe": None}


class Job:
    def __init__(self, prompt: str):
        self.id = uuid.uuid4().hex[:10]
        self.prompt = prompt
        self.status = "queued"
        self.progress = 0.0
        self.step = 0
        self.total_steps = 0
        self.error = ""
        self.filename = ""
        self.url = ""


JOBS: dict[str, Job] = {}


def unload() -> None:
    with _lock:
        if _state["pipe"] is not None:
            del _state["pipe"]
        _state["pipe"] = None
        _state["model_id"] = None

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load(model_id: str | None = None, force: bool = False):
    model_id = model_id or CONFIG.get("image_model") or ""
    if not model_id:
        raise RuntimeError("No image model configured")

    from . import hub
    local_path = hub.local_repo_path(model_id)
    load_source = str(local_path) if hub.is_cached(model_id) else model_id

    with _lock:
        if not force and _state["pipe"] is not None and _state["model_id"] == model_id:
            return _state["pipe"]

        unload()

        from diffusers import AutoPipelineForText2Image

        device = resolve_device()
        dtype = torch.float32 if device == "cpu" else None
        if dtype is None:
            from .llm import resolve_dtype
            dtype = resolve_dtype(device)

        kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "safety_checker": None,
            "requires_safety_checker": False,
            "local_files_only": load_source != model_id,
            "variant": "fp16" if (dtype == torch.float16 and device != "cpu") else None,
        }
        kwargs.update(dtype_kwargs(dtype))
        kwargs = {k: v for k, v in kwargs.items() if v is not None or k in
                  ("safety_checker", "requires_safety_checker")}

        try:
            pipe = AutoPipelineForText2Image.from_pretrained(load_source, **kwargs)
        except Exception:
            kwargs.pop("variant", None)
            pipe = AutoPipelineForText2Image.from_pretrained(load_source, **kwargs)

        pipe = pipe.to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        _state.update({"model_id": model_id, "pipe": pipe})
        print(f"[gapshap] image engine ready: {model_id} on {device}", flush=True)
        return pipe


def info() -> dict:
    return {
        "model_id": _state["model_id"] or CONFIG.get("image_model"),
        "loaded": _state["pipe"] is not None,
        "device": resolve_device(),
        "cpu_warning": resolve_device() == "cpu",
    }


def generate(prompt: str, negative: str = "", steps: int | None = None,
             width: int | None = None, height: int | None = None,
             guidance: float | None = None, seed: int | None = None,
             model_id: str | None = None, job: Job | None = None) -> dict:
    with _gen_lock:
        pipe = load(model_id)

        steps = int(steps or CONFIG.get("image_steps", 4))
        width = int(width or CONFIG.get("image_width", 512))
        height = int(height or CONFIG.get("image_height", 512))
        guidance = float(guidance if guidance is not None else CONFIG.get("image_guidance", 0.0))
        negative = negative if negative is not None else CONFIG.get("image_negative", "")
        seed = int(seed) if seed is not None else random.randint(1, 2 ** 31 - 1)

        width = max(256, min(2048, width // 8 * 8))
        height = max(256, min(2048, height // 8 * 8))
        steps = max(1, min(150, steps))

        if job:
            job.status = "running"
            job.total_steps = steps

        def _cb(pipe_obj, step_index, timestep, callback_kwargs):
            if job:
                job.step = step_index + 1
                job.progress = min(0.99, (step_index + 1) / steps)
            return callback_kwargs

        generator = torch.Generator(device=pipe.device).manual_seed(seed)

        call_kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "generator": generator,
            "callback_on_step_end": _cb,
        }
        if negative:
            call_kwargs["negative_prompt"] = negative

        try:
            result = pipe(**call_kwargs)
        except TypeError:
            call_kwargs.pop("callback_on_step_end", None)
            result = pipe(**call_kwargs)

        image = result.images[0]
        filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        path = GENERATED_DIR / filename
        image.save(path)

        if job:
            job.progress = 1.0
            job.status = "done"
            job.filename = filename
            job.url = f"/generated/{filename}"

        return {
            "filename": filename,
            "url": f"/generated/{filename}",
            "path": str(path),
            "prompt": prompt,
            "negative": negative,
            "steps": steps,
            "width": width,
            "height": height,
            "seed": seed,
            "model": _state["model_id"],
        }