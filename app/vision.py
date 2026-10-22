import threading
from pathlib import Path

import torch
from PIL import Image

from .config import CONFIG
from .llm import dtype_kwargs, resolve_device


_lock = threading.RLock()
_state = {"model_id": None, "model": None, "processor": None, "flavour": None}


def _is_florence(model_id: str) -> bool:
    return "florence" in model_id.lower()


def load(model_id: str | None = None, force: bool = False):
    model_id = model_id or CONFIG.get("vision_model") or ""
    if not model_id:
        raise RuntimeError("No vision model configured")

    with _lock:
        if not force and _state["model"] is not None and _state["model_id"] == model_id:
            return _state["model"], _state["processor"], _state["flavour"]

        unload()

        from transformers import AutoProcessor

        device = resolve_device()
        flavour = "florence" if _is_florence(model_id) else "vlm"

        # Florence-2 remote code is safest in fp32; VLMs use the configured dtype.
        if flavour == "florence":
            dtype = torch.float32
        else:
            from .llm import resolve_dtype
            dtype = resolve_dtype(device)

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        kwargs.update(dtype_kwargs(dtype))

        model = None
        for factory in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                        "AutoModelForCausalLM"):
            try:
                cls = getattr(__import__("transformers", fromlist=[factory]), factory)
                model = cls.from_pretrained(model_id, **kwargs)
                break
            except Exception:
                continue

        if model is None:
            raise RuntimeError(f"Could not load vision model {model_id}")

        model = model.to(device)
        model.eval()

        _state.update({"model_id": model_id, "model": model,
                       "processor": processor, "flavour": flavour})
        print(f"[gapshap] vision engine ready: {model_id} ({flavour}, {device})", flush=True)
        return model, processor, flavour


def unload() -> None:
    with _lock:
        if _state["model"] is not None:
            del _state["model"]
        _state["model"] = None
        _state["processor"] = None
        _state["model_id"] = None
        _state["flavour"] = None

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def is_ready() -> bool:
    from . import hub
    return bool(hub.is_cached(CONFIG.get("vision_model") or ""))


def info() -> dict:
    return {
        "model_id": _state["model_id"] or CONFIG.get("vision_model"),
        "loaded": _state["model"] is not None,
        "flavour": _state["flavour"],
        "downloaded": is_ready(),
        "mode": CONFIG.get("vision_mode", "auto"),
    }


def _open(path) -> Image.Image:
    image = Image.open(path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def _florence_describe(image: Image.Image) -> dict:
    model, processor, _ = load()
    out = {}

    tasks = [("<OCR>", "ocr"), ("<MORE_DETAILED_CAPTION>", "caption")]
    for task, key in tasks:
        try:
            inputs = processor(text=task, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=512 if key == "ocr" else 128,
                                     num_beams=3)
            raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                raw, task=task, image_size=(image.width, image.height)
            )
            if isinstance(parsed, dict):
                value = parsed.get(task) or next(iter(parsed.values()))
            else:
                value = parsed

            if isinstance(value, dict):
                value = value.get("text") or ", ".join(str(v) for v in value.values())
            out[key] = str(value or "").strip()
        except Exception as exc:
            out[key] = ""
            out[f"{key}_error"] = str(exc)

    return out


def _vlm_describe(image: Image.Image, question: str) -> dict:
    model, processor, _ = load()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]

    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    except Exception:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")

    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=512)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
    return {"answer": answer.strip(), "caption": answer.strip(), "ocr": ""}


def analyze(image_path, question: str | None = None) -> dict:
    """Return a structured description of an image."""
    model_id = CONFIG.get("vision_model") or ""
    mode = str(CONFIG.get("vision_mode", "auto")).lower()

    if mode == "off":
        return {"skipped": True, "reason": "vision disabled"}

    path = Path(image_path)
    if not path.exists():
        return {"error": f"image not found: {path}"}

    image = _open(path)
    flavour = "florence" if _is_florence(model_id) else "vlm"

    if mode == "auto":
        mode = flavour

    result = {"model": model_id, "mode": mode,
              "width": image.width, "height": image.height}

    try:
        if mode == "caption" or flavour == "florence":
            parts = _florence_describe(image)
        else:
            parts = _vlm_describe(image, question or
                                  "Describe this image in detail, including any visible text.")

        result.update(parts)

        if flavour == "vlm" and question:
            result["question"] = question
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["text"] = build_text(result)
    return result


def build_text(result: dict) -> str:
    chunks = []
    if result.get("caption"):
        chunks.append(f"Caption: {result['caption']}")
    if result.get("answer") and result.get("answer") != result.get("caption"):
        chunks.append(f"Analysis: {result['answer']}")
    if result.get("ocr"):
        chunks.append(f"Text found in image: {result['ocr']}")
    return "\n".join(chunks)


def ask(image_paths: list, question: str) -> dict:
    """Multi-modal Q&A over one or more images."""
    model_id = CONFIG.get("vision_model") or ""

    if not _is_florence(model_id):
        try:
            image = _open(image_paths[0])
            return {"answer": _vlm_describe(image, question).get("answer", ""),
                    "engine": model_id, "mode": "vlm"}
        except Exception as exc:
            return {"error": str(exc), "engine": model_id}

    # Florence cannot chat - describe + OCR, then let the LLM reason over the text.
    blobs = []
    for p in image_paths[:3]:
        res = analyze(p)
        blobs.append(f"[{Path(p).name}] {res.get('text', '')}")

    from . import llm
    prompt = (
        "You are a vision assistant. Using ONLY the machine-generated image analysis below, "
        "answer the user's question. If the analysis lacks the answer, say what is missing.\n\n"
        f"IMAGE ANALYSIS:\n" + "\n\n".join(blobs) + f"\n\nQUESTION: {question}\n\nANSWER:"
    )
    answer = llm.chat([{"role": "user", "content": prompt}])
    return {"answer": answer, "engine": model_id, "mode": "caption+llm",
            "evidence": blobs}