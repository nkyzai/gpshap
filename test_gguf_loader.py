"""Mocked smoke test for GGUF routing; no model weights are loaded."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import llama_cpp

from app import config, llm


def main() -> None:
    if not callable(getattr(llama_cpp, "llama_supports_gpu_offload", None)):
        raise RuntimeError("llama-cpp-python native runtime is unavailable")
    llama_cpp.llama_supports_gpu_offload()

    config_keys = (
        "active_backend",
        "active_model_path",
        "active_gguf_model",
        "gguf_n_ctx",
        "gguf_n_gpu_layers",
    )
    previous_config = {key: config.CONFIG.get(key) for key in config_keys}

    try:
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "mock-model.gguf"
            model_path.touch()
            fake_model = object()

            config.CONFIG.update({
                "active_backend": "llama",
                "active_model_path": str(model_path),
                "active_gguf_model": model_path.name,
                "gguf_n_ctx": 512,
                "gguf_n_gpu_layers": 0,
            })

            with patch.object(llama_cpp, "Llama", return_value=fake_model) as llama:
                model, tokenizer = llm.load_gguf_model()

            if model is not fake_model or tokenizer is not None:
                raise AssertionError("GGUF loader returned unexpected model state")
            llama.assert_called_once_with(
                model_path=str(model_path.resolve()),
                n_ctx=512,
                n_gpu_layers=0,
                verbose=False,
            )
            print("GGUF Backend Ready")
    finally:
        llm.unload()
        config.CONFIG.update(previous_config)


if __name__ == "__main__":
    main()
