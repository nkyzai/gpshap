import os
from typing import Optional, Any

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelLoader:
    @staticmethod
    def load_model(model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1) -> Any:
        """
        Detects if the model is a GGUF file or standard HF directory and loads accordingly.
        """
        if os.path.isfile(model_path) and model_path.endswith(".gguf"):
            if not HAS_LLAMA_CPP:
                raise ImportError("llama-cpp-python is required to load GGUF models. Run: pip install llama-cpp-python")
            
            print(f"Loading GGUF model from {model_path} with {n_gpu_layers} GPU layers offloaded...")
            return Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
        else:
            print(f"Loading Hugging Face model/tokenizer from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )
            return {"model": model, "tokenizer": tokenizer}