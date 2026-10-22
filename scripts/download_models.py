import os
from huggingface_hub import snapshot_download
targets = []
targets.append(os.environ["GAPSHAP_MODEL"])
targets.append(os.environ["GAPSHAP_EMBED_MODEL"])
if os.environ.get("GAPSHAP_DL_VISION") == "1":
    targets.append(os.environ["GAPSHAP_VISION_MODEL"])
if os.environ.get("GAPSHAP_DL_IMAGEGEN") == "1":
    targets.append(os.environ["GAPSHAP_IMAGE_MODEL"])
for repo in targets:
    print("Downloading:", repo, flush=True)
    try:
        snapshot_download(repo_id=repo, ignore_patterns=["*.msgpack","*.h5","*.ot","*.onnx","*.tflite","original/*"])
        print("  OK:", repo, flush=True)
    except Exception as exc:
        print("  FAILED:", repo, exc, flush=True)
print("Bootstrap complete.", flush=True)
