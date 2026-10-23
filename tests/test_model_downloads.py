import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app import hub, hub_api


class _FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class ModelDownloadTests(unittest.TestCase):
    def test_cancel_terminates_blocking_download_process(self):
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            with (
                patch.object(hub, "MODELS_DIR", Path(directory)),
                patch.object(hub, "JOBS", {}),
                patch.object(hub, "CANCEL_EVENTS", {}),
                patch("app.hub.subprocess.Popen", return_value=process),
            ):
                result = hub.download("example/model")
                deadline = time.monotonic() + 5
                while hub.jobs()[0]["status"] not in ("downloading", "cancelling"):
                    if time.monotonic() >= deadline:
                        self.fail("Download worker did not start")
                    time.sleep(0.01)

                hub.cancel("example/model")
                while hub.jobs()[0]["status"] != "cancelled":
                    if time.monotonic() >= deadline:
                        self.fail("Download process did not stop")
                    time.sleep(0.01)

                self.assertTrue(result["ok"])
                self.assertTrue(process.terminated)
                self.assertEqual(hub.jobs()[0]["status"], "cancelled")

    def test_local_inventory_reports_actual_disk_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "example--model"
            repo.mkdir()
            (repo / "config.json").write_text("{}", encoding="utf-8")
            weight_file = repo / "model.safetensors"
            weight_file.write_bytes(b"actual model bytes")

            with patch.object(hub, "MODELS_DIR", Path(directory)):
                model = hub.local_models()[0]

            expected_size = sum(
                path.stat().st_size for path in repo.rglob("*") if path.is_file()
            )
            self.assertEqual(model["local_bytes"], expected_size)
            self.assertFalse(model["size_estimated"])

    def test_hub_used_storage_is_reported_as_exact_size(self):
        raw = {
            "id": "example/model",
            "usedStorage": 123456,
            "tags": [],
            "pipeline_tag": "text-generation",
        }
        with patch.object(hub_api, "_text", return_value='{"architectures": []}'):
            model = hub_api._one(raw)

        self.assertEqual(model["size_bytes"], 123456)
        self.assertFalse(model["size_estimated"])


if __name__ == "__main__":
    unittest.main()
