import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import auth, db, main


class ImageGenerationRouteTests(unittest.TestCase):
    def setUp(self):
        main.app.dependency_overrides[auth.get_current_user] = lambda: {"id": 3}

    def tearDown(self):
        main.app.dependency_overrides.pop(auth.get_current_user, None)

    @patch("app.main.llm.unload")
    @patch("app.main.imagegen.unload")
    @patch("app.main.db.add_generated_image", return_value=8)
    @patch("app.main.imagegen.generate")
    @patch("app.main.hub.jobs", return_value=[
        {"repo_id": "segmind/tiny-sd", "status": "ready", "downloaded": 0, "total": 0}
    ])
    @patch("app.main.hub.download", return_value={"ok": True, "job_id": "image"})
    @patch("app.main.hub.is_cached", return_value=False)
    def test_first_studio_generation_downloads_then_generates(
        self, is_cached, start_download, download_jobs, generate, add_generated_image,
        unload_imagegen, unload_llm
    ):
        events = []
        unload_llm.side_effect = lambda: events.append("unload_llm")
        unload_imagegen.side_effect = lambda: events.append("unload_imagegen")
        start_download.side_effect = lambda *_: (
            events.append("download"),
            {"ok": True, "job_id": "image"},
        )[1]
        generate.side_effect = lambda *args, **kwargs: (
            events.append("generate"),
            {
                "filename": "generated.png",
                "url": "/generated/generated.png",
                "path": "generated.png",
                "prompt": "a quiet lake",
                "negative": "",
                "steps": 4,
                "width": 512,
                "height": 512,
                "seed": 10,
                "model": "segmind/tiny-sd",
            },
        )[1]

        response = TestClient(main.app).post(
            "/api/imagegen/stream",
            json={"prompt": "a quiet lake"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("First-time setup", response.text)
        self.assertIn('"stage": "download"', response.text)
        self.assertIn("event: image\n", response.text)
        is_cached.assert_called_once_with("segmind/tiny-sd")
        start_download.assert_called_once_with("segmind/tiny-sd", "image")
        generate.assert_called_once()
        add_generated_image.assert_called_once()
        self.assertLess(events.index("unload_llm"), events.index("download"))
        self.assertLess(events.index("unload_imagegen"), events.index("download"))
        self.assertLess(events.index("download"), events.index("generate"))


if __name__ == "__main__":
    unittest.main()
