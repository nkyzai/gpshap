import unittest
from unittest.mock import patch

from app import hub


class ModelFilterTests(unittest.TestCase):
    @patch("app.hub.jobs", return_value=[])
    @patch("app.hub.disk_free", return_value={"free_gb": 10, "models_gb": 2})
    @patch("app.hub_api.enrich_many")
    @patch("app.hub_api.search_models", return_value=[{"id": "Qwen/Qwen-GGUF"}])
    def test_gguf_filter_searches_and_returns_only_gguf_models(
        self, search_models, enrich_many, _disk_free, _jobs
    ):
        enrich_many.return_value = [
            {"id": "Qwen/Qwen-GGUF", "format": "gguf"},
            {"id": "Qwen/Qwen", "format": "transformers"},
        ]

        result = hub.browse(
            search="Qwen",
            task="text-generation",
            limit=20,
            model_format="gguf",
        )

        search_models.assert_called_once_with(
            "Qwen gguf", "text-generation", "downloads", 20
        )
        self.assertEqual([model["id"] for model in result["models"]], ["Qwen/Qwen-GGUF"])

    def test_unknown_format_filter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model format filter"):
            hub.browse(model_format="pickle")


if __name__ == "__main__":
    unittest.main()
