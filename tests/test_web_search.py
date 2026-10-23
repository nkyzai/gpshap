import io
import unittest
from unittest.mock import patch

from app.websearch import search_duckduckgo


class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class DuckDuckGoSearchTests(unittest.TestCase):
    @patch("app.websearch.build_opener")
    def test_search_parses_titles_snippets_and_only_public_urls(self, mocked_opener):
        html = b"""
            <div class="result">
              <a class="result__a" href="https://example.com/article">Example <b>title</b></a>
              <a class="result__snippet">Fresh information from the page.</a>
            </div>
            <div class="result">
              <a class="result__a" href="javascript:alert(1)">Unsafe result</a>
              <a class="result__snippet">This result must be excluded.</a>
            </div>
        """
        opener = mocked_opener.return_value
        opener.open.return_value = _FakeResponse(html)

        results = search_duckduckgo("  latest   information ")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Example title")
        self.assertEqual(results[0]["url"], "https://example.com/article")
        self.assertEqual(results[0]["snippet"], "Fresh information from the page.")
        request = opener.open.call_args.args[0]
        self.assertTrue(request.full_url.startswith("https://html.duckduckgo.com/html/?q="))
        self.assertEqual(request.get_header("User-agent"), "Gapshap/1.0 (local opt-in web search)")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 8)

    def test_empty_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "query is required"):
            search_duckduckgo("   ")


if __name__ == "__main__":
    unittest.main()
