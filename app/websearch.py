from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_RESPONSE_BYTES = 1_000_000


class _DuckDuckGoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urlparse(new_url).hostname not in {
            "html.duckduckgo.com",
            "duckduckgo.com",
            "www.duckduckgo.com",
        }:
            raise HTTPError(request.full_url, code, "Unexpected search redirect", headers, response)
        return super().redirect_request(request, response, code, message, headers, new_url)


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._capture: dict[str, str | int] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if self._capture is None:
            if tag == "a" and "result__a" in classes:
                self._capture = {
                    "kind": "title",
                    "tag": tag,
                    "depth": 1,
                    "href": attributes.get("href") or "",
                    "text": "",
                }
            elif classes and "result__snippet" in classes and self.results:
                self._capture = {"kind": "snippet", "tag": tag, "depth": 1, "text": ""}
        else:
            self._capture["depth"] = int(self._capture["depth"]) + 1

    def handle_endtag(self, tag: str) -> None:
        if self._capture is None:
            return
        self._capture["depth"] = int(self._capture["depth"]) - 1
        if self._capture["depth"] != 0 or tag != self._capture["tag"]:
            return
        kind = str(self._capture["kind"])
        text = str(self._capture["text"]).strip()
        if kind == "title" and text:
            self.results.append({
                "title": text[:240],
                "url": str(self._capture["href"])[:2048],
                "snippet": "",
            })
        elif kind == "snippet" and text:
            self.results[-1]["snippet"] = text[:600]
        self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._capture["text"] = str(self._capture["text"]) + data


def _destination_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in {"duckduckgo.com", "www.duckduckgo.com"} and parsed.path == "/l/":
        destination = parse_qs(parsed.query).get("uddg", [""])[0]
        if destination:
            url = unquote(destination)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def search_duckduckgo(query: str, limit: int = 5) -> list[dict[str, str]]:
    query = " ".join((query or "").split())
    if not query:
        raise ValueError("A search query is required")
    if len(query) > 400:
        query = query[:400]

    request = Request(
        f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
        headers={"User-Agent": "Gapshap/1.0 (local opt-in web search)"},
    )
    opener = build_opener(_DuckDuckGoRedirectHandler())
    with opener.open(request, timeout=8) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("DuckDuckGo search response exceeded the size limit")

    parser = _DuckDuckGoParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    results = []
    for result in parser.results:
        url = _destination_url(result["url"])
        if url:
            results.append({**result, "url": url})
        if len(results) >= max(1, min(limit, 10)):
            break
    return results
