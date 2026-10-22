import json
import re
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

from . import db, rag


HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
}


MCP_TOOLS = [
    {
        "name": "web_search",
        "description": "Search the live web (DuckDuckGo). Use for anything current, "
                       "factual, price/news/version related.",
        "inputSchema": {"type": "object",
                        "properties": {"query": {"type": "string",
                                                 "description": "concise search query"}},
                        "required": ["query"]},
    },
    {
        "name": "fetch_url",
        "description": "Download a URL and extract its readable text.",
        "inputSchema": {"type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"]},
    },
    {
        "name": "search_project_documents",
        "description": "Semantic search over the documents uploaded to this project.",
        "inputSchema": {"type": "object",
                        "properties": {"project_id": {"type": "integer"},
                                       "query": {"type": "string"}},
                        "required": ["query"]},
    },
    {
        "name": "describe_image",
        "description": "Analyse the attached image(s): captions, OCR and answers to "
                       "visual questions.",
        "inputSchema": {"type": "object",
                        "properties": {"file_id": {"type": "integer"},
                                       "question": {"type": "string"}}},
    },
    {
        "name": "generate_image",
        "description": "Create an image locally with the diffusion model.",
        "inputSchema": {"type": "object",
                        "properties": {"prompt": {"type": "string"},
                                       "steps": {"type": "integer"},
                                       "width": {"type": "integer"},
                                       "height": {"type": "integer"}},
                        "required": ["prompt"]},
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly.",
        "inputSchema": {"type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"]},
    },
    {
        "name": "list_projects",
        "description": "List the user's gapshap projects.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def web_search(query: str) -> dict:
    query = (query or "").strip()
    if not query:
        return {"error": "query is required"}

    response = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                             headers=HEADERS, timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_nodes = soup.select('a[data-testid="result-title-a"], a.result__a')
    snippet_nodes = soup.select(".result__snippet")

    results = []
    for i, node in enumerate(title_nodes[:6]):
        href = node.get("href", "")
        if href.startswith("//"):
            href = "https:" + href
        try:
            qs = parse_qs(urlparse(href).query)
            if "uddg" in qs:
                href = qs["uddg"][0]
        except Exception:
            pass

        results.append({
            "title": node.get_text(" ", strip=True),
            "url": href,
            "snippet": snippet_nodes[i].get_text(" ", strip=True) if i < len(snippet_nodes) else "",
        })

    return {"query": query, "count": len(results), "results": results}


def fetch_url(url: str) -> dict:
    url = (url or "").strip()
    if not url:
        return {"error": "url is required"}
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return {"url": response.url, "status_code": response.status_code, "text": text[:9000]}


def search_project_documents(project_id, query: str) -> dict:
    if project_id is None:
        return {"error": "project_id is required"}
    query = (query or "").strip()
    if not query:
        return {"error": "query is required"}
    return {"project_id": int(project_id), "query": query,
            "results": rag.search_project(int(project_id), query, top_k=5)}


def describe_image(file_id=None, question: str = "", project_id=None) -> dict:
    from . import vision

    paths = []
    if file_id:
        row = db.get_file(int(file_id))
        if row:
            paths.append(row["stored_path"])
    elif project_id:
        paths = [f["stored_path"] for f in db.project_image_files(int(project_id), 2)]

    if not paths:
        return {"error": "no image found for this project"}

    if not vision.is_ready():
        return {"error": "vision model not downloaded - open the Models tab and install it",
                "paths": paths}

    return vision.ask(paths, question or "Describe this image in detail, including any text.")


def generate_image(prompt: str, steps: int | None = None, width: int | None = None,
                   height: int | None = None) -> dict:
    from . import hub, imagegen

    if not hub.is_cached(imagegen_model()):
        return {"error": "image model not downloaded - open the Models tab and install it"}

    job = imagegen.Job(prompt)
    imagegen.JOBS[job.id] = job
    result = imagegen.generate(prompt, steps=steps, width=width, height=height, job=job)
    return {"images": [result], "job_id": job.id}


def imagegen_model() -> str:
    from .config import CONFIG
    return CONFIG.get("image_model") or ""


def list_projects(user: dict | None) -> dict:
    if not user:
        return {"error": "authentication required"}
    return {"projects": [{"id": r["id"], "name": r["name"]} for r in db.list_projects(user["id"])]}


def execute_tool(name: str, arguments: dict | None = None, user: dict | None = None,
                 project_id: int | None = None) -> dict:
    arguments = arguments or {}
    try:
        if name == "web_search":
            return web_search(arguments.get("query", ""))
        if name == "fetch_url":
            return fetch_url(arguments.get("url", ""))
        if name == "search_project_documents":
            pid = arguments.get("project_id", project_id)
            if pid is None:
                return {"error": "project_id is required"}
            return search_project_documents(pid, arguments.get("query", ""))
        if name == "describe_image":
            return describe_image(arguments.get("file_id"), arguments.get("question", ""),
                                  arguments.get("project_id", project_id))
        if name == "generate_image":
            return generate_image(arguments.get("prompt", ""), arguments.get("steps"),
                                  arguments.get("width"), arguments.get("height"))
        if name == "calculator":
            from .agent import safe_calculate
            expr = arguments.get("expression", "")
            return {"expression": expr, "result": safe_calculate(expr)}
        if name == "list_projects":
            return list_projects(user)
        return {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def parse_tool_request(text: str) -> dict | None:
    from .agent import parse_tool_call
    return parse_tool_call(text)