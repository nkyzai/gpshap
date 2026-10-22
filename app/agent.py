import ast
import json
import re
from typing import Iterator

from . import db, llm, rag, tools, vision
from .config import CONFIG

# FIXED: Properly closed list of strings, added Qwen tool call tags
STOP_STRINGS = [
    "Observation:",
    "observation:",
    "\n<|im_start|>user",
    "\nHuman:",
    "\n### Response",
    "\n[TOOL RESULT]",
    "</tool_call>",
    "<|end|>",
]

REFUSAL_PATTERNS = [
    r"i('|\s)a?m (sorry|afraid|unable)",
    r"as an ai",
    r"as an artificial intelligence",
    r"i (do not|don't|cannot|can't) have (direct )?(access|internet|real[- ]time)",
    r"i (do not|don't|cannot|can't) (browse|access) the (internet|web)",
    r"my knowledge (cutoff|is limited)",
    r"i have no (way|ability) to",
    r"as a language model",
]

WEB_INTENT = re.compile(
    r"\b(latest|recent|news|today|tonight|current(ly)?|now|live|real[- ]?time|price|pricing|"
    r"cost|weather|forecast|score|result(s)?|who won|release(d)? date|version|update|"
    r"search|google|look up|lookup|find online|online|web|internet|website|url|link|"
    r"stock|ticker|exchange rate|population|president|ceo of)\b", re.I)

URL_RE = re.compile(r"(https?://[^\s<>\"')]+)", re.I)

DOC_INTENT = re.compile(
    r"\b(document|doc|docs|pdf|file|files|attachment|attachments|uploaded|paper|report|"
    r"spreadsheet|contract|resume|slide|slides|transcript|this text|source material|"
    r"according to|based on the)\b", re.I)

IMAGE_INTENT = re.compile(
    r"\b(generate|create|make|draw|paint|render|imagine)\b.{0,40}\b(image|picture|photo|"
    r"illustration|artwork|logo|poster|wallpaper|diagram)\b", re.I)

IMAGE_QUESTION = re.compile(
    r"\b(image|picture|photo|screenshot|diagram|chart|graph|slide|this)\b.{0,40}"
    r"\b(show|shows|showing|says?|contain|contains|depict|describe|read|what)\b", re.I)

MATH_RE = re.compile(
    r"^\s*[-+*/().0-9\s]{3,}=?\s*$|"
    r"\b(calculate|compute|what is|whats|sum of|average of|how much is)\b.{0,60}[0-9]", re.I)

SAFE_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "int": int, "float": float,
}


def _get_llm_engine():
    """Returns the correct generation function based on active backend."""
    backend = CONFIG.get("active_backend", "transformers")
    if backend == "llama":
        from . import llama_loader
        return llama_loader.generate_stream
    else:
        from . import llm
        return llm.generate_stream


def is_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(re.search(p, low) for p in REFUSAL_PATTERNS)


def looks_like_tool_call(text: str) -> bool:
    head = (text or "").lstrip()[:24].lower()
    return (head.startswith("{") or head.startswith("<tool_call>") or
            head.startswith("action:") or head.startswith("action input") or
            head.startswith("tool:"))


def plan_tools(message: str, project_id: int, has_images: bool,
               has_documents: bool) -> list[dict]:
    if not CONFIG.get("agent_router", True):
        return []

    plans: list[dict] = []
    seen = set()

    def add(name: str, arguments: dict):
        key = (name, json.dumps(arguments, sort_keys=True))
        if key not in seen:
            seen.add(key)
            plans.append({"tool": name, "arguments": arguments})

    urls = URL_RE.findall(message)
    if urls:
        add("fetch_url", {"url": urls[0]})
    elif IMAGE_INTENT.search(message):
        pass  # handled by the caller as an /imagine command
    elif WEB_INTENT.search(message):
        query = re.sub(r"[?！!。.]+$", "", message).strip()
        query = re.sub(
            r"^(hey|hi|hello|please|can you|could you|would you)\b[\s,]*", "", query, flags=re.I)
        add("web_search", {"query": query[:180]})

    if has_images and IMAGE_QUESTION.search(message) and not IMAGE_INTENT.search(message):
        add("describe_image", {"question": message.strip()[:400]})

    if has_documents and DOC_INTENT.search(message):
        add("search_project_documents",
            {"project_id": project_id, "query": message.strip()[:300]})

    if MATH_RE.search(message) and len(message) < 220:
        expr = extract_math(message)
        if expr:
            add("calculator", {"expression": expr})

    return plans[:3]


def extract_math(message: str) -> str:
    match = re.search(r"([-+*/().0-9 ]{3,}={0,1})", message)
    if not match:
        return ""
    candidate = match.group(1).strip().rstrip("=")
    candidate = re.sub(r"[^0-9+\-*/(). ]", "", candidate).strip()
    if len(candidate) < 3 or not re.search(r"[0-9]", candidate):
        return ""
    if not re.search(r"[-+*/()]", candidate):
        return ""
    return candidate


def safe_calculate(expression: str):
    tree = ast.parse(expression, mode="eval")

    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Constant, ast.Expression,
                             ast.Load, ast.Call, ast.Name)):
            continue
        raise ValueError("unsupported expression")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") not in SAFE_FUNCTIONS:
            raise ValueError("unsupported function")
        if isinstance(node, ast.Name) and node.id not in SAFE_FUNCTIONS:
            raise ValueError("unsupported name")

    return eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, dict(SAFE_FUNCTIONS))


TOOL_DOC = """
You are Gapshap, an autonomous local agent on the user's Windows PC.

TOOL ACCESS (you are NOT limited to training data):
- web_search { "query": string }        -> live internet results
- fetch_url  { "url": string }          -> full text of a page
- search_project_documents { "project_id": %d, "query": string } -> user's uploaded files
- describe_image { "question": string } -> analysis of the attached image(s)
- generate_image { "prompt": string }   -> creates a local image
- list_projects {}                      -> the user's projects
- calculator { "expression": string }   -> exact arithmetic

HARD RULES:
1. You DO have live internet access through web_search and fetch_url.
2. NEVER say you cannot access the internet, real-time data, or current information.
3. If information is missing, outdated, or user-specific: call a tool. Do not guess, do not refuse.
4. Tool results are provided to you as [TOOL RESULT] blocks. Treat them as ground truth and cite them.
5. To call a tool yourself, reply with JSON ONLY, no prose, no markdown:
   {"tool": "web_search", "arguments": {"query": "..."}}
6. If you already have enough context, answer directly and completely.
""".strip()


def build_system_prompt(project: dict, project_id: int, agent_mode: bool,
                        has_documents: bool, has_images: bool) -> str:
    base = (project.get("system_prompt") or "").strip()
    parts = []

    if base:
        parts.append(base)

    if agent_mode:
        parts.append(TOOL_DOC % project_id)

    facts = [f"Today's date context matters: prefer tools over memory for anything time-sensitive."]
    if has_documents:
        facts.append("This project has uploaded documents indexed for retrieval - "
                     "use search_project_documents when the user references them.")
    if has_images:
        facts.append("This project has image attachments - use describe_image for visual questions.")

    parts.append("CONTEXT:\n- " + "\n- ".join(facts))
    return "\n\n".join(parts)


NUDGE = (
    "STOP. You claimed you cannot access live information, which is false - you have web_search "
    "and fetch_url. Re-answer now: if [TOOL RESULT] data is present, use it and cite it; "
    "otherwise reply with ONLY the JSON tool call you need. Never repeat the refusal."
)


def parse_tool_call(text: str) -> dict | None:
    if not text:
        return None

    # Check for Qwen-style <tool_call>
    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S | re.I)
    if tc_match:
        try:
            obj = json.loads(tc_match.group(1))
            name = obj.get("name") or obj.get("tool")
            args = obj.get("arguments") or obj.get("args") or {}
            if name:
                return {"tool": str(name), "arguments": args, "format": "json"}
        except Exception:
            pass

    # ReAct style (Qwen/Llama native tool templates)
    m = re.search(r"Action\s*:\s*([A-Za-z_][\w]*)\s*\n?\s*Action Input\s*:\s*(\{.*?\}|\[.*?\]|\"[^\"]*\")",
                  text, re.S)
    if m:
        name = m.group(1).strip()
        raw = m.group(2).strip()
        try:
            args = json.loads(raw)
        except Exception:
            try:
                args = ast.literal_eval(raw)
            except Exception:
                args = {"query": raw.strip('"')}
        if not isinstance(args, dict):
            args = {"value": args}
        return {"tool": name, "arguments": args, "format": "react"}

    candidates = []
    for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        candidates.append(block)
    candidates.append(text)
    candidates.extend(m.group(0) for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.S))

    for candidate in candidates:
        obj = None
        try:
            obj = json.loads(candidate)
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
            except Exception:
                obj = None

        if not isinstance(obj, dict):
            continue

        name = obj.get("tool") or obj.get("name") or obj.get("function") or \
            (obj.get("tool_call") or {}).get("name")
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or \
            obj.get("input") or {}

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"query": args}
        if not isinstance(args, dict):
            args = {"value": args}

        if name:
            return {"tool": str(name), "arguments": args, "format": "json"}

    return None


def summarize_tool_output(payload, limit: int = 2600) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > limit:
        text = text[:limit] + " …[truncated]"
    return text


def run(project: dict, project_id: int, user_message: str, history: list[dict],
        user: dict, use_rag: bool, agent_mode: bool,
        image_paths: list[str] | None = None) -> Iterator[dict]:
    """Yields event dicts: status|tool|sources|delta|images|done|error"""
    try:
        files = db.list_files(project_id)
        has_documents = any(f["kind"] != "image" and (f["text_extracted"] or "").strip()
                            for f in files)
        has_images = bool(image_paths) or any(f["kind"] == "image" for f in files)

        # ---- slash command: /imagine ------------------------------------
        if user_message.lower().startswith(("/imagine", "/image", "/draw")):
            prompt = re.sub(r"^/(imagine|image|draw)\s*", "", user_message, flags=re.I).strip()
            if not prompt:
                yield {"event": "error", "data": {"message": "Usage: /imagine <prompt>"}}
                return

            yield {"event": "status", "data": {"stage": "image-generation", "prompt": prompt}}
            from . import imagegen
            job = imagegen.Job(prompt)
            imagegen.JOBS[job.id] = job
            yield {"event": "job", "data": {"job_id": job.id}}

            try:
                result = imagegen.generate(prompt, job=job)
            except Exception as exc:
                yield {"event": "error",
                       "data": {"message": f"Image generation failed: {exc}"}}
                return

            db.add_generated_image(user["id"], project_id, prompt, "", result["filename"],
                                   result["width"], result["height"], result["steps"],
                                   result["seed"], result["model"])
            yield {"event": "images", "data": {"images": [result]}}

            answer = f"Generated an image for: {prompt}"
            db.add_message(project_id, "assistant", answer, meta={"images": [result]})
            yield {"event": "done", "data": {"answer": answer, "images": [result],
                                             "sources": [], "tool_calls": []}}
            return

        # ---- 1. deterministic router (pre-LLM tool execution) -----------
        router_results = []
        if agent_mode:
            plans = plan_tools(user_message, project_id, has_images, has_documents)
            for plan in plans:
                yield {"event": "tool", "data": {**plan, "state": "start", "source": "router"}}
                try:
                    if plan["tool"] == "describe_image":
                        payload = _describe_images(image_paths or
                                                   [f["stored_path"] for f in files
                                                    if f["kind"] == "image"][:2],
                                                   plan["arguments"].get("question", user_message))
                    elif plan["tool"] == "calculator":
                        payload = {"expression": plan["arguments"]["expression"],
                                   "result": safe_calculate(plan["arguments"]["expression"])}
                    else:
                        payload = tools.execute_tool(
                            plan["tool"], plan["arguments"], user=user, project_id=project_id)
                except Exception as exc:
                    payload = {"error": str(exc)}

                router_results.append({"tool": plan["tool"],
                                       "arguments": plan["arguments"], "result": payload})
                yield {"event": "tool", "data": {**plan, "state": "done", "source": "router",
                                                 "preview": summarize_tool_output(payload, 400)}}

        # ---- 2. RAG -------------------------------------------------------
        sources = []
        if use_rag:
            try:
                hits = rag.search_project(project_id, user_message,
                                          int(CONFIG.get("top_k_rag", 4)))
            except Exception:
                hits = []

            for hit in hits:
                sources.append({"text": hit.get("text", "")[:700],
                                "score": hit.get("score"),
                                "metadata": hit.get("metadata", {})})

        # ---- 3. assemble messages ----------------------------------------
        system_prompt = build_system_prompt(project, project_id, agent_mode,
                                            has_documents, has_images)
        messages = [{"role": "system", "content": system_prompt}]

        context_parts = []
        if sources:
            yield {"event": "sources", "data": {"sources": sources}}
            context_parts.append("RETRIEVED PROJECT DOCUMENTS:\n" +
                                 "\n---\n".join(s["text"] for s in sources))

        for item in router_results:
            context_parts.append(
                f"[TOOL RESULT] {item['tool']} {json.dumps(item['arguments'], ensure_ascii=False)}\n"
                f"{summarize_tool_output(item['result'], int(CONFIG.get('context_budget', 6000)))}"
            )

        if image_paths and not any(i["tool"] == "describe_image" for i in router_results):
            desc = _describe_images(image_paths[:2], user_message)
            if desc.get("text"):
                context_parts.append(f"[TOOL RESULT] describe_image\n{desc['text'][:3000]}")

        if context_parts:
            messages.append({"role": "system",
                             "content": "\n\n".join(context_parts)[:16000]})

        messages.extend(history[-int(CONFIG.get("history_messages", 12)):])

        # ---- 4. generate (with ReAct tool loop) --------------------------
        gen_func = _get_llm_engine()
        tool_specs = tools.MCP_TOOLS if agent_mode else None
        rounds = int(CONFIG.get("max_tool_rounds", 3))
        extra_calls = []
        images_out = []
        answer = ""

        for round_index in range(rounds + 1):
            collected = []
            held = ""
            decided = False
            suppress = False

            kwargs = {"stop": STOP_STRINGS}
            # Only pass 'tools' if using transformers backend (GGUF loader doesn't support it yet)
            if CONFIG.get("active_backend", "transformers") == "transformers":
                kwargs["tools"] = tool_specs

            for piece in gen_func(messages, **kwargs):
                collected.append(piece)

                if not decided:
                    held += piece
                    if len(held.strip()) >= 16:
                        suppress = looks_like_tool_call(held)
                        decided = True
                        if suppress:
                            yield {"event": "status",
                                   "data": {"stage": "tool-call-detected"}}
                        else:
                            yield {"event": "delta", "data": {"text": held}}
                            held = ""
                    continue

                if not suppress:
                    yield {"event": "delta", "data": {"text": piece}}

            raw = "".join(collected)
            answer = raw.strip() or "[no response]"

            if suppress:
                call = parse_tool_call(answer)
            else:
                call = None

            # refusal guard: force a corrective retry, once
            if is_refusal(answer) and agent_mode and round_index < rounds:
                yield {"event": "status", "data": {"stage": "refusal-detected-retrying"}}
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user", "content": NUDGE})
                if not suppress:
                    yield {"event": "reset", "data": {"reason": "refusal"}}
                continue

            if not call:
                break

            if round_index >= rounds:
                break

            extra_calls.append(call)
            yield {"event": "tool", "data": {**call, "state": "start", "source": "llm"}}

            try:
                if call["tool"] == "describe_image":
                    payload = _describe_images(image_paths or [],
                                               call["arguments"].get("question", user_message))
                elif call["tool"] == "generate_image":
                    from . import imagegen
                    prompt = call["arguments"].get("prompt") or user_message
                    job = imagegen.Job(prompt)
                    imagegen.JOBS[job.id] = job
                    yield {"event": "job", "data": {"job_id": job.id}}
                    payload = imagegen.generate(prompt, job=job)
                    db.add_generated_image(user["id"], project_id, prompt, "",
                                           payload["filename"], payload["width"],
                                           payload["height"], payload["steps"],
                                           payload["seed"], payload["model"])
                    images_out.append(payload)
                elif call["tool"] == "calculator":
                    payload = {"expression": call["arguments"].get("expression", ""),
                               "result": safe_calculate(call["arguments"].get("expression", "0"))}
                else:
                    payload = tools.execute_tool(call["tool"], call["arguments"],
                                                 user=user, project_id=project_id)
            except Exception as exc:
                payload = {"error": f"{type(exc).__name__}: {exc}"}

            yield {"event": "tool", "data": {**call, "state": "done", "source": "llm",
                                             "preview": summarize_tool_output(payload, 400)}}

            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user",
                             "content": f"[TOOL RESULT] {call['tool']}\n"
                                        f"{summarize_tool_output(payload)}\n\n"
                                        f"Now write the final answer for the user. "
                                        f"Do not call another tool unless essential."})

        # ---- 5. persist ---------------------------------------------------
        meta = {"sources": sources, "tool_calls": extra_calls,
                "router_tools": [r["tool"] for r in router_results]}
        if images_out:
            meta["images"] = images_out
        if image_paths:
            meta["attachments"] = [str(p) for p in image_paths]

        db.add_message(project_id, "assistant", answer, meta=meta)

        if images_out:
            yield {"event": "images", "data": {"images": images_out}}

        yield {"event": "done", "data": {"answer": answer, "sources": sources,
                                         "tool_calls": extra_calls,
                                         "router_tools": meta["router_tools"],
                                         "images": images_out}}

    except Exception as exc:
        import traceback
        traceback.print_exc()
        yield {"event": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}}


def _describe_images(paths: list[str], question: str) -> dict:
    paths = [p for p in (paths or []) if p]
    if not paths:
        return {"error": "no image attached"}

    if question and len(question.strip()) > 3:
        try:
            res = vision.ask(paths, question)
            if res.get("answer"):
                return {"text": res["answer"], "engine": res.get("engine"),
                        "mode": res.get("mode")}
        except Exception as exc:
            return {"error": str(exc)}

    blobs = []
    for p in paths[:2]:
        res = vision.analyze(p)
        if res.get("text"):
            blobs.append(res["text"])
    return {"text": "\n".join(blobs), "mode": "caption"}