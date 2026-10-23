import json
import re
import threading
import time
from typing import Iterator

from .config import CONFIG
from . import db, rag, tools, vision

# Global cancellation flags for projects
CANCELLED_PROJECTS: dict[int, bool] = {}

STOP_STRINGS = ["</s>", "<|im_end|>", "<|endoftext|>"]

NUDGE = (
    "The previous response was incomplete or refused. "
    "Please provide a direct, helpful answer using the available tools if necessary."
)

def _get_llm_engine():
    """Return the unified generation interface for the configured backend."""
    from . import llm
    return llm.generate_stream

def plan_tools(user_message: str, project_id: int, has_images: bool, has_documents: bool) -> list[dict]:
    """Deterministic router for common intents."""
    plans = []
    lower = user_message.lower()
    
    if has_images and any(k in lower for k in ["describe", "what is", "read", "text", "see"]):
        plans.append({"tool": "describe_image", "arguments": {"question": user_message}})
    
    if any(k in lower for k in ["calculate", "compute", "math", "sum"]):
        # Very basic regex for math
        match = re.search(r'([\d\s\+\-\*\/\(\)\.]+)', user_message)
        if match:
            expr = match.group(1)
            plans.append({"tool": "calculator", "arguments": {"expression": expr}})
            
    return plans

def summarize_tool_output(payload: dict, max_len: int = 1000) -> str:
    if isinstance(payload, dict):
        if "error" in payload:
            return f"Error: {payload['error']}"
        return json.dumps(payload, ensure_ascii=False)[:max_len]
    return str(payload)[:max_len]

def safe_calculate(expr: str) -> float | str:
    try:
        # Only allow safe chars
        if re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expr):
            return eval(expr)
        return "Invalid expression"
    except Exception as e:
        return str(e)

def parse_tool_call(text: str) -> dict | None:
    try:
        # Look for JSON block
        match = re.search(r'\{[^{}]+\}', text)
        if match:
            obj = json.loads(match.group(0))
            if "tool" in obj and "arguments" in obj:
                return obj
    except Exception:
        pass
    return None

def is_refusal(text: str) -> bool:
    lower = text.lower()
    refusals = ["i cannot", "i can't", "i am not able", "as an ai", "i don't have access"]
    return any(r in lower for r in refusals)

def _describe_images(paths: list[str], question: str) -> dict:
    if not paths:
        return {"error": "No images provided"}
    try:
        return vision.ask(paths, question)
    except Exception as e:
        return {"error": str(e)}

def run(project: dict, project_id: int, user_message: str, history: list[dict],
        user: dict, use_rag: bool, agent_mode: bool,
        image_paths: list[str] | None = None) -> Iterator[dict]:
    """Yields event dicts: status|tool|sources|delta|done|error"""
    try:
        files = db.list_files(project_id)
        has_documents = any(f["kind"] != "image" and (f["text_extracted"] or "").strip()
                            for f in files)
        has_images = bool(image_paths) or any(f["kind"] == "image" for f in files)

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

        # ---- 2. build context -------------------------------------------
        messages = []
        sys_prompt = project.get("system_prompt", "")
        if agent_mode:
            sys_prompt += "\n\n" + TOOL_DOC % project_id
            
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
            
        # Add history
        for h in history:
            messages.append(h)
            
        # Add RAG context if enabled
        sources = []
        if use_rag and has_documents:
            docs = rag.search(project_id, user_message, top_k=int(CONFIG.get("top_k_rag", 4)))
            if docs:
                context = "\n\n".join([d["text"] for d in docs])
                messages.append({"role": "system", "content": f"Context:\n{context}"})
                sources = docs

        if sources:
            yield {"event": "sources", "data": sources}

        # Append current user message
        messages.append({"role": "user", "content": user_message})

        # ---- 4. generate (with ReAct tool loop) --------------------------
        gen_func = _get_llm_engine()
        tool_specs = tools.MCP_TOOLS if agent_mode else None
        rounds = int(CONFIG.get("max_tool_rounds", 3))
        extra_calls = []
        images_out = []
        answer = ""

        for round_index in range(rounds + 1):
            # Check for cancellation
            if CANCELLED_PROJECTS.get(project_id):
                yield {"event": "done", "data": {"answer": answer + "\n\n[Stopped]", "sources": sources, "tool_calls": extra_calls, "images": images_out}}
                return

            collected = []
            held = ""
            decided = False
            suppress = False

            kwargs = {"stop": STOP_STRINGS}
            # Only pass 'tools' if using transformers backend (GGUF loader doesn't support it yet)
            if CONFIG.get("active_backend", "transformers") == "transformers":
                kwargs["tools"] = tool_specs

            for piece in gen_func(messages, **kwargs):
                # Check for cancellation inside the stream
                if CANCELLED_PROJECTS.get(project_id):
                    yield {"event": "delta", "data": {"text": piece}}
                    yield {"event": "done", "data": {"answer": "".join(collected) + piece + "\n\n[Stopped]", "sources": sources, "tool_calls": extra_calls, "images": images_out}}
                    return
                    
                collected.append(piece)
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

        # Save final answer
        if answer:
            db.add_message(project_id, "assistant", answer, meta={
                "sources": sources,
                "tool_calls": extra_calls,
                "images": images_out,
                "router_tools": [p["tool"] for p in router_results]
            })
            
        yield {"event": "done", "data": {"answer": answer, "sources": sources, "tool_calls": extra_calls, "images": images_out}}

    except Exception as exc:
        yield {"event": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}}

TOOL_DOC = """
You are Gapshap, an autonomous local agent on the user's Windows PC.

LOCAL TOOL ACCESS:
- search_project_documents { "project_id": %d, "query": string } -> user's uploaded files
- describe_image { "question": string } -> analysis of the attached image(s)
- generate_image { "prompt": string }   -> creates a local image
- list_projects {}                      -> the user's projects
- calculator { "expression": string }   -> exact arithmetic

All tools run locally. Do not claim live internet access or invent current facts.
If information is missing or uncertain, say so. If a local tool provides results,
use those results accurately. Answer directly when you have enough context.
""".strip()