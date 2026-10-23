import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import agent, auth, db, hub, imagegen, llm, rag, tools, vision
from .config import CONFIG, GENERATED_DIR, INDEX_DIR, STATIC_DIR, UPLOAD_DIR, update_config
from .mcp import router as mcp_router
from .model_runtime import MODEL_SESSION_LOCK
from .schemas import (AnalyzeRequest, AuthRequest, ChatRequest, ImageGenRequest,
                      ModelDownloadRequest, ModelUseRequest, ProjectCreate,
                      SettingsUpdate)

# Import platform safely so a typo in platform.py doesn't kill the whole server
try:
    from .platform import router as platform_router
    PLATFORM_OK = True
    PLATFORM_ERR = ""
except Exception as e:
    import traceback
    traceback.print_exc()
    platform_router = None
    PLATFORM_OK = False
    PLATFORM_ERR = str(e)

app = FastAPI(title="Gpshap", version="2.0.0")


def _exclusive_model_events(event_factory, prepare):
    def stream():
        with MODEL_SESSION_LOCK:
            prepare()
            yield from event_factory()

    return stream()


def _exclusive_agent_events(*args, **kwargs):
    return _exclusive_model_events(
        lambda: agent.run(*args, **kwargs),
        _prepare_text_generation,
    )


def _prepare_text_generation():
    imagegen.unload()


def _prepare_image_generation():
    llm.unload()
    imagegen.unload()


def _warmup_text_model():
    with MODEL_SESSION_LOCK:
        _prepare_text_generation()
        llm.warmup()


# 1. Mount API routers FIRST (This fixes the 404/405 errors)
app.include_router(auth.router, prefix="/api")
app.include_router(mcp_router)

if PLATFORM_OK and platform_router is not None:
    app.include_router(platform_router, prefix="/api")
else:
    @app.get("/api/platform-error")
    def platform_error():
        return {"error": PLATFORM_ERR, "hint": "Check terminal for traceback in app/platform.py"}

# 2. Health check
@app.get("/api/health")
def health():
    return {"ok": True, "service": "gpshap", "platform_loaded": PLATFORM_OK}

# 3. Serve the frontend
@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")

# --------------------------------------------------------------------------
# status & settings
# --------------------------------------------------------------------------
@app.get("/api/status")
def status(user=Depends(auth.get_current_user)):
    llm_info = llm.info()

    return {
        "llm": llm_info,
        "vision": vision.info(),
        "imagegen": imagegen.info(),
        "embedding": {"model": CONFIG.get("embedding_model"),
                      "loaded": rag._embed_model is not None},
        "disk": hub.disk_free(),
        "config": {k: v for k, v in CONFIG.items() if k != "hf_token"},
        "hf_token_set": bool(CONFIG.get("hf_token")),
    }

@app.get("/api/settings")
def get_settings(user=Depends(auth.get_current_user)):
    return {"settings": {k: v for k, v in CONFIG.items() if k != "hf_token"},
            "hf_token_set": bool(CONFIG.get("hf_token"))}

@app.post("/api/settings")
def save_settings(body: SettingsUpdate, user=Depends(auth.get_current_user)):
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    prev = {k: CONFIG.get(k) for k in patch}
    update_config(patch)
    if patch.get("model") and patch["model"] != prev.get("model"):
        threading.Thread(target=_warmup_text_model, daemon=True).start()
    if patch.get("embedding_model") and patch["embedding_model"] != prev.get("embedding_model"):
        rag.unload_embedder()
    if patch.get("vision_model") and patch["vision_model"] != prev.get("vision_model"):
        vision.unload()
    if patch.get("image_model") and patch["image_model"] != prev.get("image_model"):
        imagegen.unload()
    return {"ok": True, "settings": {k: v for k, v in CONFIG.items() if k != "hf_token"}}


# --------------------------------------------------------------------------
# projects (V2 - includes metadata for UI features like pin/archive)
# --------------------------------------------------------------------------
@app.get("/api/projects/v2")
def get_projects_v2(user=Depends(auth.get_current_user)):
    rows = db.list_projects(user["id"])
    return {"projects": [dict(r) for r in rows]}

@app.post("/api/projects")
def create_project(body: ProjectCreate, user=Depends(auth.get_current_user)):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Project name is required")
    pid = db.create_project(user["id"], name, body.system_prompt or "")
    return {"id": pid, "name": name, "system_prompt": body.system_prompt or ""}

@app.patch("/api/projects/{project_id}")
def patch_project(project_id: int, body: dict, user=Depends(auth.get_current_user)):
    if not db.get_project(user["id"], project_id):
        raise HTTPException(404, "Project not found")
    allowed = {"name", "system_prompt", "pinned", "archived", "sort_order"}
    patch = {k: v for k, v in body.items() if k in allowed}
    if patch:
        db.update_project_fields(project_id, patch)
    return {"ok": True}

@app.delete("/api/projects/{project_id}")
def delete_project(project_id: int, user=Depends(auth.get_current_user)):
    if not db.get_project(user["id"], project_id):
        raise HTTPException(404, "Project not found")
    db.delete_project(user["id"], project_id)
    shutil.rmtree(UPLOAD_DIR / str(project_id), ignore_errors=True)
    for suffix in (".faiss", ".chunks.json"):
        try:
            (INDEX_DIR / f"{project_id}{suffix}").unlink(missing_ok=True)
        except Exception:
            pass
    return {"ok": True}

@app.post("/api/projects/reorder")
def reorder_projects(body: dict, user=Depends(auth.get_current_user)):
    ids = body.get("project_ids") or []
    if not isinstance(ids, list):
        raise HTTPException(400, "project_ids must be a list")
    db.reorder_projects(user["id"], [int(i) for i in ids])
    return {"ok": True}

@app.post("/api/projects/{project_id}/duplicate")
def duplicate_project(project_id: int, user=Depends(auth.get_current_user)):
    if not db.get_project(user["id"], project_id):
        raise HTTPException(404, "Project not found")
    new_id = db.duplicate_project(user["id"], project_id)
    return {"ok": True, "new_id": new_id}

@app.get("/api/projects/{project_id}/export")
def export_project(project_id: int, user=Depends(auth.get_current_user)):
    proj = db.get_project(user["id"], project_id)
    if not proj:
        raise HTTPException(404, "Project not found")
    
    messages = []
    for row in db.list_messages(project_id, 1000):
        try:
            meta = json.loads(row["meta"] or "{}")
        except Exception:
            meta = {}
        messages.append({
            "role": row["role"],
            "content": row["content"],
            "meta": meta,
            "created_at": row["created_at"],
        })
        
    files = []
    for row in db.list_files(project_id):
        files.append({
            "filename": row["filename"],
            "kind": row["kind"],
            "mime": row["mime"],
            "indexed_chars": len(row["text_extracted"] or ""),
        })

    return {
        "project": dict(proj),
        "messages": messages,
        "files": files,
        "exported_at": time.time(),
    }

# --------------------------------------------------------------------------
# messages (V2 - includes number/meta for UI rendering)
# --------------------------------------------------------------------------
@app.get("/api/projects/{project_id}/messages/v2")
def get_messages_v2(project_id: int, user=Depends(auth.get_current_user)):
    if not db.get_project(user["id"], project_id):
        raise HTTPException(404, "Project not found")
    out = []
    for row in db.list_messages(project_id, 500):
        try:
            meta = json.loads(row["meta"] or "{}")
        except Exception:
            meta = {}
        out.append({
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "meta": meta,
            "number": row["number"] or 0,
            "created_at": row["created_at"],
        })
    return {"messages": out}

@app.delete("/api/projects/{project_id}/messages")
def clear_messages(project_id: int, user=Depends(auth.get_current_user)):
    if not db.get_project(user["id"], project_id):
        raise HTTPException(404, "Project not found")
    db.delete_project_messages(project_id)
    return {"ok": True}

@app.patch("/api/messages/{message_id}")
def patch_message(message_id: int, body: dict, user=Depends(auth.get_current_user)):
    content = body.get("content")
    if not content:
        raise HTTPException(400, "Content is required")
    updated = db.update_message_content(message_id, user["id"], content)
    if not updated:
        raise HTTPException(404, "Message not found or not editable")
    return {"ok": True, "message": updated}

@app.delete("/api/messages/{message_id}")
def delete_message_route(message_id: int, user=Depends(auth.get_current_user)):
    ok = db.delete_message(message_id, user["id"])
    if not ok:
        raise HTTPException(404, "Message not found")
    return {"ok": True}

# --------------------------------------------------------------------------
# chat (blocking + SSE streaming + Regenerate)
# --------------------------------------------------------------------------
def _project_context(project_id: int, user: dict, body: ChatRequest):
    project = db.get_project(user["id"], project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "Message is required")
    db.add_message(project_id, "user", message)
    history = []
    for row in db.list_messages(project_id, 60):
        if row["role"] in ("user", "assistant"):
            history.append({"role": row["role"], "content": row["content"]})
    history = history[-int(CONFIG.get("history_messages", 12)):]
    image_paths = []
    if body.file_ids:
        for fid in body.file_ids:
            row = db.get_file(int(fid))
            if row and row["project_id"] == project_id and row["kind"] == "image":
                image_paths.append(row["stored_path"])
    if not image_paths:
        image_paths = [r["stored_path"] for r in db.project_image_files(project_id, 2)]
    return project, message, history, image_paths

@app.post("/api/projects/{project_id}/chat")
def chat(project_id: int, body: ChatRequest, user=Depends(auth.get_current_user)):
    project, message, history, image_paths = _project_context(project_id, user, body)
    events = list(_exclusive_agent_events(
        project, project_id, message, history, user,
        body.use_rag, body.agent_mode, image_paths, body.use_web,
    ))
    done = next((e["data"] for e in events if e["event"] == "done"), None)
    error = next((e["data"] for e in events if e["event"] == "error"), None)
    if error and not done:
        raise HTTPException(500, error.get("message", "agent error"))
    return done or {"answer": "", "sources": [], "tool_calls": []}

def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"}

@app.post("/api/projects/{project_id}/chat/stream")
def chat_stream(project_id: int, body: ChatRequest, user=Depends(auth.get_current_user)):
    project, message, history, image_paths = _project_context(project_id, user, body)
    
    # Initialize cancellation flag for this project
    agent.CANCELLED_PROJECTS[project_id] = False

    def event_source():
        yield sse("start", {"project_id": project_id, "ts": time.time()})
        try:
            for event in _exclusive_agent_events(
                project, project_id, message, history, user,
                body.use_rag, body.agent_mode, image_paths, body.use_web,
            ):
                yield sse(event["event"], event["data"])
                # Check if cancelled during yield
                if agent.CANCELLED_PROJECTS.get(project_id):
                    break
        except Exception as exc:
            yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            agent.CANCELLED_PROJECTS.pop(project_id, None)
            yield sse("end", {})
            
    return StreamingResponse(event_source(), media_type="text/event-stream", headers=SSE_HEADERS)

@app.post("/api/projects/{project_id}/chat/stop")
def stop_chat(project_id: int, user=Depends(auth.get_current_user)):
    """Stop an ongoing generation for a specific project."""
    if not db.get_project(user["id"], project_id):
        raise HTTPException(404, "Project not found")
    agent.CANCELLED_PROJECTS[project_id] = True
    return {"ok": True, "stopped": True}

@app.post("/api/projects/{project_id}/regenerate/stream")
def regenerate_stream(project_id: int, body: dict, user=Depends(auth.get_current_user)):
    project = db.get_project(user["id"], project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    
    last_user = db.get_last_user_message(project_id)
    if not last_user:
        raise HTTPException(400, "No user message to regenerate")
    
    # Delete trailing assistant messages
    db.delete_trailing_assistant_messages(project_id)
    
    history = []
    for row in db.list_messages(project_id, 60):
        if row["role"] in ("user", "assistant"):
            history.append({"role": row["role"], "content": row["content"]})
    history = history[-int(CONFIG.get("history_messages", 12)):]
    
    use_rag = bool(body.get("use_rag", True))
    agent_mode = bool(body.get("agent_mode", True))
    use_web = bool(body.get("use_web", False))
    
    image_paths = [r["stored_path"] for r in db.project_image_files(project_id, 2)]

    agent.CANCELLED_PROJECTS[project_id] = False

    def event_source():
        yield sse("start", {"project_id": project_id, "ts": time.time()})
        try:
            for event in _exclusive_agent_events(
                project, project_id, last_user["content"], history, user,
                use_rag, agent_mode, image_paths, use_web,
            ):
                yield sse(event["event"], event["data"])
                if agent.CANCELLED_PROJECTS.get(project_id):
                    break
        except Exception as exc:
            yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            agent.CANCELLED_PROJECTS.pop(project_id, None)
            yield sse("end", {})
            
    return StreamingResponse(event_source(), media_type="text/event-stream", headers=SSE_HEADERS)

# --------------------------------------------------------------------------
# conversations
# --------------------------------------------------------------------------
@app.get("/api/chats")
def get_chats(user=Depends(auth.get_current_user)):
    return {"chats": [dict(row) for row in db.list_chats(user["id"])]}


@app.post("/api/chats")
def create_chat(body: dict, user=Depends(auth.get_current_user)):
    project_id = body.get("project_id")
    if project_id is not None:
        try:
            project_id = int(project_id)
        except (TypeError, ValueError):
            raise HTTPException(400, "Invalid project_id")
    title = str(body.get("title") or "New chat").strip()
    if not title:
        raise HTTPException(400, "Chat title cannot be empty")
    try:
        chat_id = db.create_chat(user["id"], project_id, title)
    except ValueError:
        raise HTTPException(404, "Project not found")
    return {"id": chat_id, "project_id": project_id, "title": title}


@app.patch("/api/chats/{chat_id}")
def patch_chat(chat_id: int, body: dict, user=Depends(auth.get_current_user)):
    title = str(body.get("title") or "").strip()
    if not title:
        raise HTTPException(400, "Chat title cannot be empty")
    if not db.update_chat_title(user["id"], chat_id, title):
        raise HTTPException(404, "Chat not found")
    return {"ok": True}


@app.delete("/api/chats/{chat_id}")
def remove_chat(chat_id: int, user=Depends(auth.get_current_user)):
    if not db.delete_chat(user["id"], chat_id):
        raise HTTPException(404, "Chat not found")
    return {"ok": True}


@app.post("/api/chats/{chat_id}/duplicate")
def duplicate_chat(chat_id: int, user=Depends(auth.get_current_user)):
    try:
        new_id = db.duplicate_chat(user["id"], chat_id)
    except ValueError:
        raise HTTPException(404, "Chat not found")
    return {"id": new_id}


@app.get("/api/chats/{chat_id}/messages/v2")
def get_chat_messages(chat_id: int, user=Depends(auth.get_current_user)):
    if not db.get_chat(user["id"], chat_id):
        raise HTTPException(404, "Chat not found")
    out = []
    for row in db.list_chat_messages(chat_id, 1000):
        try:
            meta = json.loads(row["meta"] or "{}")
        except (TypeError, ValueError):
            meta = {}
        out.append({
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "number": row["number"] or 0,
            "meta": meta,
            "created_at": row["created_at"],
        })
    return {"messages": out}


@app.delete("/api/chats/{chat_id}/messages")
def clear_chat_messages(chat_id: int, user=Depends(auth.get_current_user)):
    if not db.get_chat(user["id"], chat_id):
        raise HTTPException(404, "Chat not found")
    db.clear_chat(chat_id)
    return {"ok": True}


@app.get("/api/chats/{chat_id}/export")
def export_chat(chat_id: int, user=Depends(auth.get_current_user)):
    chat = db.get_chat(user["id"], chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    messages = []
    for row in db.list_chat_messages(chat_id, 1000000):
        try:
            meta = json.loads(row["meta"] or "{}")
        except (TypeError, ValueError):
            meta = {}
        messages.append({
            "id": row["id"], "role": row["role"], "content": row["content"],
            "number": row["number"] or 0, "meta": meta, "created_at": row["created_at"],
        })
    return JSONResponse(
        content={"app": "gpshap", "version": 2, "exported_at": time.time(),
                 "chat": chat, "messages": messages},
        headers={"Content-Disposition": f'attachment; filename="gpshap_chat_{chat_id}.json"'},
    )


def _chat_context(chat_id: int, body: ChatRequest, user: dict):
    chat = db.get_chat(user["id"], chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "Message is required")
    project_id = chat["project_id"]
    db.add_chat_message(chat_id, project_id, "user", message)
    history = [
        {"role": row["role"], "content": row["content"]}
        for row in db.list_chat_messages(chat_id, 60)
        if row["role"] in ("user", "assistant")
    ][-int(CONFIG.get("history_messages", 12)):]
    image_paths = []
    if project_id is not None:
        if body.file_ids:
            for file_id in body.file_ids:
                file_row = db.get_file(int(file_id))
                if file_row and file_row["project_id"] == project_id and file_row["kind"] == "image":
                    image_paths.append(file_row["stored_path"])
        if not image_paths:
            image_paths = [row["stored_path"] for row in db.project_image_files(project_id, 2)]
    project = {"system_prompt": chat.get("project_system_prompt") or ""}
    return chat, project_id, project, message, history, image_paths


@app.post("/api/chats/{chat_id}/chat/stream")
def stream_chat(chat_id: int, body: ChatRequest, user=Depends(auth.get_current_user)):
    chat, project_id, project, message, history, image_paths = _chat_context(chat_id, body, user)
    agent.CANCELLED_PROJECTS[chat_id] = False

    def event_source():
        yield sse("start", {"chat_id": chat_id, "project_id": project_id, "ts": time.time()})
        try:
            for event in _exclusive_agent_events(
                project, project_id, message, history, user, body.use_rag,
                body.agent_mode, image_paths, body.use_web, chat_id=chat_id,
            ):
                yield sse(event["event"], event["data"])
                if agent.CANCELLED_PROJECTS.get(chat_id):
                    break
        except Exception as exc:
            yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            agent.CANCELLED_PROJECTS.pop(chat_id, None)
            yield sse("end", {})

    return StreamingResponse(event_source(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/chats/{chat_id}/chat/stop")
def stop_chat_by_id(chat_id: int, user=Depends(auth.get_current_user)):
    if not db.get_chat(user["id"], chat_id):
        raise HTTPException(404, "Chat not found")
    agent.CANCELLED_PROJECTS[chat_id] = True
    return {"ok": True, "stopped": True}


@app.post("/api/chats/{chat_id}/regenerate/stream")
def regenerate_chat(chat_id: int, body: dict, user=Depends(auth.get_current_user)):
    chat = db.get_chat(user["id"], chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    last_user = db.get_last_chat_user_message(chat_id)
    if not last_user:
        raise HTTPException(400, "No user message to regenerate")
    db.delete_trailing_chat_assistant_messages(chat_id)
    history = [
        {"role": row["role"], "content": row["content"]}
        for row in db.list_chat_messages(chat_id, 60)
        if row["role"] in ("user", "assistant")
    ][-int(CONFIG.get("history_messages", 12)):]
    project_id = chat["project_id"]
    image_paths = [row["stored_path"] for row in db.project_image_files(project_id, 2)] if project_id else []
    project = {"system_prompt": chat.get("project_system_prompt") or ""}
    agent.CANCELLED_PROJECTS[chat_id] = False

    def event_source():
        yield sse("start", {"chat_id": chat_id, "project_id": project_id, "ts": time.time()})
        try:
            for event in _exclusive_agent_events(
                project, project_id, last_user["content"], history, user,
                bool(body.get("use_rag", True)), bool(body.get("agent_mode", True)),
                image_paths, bool(body.get("use_web", False)), chat_id=chat_id,
            ):
                yield sse(event["event"], event["data"])
                if agent.CANCELLED_PROJECTS.get(chat_id):
                    break
        except Exception as exc:
            yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            agent.CANCELLED_PROJECTS.pop(chat_id, None)
            yield sse("end", {})

    return StreamingResponse(event_source(), media_type="text/event-stream", headers=SSE_HEADERS)

# --------------------------------------------------------------------------
# files + vision
# --------------------------------------------------------------------------
@app.post("/api/projects/{project_id}/upload")
async def upload_file(project_id: int, file: UploadFile = File(...),
                      analyze: bool = Query(True),
                      user=Depends(auth.get_current_user)):
    if not db.get_project(user["id"], project_id):
        raise HTTPException(404, "Project not found")
    if not file.filename:
        raise HTTPException(400, "Filename is required")
    original = Path(file.filename).name
    stored_name = f"{uuid.uuid4().hex}_{original}"
    upload_dir = UPLOAD_DIR / str(project_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / stored_name
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    content_type = file.content_type or "application/octet-stream"
    kind = "image" if content_type.startswith("image/") else "file"
    vision_text = ""
    vision_ready = vision.is_ready() and CONFIG.get("vision_auto_analyze", True)
    if kind == "image" and analyze and vision_ready:
        try:
            result = vision.analyze(destination)
            vision_text = result.get("text", "")
        except Exception as exc:
            vision_text = f"vision analysis failed: {exc}"
    extracted = rag.extract_text(destination, original, content_type, extra_text=vision_text)
    file_id = db.add_file(project_id, user["id"], original, str(destination),
                          content_type, kind, extracted[:200_000])
    chunks = rag.chunk_text(extracted)
    if chunks:
        rag.add_texts(project_id, chunks,
                      [{"file_id": file_id, "filename": original, "mime": content_type}
                       for _ in chunks])
    return {"ok": True, "file_id": file_id, "filename": original, "kind": kind,
            "chunks": len(chunks), "vision_analyzed": bool(vision_text),
            "vision_available": vision_ready,
            "vision_preview": vision_text[:600]}

@app.post("/api/projects/{project_id}/analyze")
def analyze_image(project_id: int, body: AnalyzeRequest,
                  user=Depends(auth.get_current_user)):
    if not db.get_project(user["id"], project_id):
        raise HTTPException(404, "Project not found")
    paths = []
    if body.file_id:
        row = db.get_file(body.file_id)
        if not row or row["project_id"] != project_id:
            raise HTTPException(404, "File not found")
        paths = [row["stored_path"]]
    else:
        paths = [r["stored_path"] for r in db.project_image_files(project_id, 2)]
    if not paths:
        raise HTTPException(400, "No image in this project")
    result = vision.ask(paths, body.question or "Describe this image in full detail, "
                                                "including all visible text.")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    if body.file_id:
        row = db.get_file(body.file_id)
        merged = (row["text_extracted"] or "") + "\n\n" + result.get("answer", "")
        db.update_file_text(body.file_id, merged)
        chunks = rag.chunk_text(result.get("answer", ""))
        if chunks:
            rag.add_texts(project_id, chunks,
                          [{"file_id": body.file_id, "filename": row["filename"],
                            "vision": True} for _ in chunks])
    return result

@app.get("/api/projects/{project_id}/files")
def get_files(project_id: int, user=Depends(auth.get_current_user)):
    if not db.get_project(user["id"], project_id):
        raise HTTPException(404, "Project not found")
    return {"files": [{"id": r["id"], "filename": r["filename"], "mime": r["mime"],
                       "kind": r["kind"], "created_at": r["created_at"],
                       "indexed_chars": len(r["text_extracted"] or "")}
                      for r in db.list_files(project_id)]}

@app.get("/api/files/{file_id}")
def get_file(file_id: int, user=Depends(auth.get_current_user)):
    row = db.get_file(file_id)
    if not row or not db.get_project(user["id"], row["project_id"]):
        raise HTTPException(404, "File not found")
    path = Path(row["stored_path"])
    if not path.exists():
        raise HTTPException(404, "Stored file missing")
    return FileResponse(path, filename=row["filename"],
                        media_type=row["mime"] or "application/octet-stream")

@app.delete("/api/files/{file_id}")
def delete_file_route(file_id: int, user=Depends(auth.get_current_user)):
    row = db.get_file(file_id)
    if not row or not db.get_project(user["id"], row["project_id"]):
        raise HTTPException(404, "File not found")
    
    # Remove from disk
    try:
        Path(row["stored_path"]).unlink(missing_ok=True)
    except Exception:
        pass
        
    # Remove from DB
    db.delete_file(file_id)
    
    # Note: RAG index cleanup would happen here if we had a direct handle to FAISS
    # For now, the stale vectors remain until re-indexed, which is acceptable for MVP
    
    return {"ok": True}

# --------------------------------------------------------------------------
# model hub
# --------------------------------------------------------------------------
@app.get("/api/models/browse")
def models_browse(search: str = "", q: str = "", task: str = "all",
                  sort: str = "downloads", limit: int = 40,
                  deep: bool = False, force: bool = False,
                  user=Depends(auth.get_current_user)):
    try:
        return hub.browse(
            (q or search or "").strip(), task, sort,
            max(1, min(int(limit), 100)), deep, force
        )
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}")

@app.get("/api/hub/diagnose")
def hub_diagnose(user=Depends(auth.get_current_user)):
    from . import hub_api
    try:
        return hub_api.diagnostics()
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")

@app.get("/api/models/local")
def models_local(user=Depends(auth.get_current_user)):
    active_llm = (
        CONFIG.get("active_gguf_model")
        if CONFIG.get("active_backend") == "llama"
        else CONFIG.get("model")
    )
    return {"models": hub.local_models(), "disk": hub.disk_free(),
            "active": {"llm": active_llm, "vision": CONFIG.get("vision_model"),
                       "embedding": CONFIG.get("embedding_model"),
                       "image": CONFIG.get("image_model"),
                       "backend": CONFIG.get("active_backend", "transformers")},
            "gguf_config": {
                "gguf_n_ctx": CONFIG.get("gguf_n_ctx", 4096),
                "gguf_n_gpu_layers": CONFIG.get("gguf_n_gpu_layers", -1),
            }}

@app.get("/api/models/details/{repo_id:path}")
def models_details(repo_id: str, user=Depends(auth.get_current_user)):
    try:
        return hub.details(repo_id)
    except Exception as exc:
        raise HTTPException(502, str(exc))

@app.post("/api/models/download")
def models_download(body: ModelDownloadRequest, user=Depends(auth.get_current_user)):
    return hub.download(body.model_id.strip(), body.kind)

@app.post("/api/models/cancel")
def models_cancel(body: ModelDownloadRequest, user=Depends(auth.get_current_user)):
    result = hub.cancel(body.model_id.strip())
    if not result["ok"]:
        raise HTTPException(409, result["error"])
    return result

@app.post("/api/models/jobs/delete")
def models_delete_job(body: ModelDownloadRequest, user=Depends(auth.get_current_user)):
    try:
        return hub.delete_job(body.model_id.strip())
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))

@app.post("/api/models/use")
def models_use(body: ModelUseRequest, user=Depends(auth.get_current_user)):
    try:
        return hub.use_model(
            body.model_id.strip(),
            body.kind,
            gguf_n_ctx=body.gguf_n_ctx,
            gguf_n_gpu_layers=body.gguf_n_gpu_layers,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc))

@app.post("/api/models/delete")
def models_delete(body: ModelDownloadRequest, user=Depends(auth.get_current_user)):
    try:
        return hub.remove_model(body.model_id.strip())
    except Exception as exc:
        raise HTTPException(400, str(exc))

@app.get("/api/models/jobs")
def models_jobs(user=Depends(auth.get_current_user)):
    return {"jobs": hub.jobs()}

@app.get("/api/models/progress")
def models_progress(request: Request, user=Depends(auth.get_current_user)):
    def stream():
        last = ""
        while True:
            payload = {"jobs": hub.jobs(), "active": hub.active_job(),
                       "disk": hub.disk_free()}
            blob = json.dumps(payload, sort_keys=True, default=str)
            if blob != last:
                last = blob
                yield sse("progress", payload)
            if not hub.active_job():
                yield sse("idle", {})
                break
            time.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)

# --------------------------------------------------------------------------
# image generation
# --------------------------------------------------------------------------
def _imagegen_context(body: ImageGenRequest, user: dict):
    project_id = body.project_id
    if body.chat_id is not None:
        chat = db.get_chat(user["id"], body.chat_id)
        if not chat:
            raise HTTPException(404, "Chat not found")
        if project_id is not None and project_id != chat["project_id"]:
            raise HTTPException(400, "Chat does not belong to the selected project")
        project_id = chat["project_id"]
    elif project_id is not None and not db.get_project(user["id"], project_id):
        raise HTTPException(404, "Project not found")
    return body.chat_id, project_id


@app.post("/api/imagegen")
def do_imagegen(body: ImageGenRequest, user=Depends(auth.get_current_user)):
    chat_id, project_id = _imagegen_context(body, user)
    if not hub.is_cached(CONFIG.get("image_model") or ""):
        raise HTTPException(400, "Image model not downloaded. Open Models -> install it.")
    job = imagegen.Job(body.prompt)
    imagegen.JOBS[job.id] = job
    try:
        with MODEL_SESSION_LOCK:
            _prepare_image_generation()
            result = imagegen.generate(body.prompt, body.negative, body.steps, body.width,
                                       body.height, body.guidance, body.seed, body.model_id,
                                       job=job)
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        raise HTTPException(500, f"Image generation failed: {exc}")
    image_id = db.add_generated_image(user["id"], project_id, body.prompt, body.negative,
                                      result["filename"], result["width"], result["height"],
                                      result["steps"], result["seed"], result["model"])
    result["id"] = image_id
    if chat_id is not None:
        db.add_chat_message(chat_id, project_id, "assistant",
                            f"Generated image: {body.prompt}", meta={"images": [result]})
    elif project_id is not None:
        db.add_message(project_id, "assistant", f"Generated image: {body.prompt}",
                       meta={"images": [result]})
    return result

@app.post("/api/imagegen/stream")
def do_imagegen_stream(body: ImageGenRequest, user=Depends(auth.get_current_user)):
    chat_id, project_id = _imagegen_context(body, user)
    def stream():
        job = imagegen.Job(body.prompt)
        imagegen.JOBS[job.id] = job
        yield sse("job", {"job_id": job.id, "prompt": body.prompt})
        model_id = body.model_id or CONFIG.get("image_model") or ""
        if not model_id:
            yield sse("error", {"message": "No image model is configured."})
            job.status = "error"
            job.error = "No image model is configured."
            return

        if not hub.is_cached(model_id):
            yield sse("status", {
                "stage": "First-time setup: downloading the image model and its text encoder, VAE, and pipeline files from Hugging Face. This may take a while."
            })
            download = hub.download(model_id, "image")
            if download.get("ok") is False:
                job.status = "error"
                job.error = download.get("error", "Image model download could not start.")
                yield sse("error", {"message": job.error})
                return
            while True:
                download_job = next(
                    (entry for entry in hub.jobs() if entry["repo_id"] == model_id),
                    None,
                )
                if download_job is None:
                    job.status = "error"
                    job.error = "Image model download ended without a status."
                    yield sse("error", {"message": job.error})
                    return
                yield sse("progress", {
                    "stage": "download",
                    "status": download_job["status"],
                    "downloaded": download_job.get("downloaded", 0),
                    "total": download_job.get("total", 0),
                    "message": "Downloading model files and required components…",
                })
                if download_job["status"] == "ready":
                    yield sse("status", {"stage": "Image model ready; loading pipeline"})
                    break
                if download_job["status"] == "error":
                    job.status = "error"
                    job.error = download_job.get("error", "Image model download failed.")
                    yield sse("error", {"message": job.error})
                    return
                if download_job["status"] == "cancelled":
                    job.status = "cancelled"
                    yield sse("error", {"message": "Image model download was cancelled."})
                    return
                time.sleep(0.5)

        holder: dict = {}
        def worker():
            try:
                holder["result"] = imagegen.generate(
                    body.prompt, body.negative, body.steps, body.width, body.height,
                    body.guidance, body.seed, model_id, job=job)
            except Exception as exc:
                holder["error"] = str(exc)
                job.status = "error"
                job.error = str(exc)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while thread.is_alive():
            yield sse("progress", {"stage": "generation", "progress": job.progress, "step": job.step,
                                   "total_steps": job.total_steps, "status": job.status})
            time.sleep(0.4)
        thread.join(timeout=5)
        if holder.get("error"):
            yield sse("error", {"message": holder["error"]})
            return
        result = holder.get("result") or {}
        image_id = db.add_generated_image(
            user["id"], project_id, body.prompt, body.negative, result.get("filename", ""),
            result.get("width", 0), result.get("height", 0), result.get("steps", 0),
            result.get("seed", 0), result.get("model", ""))
        result["id"] = image_id
        if chat_id is not None:
            db.add_chat_message(chat_id, project_id, "assistant",
                                f"Generated image: {body.prompt}", meta={"images": [result]})
        elif project_id is not None:
            db.add_message(project_id, "assistant", f"Generated image: {body.prompt}",
                           meta={"images": [result]})
        yield sse("image", {"images": [result]})
        yield sse("done", {})
    return StreamingResponse(
        _exclusive_model_events(stream, _prepare_image_generation),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )

@app.get("/api/imagegen/progress/{job_id}")
def imagegen_progress(job_id: str, user=Depends(auth.get_current_user)):
    job = imagegen.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {"progress": job.progress, "status": job.status, "step": job.step,
            "total_steps": job.total_steps, "url": job.url, "error": job.error}

@app.get("/api/imagegen/gallery")
def imagegen_gallery(user=Depends(auth.get_current_user)):
    rows = db.list_generated_images(user["id"], 100)
    return {"images": [{"id": r["id"], "prompt": r["prompt"], "url": f"/generated/{r['filename']}",
                        "filename": r["filename"], "width": r["width"], "height": r["height"],
                        "steps": r["steps"], "seed": r["seed"], "model": r["model"],
                        "created_at": r["created_at"]} for r in rows]}

@app.delete("/api/imagegen/{image_id}")
def imagegen_delete(image_id: int, user=Depends(auth.get_current_user)):
    row = db.delete_generated_image(user["id"], image_id)
    if not row:
        raise HTTPException(404, "Image not found")
    return {"ok": True}

# --------------------------------------------------------------------------
# engine control
# --------------------------------------------------------------------------
@app.post("/api/engine/unload")
def engine_unload(user=Depends(auth.get_current_user)):
    with MODEL_SESSION_LOCK:
        llm.unload()
        vision.unload()
        imagegen.unload()
        rag.unload_embedder()
    return {"ok": True}

@app.post("/api/engine/load")
def engine_load(user=Depends(auth.get_current_user)):
    with MODEL_SESSION_LOCK:
        _prepare_text_generation()
        if CONFIG.get("active_backend") == "llama":
            path = CONFIG.get("active_model_path")
            if path:
                llm.load_gguf_model(path, force=True)
        else:
            llm.load(force=True)
    return {"ok": True, "llm": llm.info()}

# --------------------------------------------------------------------------
# Startup Event
# --------------------------------------------------------------------------
@app.on_event("startup")
def startup() -> None:
    db.init_db()
    threading.Thread(target=_warmup_text_model, daemon=True).start()