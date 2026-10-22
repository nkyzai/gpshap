import json
import time
from contextlib import closing
from pathlib import Path

import psutil
import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import agent, auth, db, hub, llm, rag
from .config import CONFIG, ROOT


router = APIRouter(tags=["platform"])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


class ProjectUpdateIn(BaseModel):
    name: str | None = None
    system_prompt: str | None = None
    pinned: bool | None = None
    archived: bool | None = None
    sort_order: int | None = None


class ReorderIn(BaseModel):
    project_ids: list[int]


class MessageUpdateIn(BaseModel):
    content: str


class RegenerateIn(BaseModel):
    use_rag: bool = True
    agent_mode: bool = True


# --------------------------------------------------------------------------
# system resources
# --------------------------------------------------------------------------
@router.get("/system/resources")
def system_resources(user=Depends(auth.get_current_user)):
    cpu = psutil.cpu_percent(interval=0.25)
    mem = psutil.virtual_memory()

    try:
        disk = psutil.disk_usage(str(ROOT))
    except Exception:
        disk = None

    data = {
        "ts": time.time(),
        "cpu_percent": cpu,
        "ram_percent": mem.percent,
        "ram_used_gb": round(mem.used / 1024 ** 3, 2),
        "ram_total_gb": round(mem.total / 1024 ** 3, 2),
        "disk": None,
        "gpu": {
            "available": False,
            "name": "",
            "total_mb": 0,
            "used_mb": 0,
            "allocated_mb": 0,
            "reserved_mb": 0,
            "memory_percent": 0,
        },
    }

    if disk:
        data["disk"] = {
            "free_gb": round(disk.free / 1024 ** 3, 2),
            "used_gb": round(disk.used / 1024 ** 3, 2),
            "total_gb": round(disk.total / 1024 ** 3, 2),
            "percent": disk.percent,
        }

    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            used = max(0, total - free)
            props = torch.cuda.get_device_properties(0)

            data["gpu"] = {
                "available": True,
                "name": props.name,
                "total_mb": round(total / 1024 ** 2, 1),
                "used_mb": round(used / 1024 ** 2, 1),
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024 ** 2, 1),
                "memory_percent": round((used / total) * 100.0, 1) if total else 0,
            }
        except Exception as exc:
            data["gpu"]["error"] = str(exc)

    return data


# --------------------------------------------------------------------------
# usage / token budget
# --------------------------------------------------------------------------
def _count_tokens(text: str) -> int:
    try:
        tokenizer = llm._state.get("tokenizer")
        if tokenizer is not None:
            return len(tokenizer.encode(text or "", add_special_tokens=False))
    except Exception:
        pass

    return max(1, len(text or "") // 4)


def _context_limit() -> int:
    ctx = llm.context_length()
    if ctx:
        return int(ctx)

    model_id = CONFIG.get("model", "")
    preset = hub.PRESET_MAP.get(model_id, {})
    return int(preset.get("context_tokens") or 32768)


@router.get("/usage")
def usage(project_id: int | None = None, user=Depends(auth.get_current_user)):
    with closing(db.get_conn()) as conn:
        total_projects = conn.execute(
            "SELECT COUNT(*) c FROM projects WHERE user_id = ?", (user["id"],)
        ).fetchone()["c"]

        archived_projects = conn.execute(
            "SELECT COUNT(*) c FROM projects WHERE user_id = ? AND archived = 1", (user["id"],)
        ).fetchone()["c"]

        pinned_projects = conn.execute(
            "SELECT COUNT(*) c FROM projects WHERE user_id = ? AND pinned = 1", (user["id"],)
        ).fetchone()["c"]

        total_messages = conn.execute(
            """
            SELECT COUNT(*) c FROM messages
            JOIN projects ON projects.id = messages.project_id
            WHERE projects.user_id = ?
            """,
            (user["id"],),
        ).fetchone()["c"]

        assistant_messages = conn.execute(
            """
            SELECT COUNT(*) c FROM messages
            JOIN projects ON projects.id = messages.project_id
            WHERE projects.user_id = ? AND messages.role = 'assistant'
            """,
            (user["id"],),
        ).fetchone()["c"]

        user_messages = conn.execute(
            """
            SELECT COUNT(*) c FROM messages
            JOIN projects ON projects.id = messages.project_id
            WHERE projects.user_id = ? AND messages.role = 'user'
            """,
            (user["id"],),
        ).fetchone()["c"]

        file_count = conn.execute(
            """
            SELECT COUNT(*) c FROM files
            JOIN projects ON projects.id = files.project_id
            WHERE projects.user_id = ?
            """,
            (user["id"],),
        ).fetchone()["c"]

        image_count = conn.execute(
            "SELECT COUNT(*) c FROM generated_images WHERE user_id = ?", (user["id"],)
        ).fetchone()["c"]

    current = None

    if project_id:
        project = db.get_project(user["id"], project_id)
        if not project:
            raise HTTPException(404, "Project not found")

        history_limit = int(CONFIG.get("history_messages", 12))
        msgs = db.list_messages(project_id, history_limit)
        used = 120  # system/tool overhead approximation

        for row in msgs:
            used += _count_tokens(row["content"]) + 4

        context_limit = _context_limit()
        reserved = int(CONFIG.get("max_new_tokens", 768))
        available = max(0, context_limit - used - reserved)

        with closing(db.get_conn()) as conn:
            project_messages = conn.execute(
                "SELECT COUNT(*) c FROM messages WHERE project_id = ?", (project_id,)
            ).fetchone()["c"]

        current = {
            "project_id": project_id,
            "project_name": project["name"],
            "message_count": project_messages,
            "history_messages": len(msgs),
            "context_limit": context_limit,
            "tokens_used": used,
            "reserved_for_reply": reserved,
            "tokens_available": available,
        }

    return {
        "projects": {
            "total": total_projects,
            "archived": archived_projects,
            "pinned": pinned_projects,
        },
        "chats": {
            "total_messages": total_messages,
            "assistant_messages": assistant_messages,
            "user_messages": user_messages,
        },
        "files": {"count": file_count},
        "images": {"count": image_count},
        "model": CONFIG.get("model", ""),
        "current": current,
    }


# --------------------------------------------------------------------------
# projects v2 + management
# --------------------------------------------------------------------------
def _project_dict(row) -> dict:
    d = dict(row)
    d["pinned"] = bool(d.get("pinned"))
    d["archived"] = bool(d.get("archived"))
    return d


@router.get("/projects/v2")
def projects_v2(user=Depends(auth.get_current_user)):
    rows = db.list_projects(user["id"])
    return {"projects": [_project_dict(r) for r in rows]}


@router.patch("/projects/{project_id}")
def update_project(project_id: int, body: ProjectUpdateIn, user=Depends(auth.get_current_user)):
    project = db.get_project(user["id"], project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    patch = body.model_dump(exclude_none=True)
    if not patch:
        return _project_dict(project)

    if "name" in patch:
        patch["name"] = str(patch["name"]).strip()
        if not patch["name"]:
            raise HTTPException(400, "Project name cannot be empty")

    db.update_project_fields(project_id, patch)
    return _project_dict(db.get_project(user["id"], project_id))


@router.post("/projects/reorder")
def reorder_projects(body: ReorderIn, user=Depends(auth.get_current_user)):
    db.reorder_projects(user["id"], body.project_ids)
    return {"ok": True}


@router.post("/projects/{project_id}/duplicate")
def duplicate_project(project_id: int, user=Depends(auth.get_current_user)):
    project = db.get_project(user["id"], project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    new_id = db.duplicate_project(user["id"], project_id)
    return {"ok": True, "id": new_id}


@router.delete("/projects/{project_id}/messages")
def clear_project_messages(project_id: int, user=Depends(auth.get_current_user)):
    project = db.get_project(user["id"], project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    db.delete_project_messages(project_id)
    return {"ok": True}


@router.get("/projects/{project_id}/export")
def export_project(project_id: int, user=Depends(auth.get_current_user)):
    project = db.get_project(user["id"], project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    messages = []
    for row in db.list_messages(project_id, 1000000):
        try:
            meta = json.loads(row["meta"] or "{}")
        except Exception:
            meta = {}

        messages.append({
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "number": row["number"] if "number" in row.keys() else 0,
            "meta": meta,
            "created_at": row["created_at"],
        })

    files = []
    for row in db.list_files(project_id):
        files.append({
            "id": row["id"],
            "filename": row["filename"],
            "mime": row["mime"],
            "kind": row["kind"],
            "indexed_chars": len(row["text_extracted"] or ""),
            "created_at": row["created_at"],
        })

    payload = {
        "app": "gpshap",
        "version": 2,
        "exported_at": time.time(),
        "project": _project_dict(project),
        "messages": messages,
        "files": files,
    }

    filename = f"gpshap_project_{project_id}.json"
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/projects/{project_id}/messages/v2")
def messages_v2(project_id: int, user=Depends(auth.get_current_user)):
    project = db.get_project(user["id"], project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    out = []
    for row in db.list_messages(project_id, 1000):
        try:
            meta = json.loads(row["meta"] or "{}")
        except Exception:
            meta = {}

        out.append({
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "number": row["number"] if "number" in row.keys() else 0,
            "meta": meta,
            "created_at": row["created_at"],
        })

    return {"messages": out}


# --------------------------------------------------------------------------
# message edit/delete/regenerate
# --------------------------------------------------------------------------
@router.patch("/messages/{message_id}")
def update_message(message_id: int, body: MessageUpdateIn, user=Depends(auth.get_current_user)):
    row = db.update_message_content(message_id, user["id"], body.content, truncate_after=True)
    if not row:
        raise HTTPException(404, "Message not found")
    return {"ok": True, "message_id": message_id}


@router.delete("/messages/{message_id}")
def delete_message(message_id: int, user=Depends(auth.get_current_user)):
    ok = db.delete_message(message_id, user["id"])
    if not ok:
        raise HTTPException(404, "Message not found")
    return {"ok": True}


@router.post("/projects/{project_id}/regenerate/stream")
def regenerate_stream(project_id: int, body: RegenerateIn, user=Depends(auth.get_current_user)):
    project = db.get_project(user["id"], project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    last_user = db.get_last_user_message(project_id)
    if not last_user:
        raise HTTPException(400, "No user message to regenerate from")

    db.delete_trailing_assistant_messages(project_id)

    history = []
    for row in db.list_messages(project_id, 60):
        if row["role"] in ("user", "assistant"):
            history.append({"role": row["role"], "content": row["content"]})

    history = history[-int(CONFIG.get("history_messages", 12)):]
    image_paths = [r["stored_path"] for r in db.project_image_files(project_id, 2)]

    def source():
        yield sse("start", {"project_id": project_id, "regenerate": True})
        try:
            for event in agent.run(
                project,
                project_id,
                last_user["content"],
                history,
                user,
                body.use_rag,
                body.agent_mode,
                image_paths,
            ):
                yield sse(event["event"], event["data"])
        except Exception as exc:
            yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            yield sse("end", {})

    return StreamingResponse(source(), media_type="text/event-stream", headers=SSE_HEADERS)


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
@router.delete("/files/{file_id}")
def delete_file(file_id: int, user=Depends(auth.get_current_user)):
    row = db.get_file(file_id)
    if not row:
        raise HTTPException(404, "File not found")

    project = db.get_project(user["id"], row["project_id"])
    if not project:
        raise HTTPException(404, "File not found")

    try:
        Path(row["stored_path"]).unlink(missing_ok=True)
    except Exception:
        pass

    try:
        rag.remove_file(row["project_id"], file_id)
    except Exception as exc:
        print(f"[gpshap] rag remove file failed: {exc}", flush=True)

    db.delete_file(file_id)
    return {"ok": True}