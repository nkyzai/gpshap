import json

from fastapi import APIRouter, Depends, HTTPException, Request

from . import tools
from .auth import get_current_user


router = APIRouter(tags=["mcp"])


@router.post("/mcp")
async def mcp_endpoint(request: Request, user=Depends(get_current_user)):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}

    def ok(result):
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": result,
        }

    def error(code: int, message: str):
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {
                "code": code,
                "message": message,
            },
        }

    if method == "initialize":
        return ok(
            {
                "protocolVersion": "2025-03-26",
                "serverInfo": {
                    "name": "gapshap",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "tools": {},
                },
            }
        )

    if method == "tools/list":
        return ok({"tools": tools.MCP_TOOLS})

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        fallback_project_id = params.get("project_id")

        if not tool_name:
            return error(-32602, "Missing tool name")

        output = tools.execute_tool(
            tool_name,
            arguments,
            user=user,
            project_id=fallback_project_id,
        )

        return ok(
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(output, ensure_ascii=False),
                    }
                ],
                "isError": False,
            }
        )

    return error(-32601, f"Method not found: {method}")