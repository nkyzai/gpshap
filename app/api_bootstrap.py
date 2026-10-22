import traceback

from fastapi import APIRouter

from .auth import router as auth_router


def mount_core_api(app):
    """
    Force-mount core API routes at the top of the route table.

    This prevents static/catch-all mounts from swallowing API requests and
    guarantees /api/login, /api/register, /api/system/resources, /api/usage, etc.
    are reachable.
    """
    if getattr(app.state, "gpshap_core_api_mounted", False):
        return

    api = APIRouter()

    @api.get("/api/ping")
    def ping():
        return {"ok": True, "service": "gpshap"}

    @api.get("/api/debug/routes")
    def debug_routes():
        routes = []
        for route in app.routes:
            routes.append({
                "path": getattr(route, "path", str(route)),
                "methods": sorted(getattr(route, "methods", []) or []),
            })
        return {"count": len(routes), "routes": routes}

    # Always mount auth.
    api.include_router(auth_router, prefix="/api")

    # Mount platform if available. If platform.py has an import error, expose it.
    platform_ok = True
    platform_error = ""

    try:
        from .platform import router as platform_router
        api.include_router(platform_router, prefix="/api")
    except Exception as exc:
        platform_ok = False
        platform_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

        @api.get("/api/platform-error")
        def platform_error():
            return {
                "ok": False,
                "error": platform_error,
                "hint": "Fix app/platform.py, then restart run.bat.",
            }

    # Insert these API routes at the very beginning so they win over any
    # later catch-all/static mounts.
    try:
        app.router.routes[:0] = list(api.routes)
    except Exception:
        # Fallback for unusual FastAPI/Starlette versions.
        app.include_router(api)

    app.state.gpshap_core_api_mounted = True
    print(f"[gpshap] core API mounted: platform_ok={platform_ok}", flush=True)

    if not platform_ok:
        print(f"[gpshap] platform import error: {platform_error}", flush=True)