from fastapi import APIRouter, Depends, HTTPException, Request, Response

from . import db
from .schemas import AuthRequest


router = APIRouter(tags=["auth"])

COOKIE_NAME = "gapshap_session"
SESSION_MAX_AGE = 30 * 86400


def get_current_user(request: Request):
    token = request.cookies.get(COOKIE_NAME)

    auth_header = request.headers.get("authorization", "")
    if not token and auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = db.get_user_from_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    return user


def start_session(response: Response, user_id: int) -> str:
    token = db.create_session(user_id)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
    )
    return token


def extract_token(request: Request) -> str | None:
    token = request.cookies.get(COOKIE_NAME)

    auth_header = request.headers.get("authorization", "")
    if not token and auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()

    return token


@router.post("/register")
def register(body: AuthRequest, response: Response):
    username = body.username.strip()
    password = body.password

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    if db.get_user_by_username(username):
        raise HTTPException(status_code=409, detail="Username already exists")

    user_id = db.create_user(username, password)
    token = start_session(response, user_id)

    return {
        "ok": True,
        "user": {
            "id": user_id,
            "username": username,
        },
        "token": token,
    }


@router.post("/login")
def login(body: AuthRequest, response: Response):
    username = body.username.strip()
    password = body.password

    user = db.get_user_by_username(username)
    if not user or not db.verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = start_session(response, user["id"])

    return {
        "ok": True,
        "user": {
            "id": user["id"],
            "username": user["username"],
        },
        "token": token,
    }


@router.post("/logout")
def logout(request: Request, response: Response):
    token = extract_token(request)
    if token:
        db.delete_session(token)

    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@router.get("/me")
def me(user=Depends(get_current_user)):
    return {
        "user": {
            "id": user["id"],
            "username": user["username"],
        }
    }