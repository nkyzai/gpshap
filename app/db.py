import hashlib
import json
import secrets
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

from .config import DATA_DIR, GENERATED_DIR


DB_PATH = DATA_DIR / "gapshap.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Safely add a column if it does not exist."""
    try:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    except Exception:
        pass


def init_db() -> None:
    with closing(get_conn()) as conn:
        # 1. Create base tables if they don't exist
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                system_prompt TEXT DEFAULT '',
                sort_order INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0,
                archived INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                chat_id INTEGER REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                meta TEXT DEFAULT '{}',
                number INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                mime TEXT DEFAULT '',
                kind TEXT DEFAULT 'file',
                text_extracted TEXT DEFAULT '',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                kind TEXT DEFAULT 'llm',
                size_bytes INTEGER DEFAULT 0,
                context_tokens INTEGER DEFAULT 0,
                params TEXT DEFAULT '',
                pipeline_tag TEXT DEFAULT '',
                capabilities TEXT DEFAULT '[]',
                status TEXT DEFAULT 'ready',
                progress REAL DEFAULT 1,
                error TEXT DEFAULT '',
                last_used REAL DEFAULT 0,
                added_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS generated_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                project_id INTEGER,
                prompt TEXT DEFAULT '',
                negative TEXT DEFAULT '',
                filename TEXT NOT NULL,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                steps INTEGER DEFAULT 0,
                seed INTEGER DEFAULT 0,
                model TEXT DEFAULT '',
                created_at REAL NOT NULL
            );

            """
        )
        conn.execute("DROP TABLE IF EXISTS chat_exports")
        conn.execute("DROP TABLE IF EXISTS chat_export_consents")

        # 2. Safely upgrade schema for GGUF support
        # These lines will fail silently if columns already exist
        _ensure_column(conn, "models", "gguf_path", "TEXT DEFAULT ''")
        _ensure_column(conn, "models", "backend", "TEXT DEFAULT 'transformers'")
        
        # Ensure other recent columns exist too
        _ensure_column(conn, "projects", "system_prompt", "TEXT DEFAULT ''")
        _ensure_column(conn, "projects", "sort_order", "INTEGER DEFAULT 0")
        _ensure_column(conn, "projects", "pinned", "INTEGER DEFAULT 0")
        _ensure_column(conn, "projects", "archived", "INTEGER DEFAULT 0")
        _ensure_column(conn, "projects", "created_at", "REAL NOT NULL DEFAULT 0")
        _ensure_column(conn, "messages", "meta", "TEXT DEFAULT '{}'")
        _ensure_column(conn, "messages", "number", "INTEGER DEFAULT 0")
        _ensure_column(conn, "messages", "chat_id", "INTEGER REFERENCES chats(id) ON DELETE CASCADE")
        _ensure_column(conn, "files", "kind", "TEXT DEFAULT 'file'")

        message_columns = {
            row["name"]: row for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if message_columns["project_id"]["notnull"]:
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute(
                """
                CREATE TABLE messages_migrated (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    chat_id INTEGER REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta TEXT DEFAULT '{}',
                    number INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO messages_migrated(id, project_id, chat_id, role, content, meta, number, created_at)
                SELECT id, project_id, chat_id, role, content, meta, number, created_at FROM messages
                """
            )
            conn.execute("DROP TABLE messages")
            conn.execute("ALTER TABLE messages_migrated RENAME TO messages")
            conn.execute("PRAGMA foreign_keys=ON")

        # Existing projects previously represented a single conversation.
        conn.execute(
            """
            INSERT INTO chats(user_id, project_id, title, created_at, updated_at)
            SELECT p.user_id, p.id, p.name, p.created_at, p.created_at
            FROM projects p
            WHERE NOT EXISTS (
                SELECT 1 FROM chats c WHERE c.project_id = p.id
            )
            """
        )
        conn.execute(
            """
            UPDATE messages
            SET chat_id = (
                SELECT c.id FROM chats c
                WHERE c.project_id = messages.project_id
                ORDER BY c.id LIMIT 1
            )
            WHERE chat_id IS NULL AND project_id IS NOT NULL
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_project ON chats(user_id, project_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id, id)")

        # 3. Backfill assistant response numbers for old rows (if any)
        rows = conn.execute(
            "SELECT id FROM messages WHERE role = 'assistant' AND COALESCE(number, 0) = 0 ORDER BY id"
        ).fetchall()
        if rows:
            current = conn.execute(
                "SELECT COALESCE(MAX(number), 0) FROM messages WHERE role = 'assistant'"
            ).fetchone()[0]
            for row in rows:
                current += 1
                conn.execute("UPDATE messages SET number = ? WHERE id = ?", (current, row["id"]))

        conn.commit()


# --------------------------------------------------------------------------
# users / sessions
# --------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt),
        n=16384,
        r=8,
        p=1,
        dklen=64,
    )
    return f"scrypt$16384$8$1${salt}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        _scheme, n, r, p, salt, expected = stored_hash.split("$")
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=64,
        )
        return secrets.compare_digest(derived.hex(), expected)
    except Exception:
        return False


def create_user(username: str, password: str) -> int:
    with closing(get_conn()) as conn:
        cur = conn.execute(
            "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, hash_password(password), time.time()),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_user_by_username(username: str):
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def update_user_password(user_id: int, password: str) -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(password), user_id),
        )
        conn.commit()


def create_session(user_id: int) -> str:
    token = uuid.uuid4().hex
    now = time.time()
    with closing(get_conn()) as conn:
        conn.execute(
            "INSERT INTO sessions(token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + 30 * 86400),
        )
        conn.commit()
    return token


def get_user_from_session(token: str):
    with closing(get_conn()) as conn:
        row = conn.execute(
            """
            SELECT users.* FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ? AND sessions.expires_at > ?
            """,
            (token, time.time()),
        ).fetchone()
        return dict(row) if row else None


def delete_session(token: str) -> None:
    with closing(get_conn()) as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


def delete_other_sessions(user_id: int, current_token: str) -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            "DELETE FROM sessions WHERE user_id = ? AND token != ?",
            (user_id, current_token),
        )
        conn.commit()


# --------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------
def create_project(user_id: int, name: str, system_prompt: str = "") -> int:
    with closing(get_conn()) as conn:
        cur = conn.execute(
            """
            INSERT INTO projects(user_id, name, system_prompt, sort_order, pinned, archived, created_at)
            VALUES (?, ?, ?, ?, 0, 0, ?)
            """,
            (user_id, name, system_prompt or "", int(time.time() * 1000), time.time()),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_projects(user_id: int):
    with closing(get_conn()) as conn:
        return conn.execute(
            """
            SELECT * FROM projects
            WHERE user_id = ?
            ORDER BY pinned DESC, COALESCE(sort_order, 1000000000) ASC, created_at DESC
            """,
            (user_id,),
        ).fetchall()


def create_chat(user_id: int, project_id: int | None = None, title: str = "New chat") -> int:
    now = time.time()
    with closing(get_conn()) as conn:
        if project_id is not None:
            project = conn.execute(
                "SELECT id FROM projects WHERE id = ? AND user_id = ?",
                (project_id, user_id),
            ).fetchone()
            if not project:
                raise ValueError("project not found")
        cur = conn.execute(
            """
            INSERT INTO chats(user_id, project_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, project_id, title.strip() or "New chat", now, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_chats(user_id: int):
    with closing(get_conn()) as conn:
        return conn.execute(
            """
            SELECT chats.*, projects.name AS project_name
            FROM chats
            LEFT JOIN projects ON projects.id = chats.project_id
            WHERE chats.user_id = ?
            ORDER BY chats.updated_at DESC, chats.id DESC
            """,
            (user_id,),
        ).fetchall()


def get_chat(user_id: int, chat_id: int):
    with closing(get_conn()) as conn:
        row = conn.execute(
            """
            SELECT chats.*, projects.name AS project_name,
                   projects.system_prompt AS project_system_prompt
            FROM chats
            LEFT JOIN projects ON projects.id = chats.project_id
            WHERE chats.id = ? AND chats.user_id = ?
            """,
            (chat_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def update_chat_title(user_id: int, chat_id: int, title: str) -> bool:
    with closing(get_conn()) as conn:
        cur = conn.execute(
            "UPDATE chats SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (title.strip(), time.time(), chat_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def delete_chat(user_id: int, chat_id: int) -> bool:
    with closing(get_conn()) as conn:
        cur = conn.execute(
            "DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
        )
        conn.commit()
        return cur.rowcount > 0


def duplicate_chat(user_id: int, chat_id: int) -> int:
    with closing(get_conn()) as conn:
        chat = conn.execute(
            "SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
        ).fetchone()
        if not chat:
            raise ValueError("chat not found")
        now = time.time()
        cur = conn.execute(
            """
            INSERT INTO chats(user_id, project_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, chat["project_id"], f"{chat['title']} (copy)", now, now),
        )
        new_chat_id = int(cur.lastrowid)
        conn.execute(
            """
            INSERT INTO messages(project_id, chat_id, role, content, meta, number, created_at)
            SELECT project_id, ?, role, content, meta, number, ?
            FROM messages WHERE chat_id = ? ORDER BY id
            """,
            (new_chat_id, now, chat_id),
        )
        conn.commit()
        return new_chat_id


def clear_chat(chat_id: int) -> None:
    with closing(get_conn()) as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.commit()


def get_project(user_id: int, project_id: int):
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def delete_project(user_id: int, project_id: int) -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            "DELETE FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        )
        conn.commit()


def update_project_fields(project_id: int, patch: dict) -> None:
    allowed = {"name", "system_prompt", "pinned", "archived", "sort_order"}
    fields = {}
    for key, value in patch.items():
        if key not in allowed:
            continue
        if key in ("pinned", "archived"):
            fields[key] = 1 if value else 0
        else:
            fields[key] = value
    
    if not fields:
        return
    
    sets = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [project_id]
    
    with closing(get_conn()) as conn:
        conn.execute(f"UPDATE projects SET {sets} WHERE id = ?", values)
        conn.commit()


def reorder_projects(user_id: int, project_ids: list[int]) -> None:
    with closing(get_conn()) as conn:
        for idx, pid in enumerate(project_ids):
            conn.execute(
                "UPDATE projects SET sort_order = ? WHERE id = ? AND user_id = ?",
                (idx, pid, user_id),
            )
        conn.commit()


def duplicate_project(user_id: int, project_id: int) -> int:
    with closing(get_conn()) as conn:
        project = conn.execute(
            "SELECT * FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        ).fetchone()
        
        if not project:
            raise ValueError("project not found")
        
        cur = conn.execute(
            """
            INSERT INTO projects(user_id, name, system_prompt, sort_order, pinned, archived, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (
                user_id,
                project["name"] + " (copy)",
                project["system_prompt"] or "",
                int(time.time() * 1000),
                project["pinned"] or 0,
                time.time(),
            ),
        )
        new_id = int(cur.lastrowid)
        source_chats = conn.execute(
            "SELECT * FROM chats WHERE user_id = ? AND project_id = ? ORDER BY id",
            (user_id, project_id),
        ).fetchall()
        for source_chat in source_chats:
            now = time.time()
            chat_cur = conn.execute(
                """
                INSERT INTO chats(user_id, project_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, new_id, source_chat["title"], now, now),
            )
            new_chat_id = int(chat_cur.lastrowid)
            conn.execute(
                """
                INSERT INTO messages(project_id, chat_id, role, content, meta, number, created_at)
                SELECT ?, ?, role, content, meta, number, ?
                FROM messages WHERE chat_id = ? ORDER BY id
                """,
                (new_id, new_chat_id, now, source_chat["id"]),
            )
            assistant_rows = conn.execute(
                "SELECT id FROM messages WHERE chat_id = ? AND role = 'assistant' ORDER BY id",
                (new_chat_id,),
            ).fetchall()
            for i, row in enumerate(assistant_rows, start=1):
                conn.execute("UPDATE messages SET number = ? WHERE id = ?", (i, row["id"]))

        legacy_messages = conn.execute(
            "SELECT 1 FROM messages WHERE project_id = ? AND chat_id IS NULL LIMIT 1",
            (project_id,),
        ).fetchone()
        if legacy_messages:
            now = time.time()
            chat_cur = conn.execute(
                """
                INSERT INTO chats(user_id, project_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, new_id, project["name"] + " (copy)", now, now),
            )
            new_chat_id = int(chat_cur.lastrowid)
            conn.execute(
                """
                INSERT INTO messages(project_id, chat_id, role, content, meta, number, created_at)
                SELECT ?, ?, role, content, meta, number, ?
                FROM messages WHERE project_id = ? AND chat_id IS NULL ORDER BY id
                """,
                (new_id, new_chat_id, now, project_id),
            )
        
        conn.commit()
        return new_id


def delete_project_messages(project_id: int) -> None:
    with closing(get_conn()) as conn:
        conn.execute("DELETE FROM messages WHERE project_id = ?", (project_id,))
        conn.commit()


# --------------------------------------------------------------------------
# messages
# --------------------------------------------------------------------------
def _next_assistant_number(conn: sqlite3.Connection, project_id: int) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(MAX(number), 0) + 1 AS next_number
        FROM messages
        WHERE project_id = ? AND role = 'assistant'
        """,
        (project_id,),
    ).fetchone()
    return int(row["next_number"]) if row else 1


def add_message(project_id: int | None, role: str, content: str, meta: dict | None = None,
                chat_id: int | None = None) -> int:
    with closing(get_conn()) as conn:
        number = 0
        if role == "assistant":
            if chat_id is not None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(number), 0) + 1 FROM messages WHERE chat_id = ? AND role = 'assistant'",
                    (chat_id,),
                ).fetchone()
                number = int(row[0])
            elif project_id is not None:
                number = _next_assistant_number(conn, project_id)
        
        cur = conn.execute(
            """
            INSERT INTO messages(project_id, chat_id, role, content, meta, number, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                chat_id,
                role,
                content,
                json.dumps(meta or {}, ensure_ascii=False),
                number,
                time.time(),
            ),
        )
        conn.commit()
        if chat_id is not None:
            conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (time.time(), chat_id))
            if role == "user":
                conn.execute(
                    """
                    UPDATE chats SET title = ?, updated_at = ?
                    WHERE id = ? AND title = 'New chat'
                    """,
                    (content.strip()[:60] or "New chat", time.time(), chat_id),
                )
            conn.commit()
        return int(cur.lastrowid)


def add_chat_message(chat_id: int, project_id: int | None, role: str, content: str,
                     meta: dict | None = None) -> int:
    return add_message(project_id, role, content, meta, chat_id=chat_id)


def list_chat_messages(chat_id: int, limit: int = 200):
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return list(reversed(rows))


def get_last_chat_user_message(chat_id: int):
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
        return dict(row) if row else None


def delete_trailing_chat_assistant_messages(chat_id: int) -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            """
            DELETE FROM messages
            WHERE chat_id = ? AND role = 'assistant'
              AND id > COALESCE(
                  (SELECT MAX(id) FROM messages WHERE chat_id = ? AND role = 'user'), 0
              )
            """,
            (chat_id, chat_id),
        )
        conn.commit()


def list_messages(project_id: int, limit: int = 200):
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE project_id = ? ORDER BY id DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()
        return list(reversed(rows))


def get_message(message_id: int):
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return dict(row) if row else None


def get_last_user_message(project_id: int):
    with closing(get_conn()) as conn:
        row = conn.execute(
            """
            SELECT * FROM messages
            WHERE project_id = ? AND role = 'user'
            ORDER BY id DESC LIMIT 1
            """,
            (project_id,),
        ).fetchone()
        return dict(row) if row else None


def delete_trailing_assistant_messages(project_id: int) -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            """
            DELETE FROM messages
            WHERE project_id = ?
            AND role = 'assistant'
            AND id > COALESCE(
                (SELECT MAX(id) FROM messages WHERE project_id = ? AND role = 'user'),
                0
            )
            """,
            (project_id, project_id),
        )
        conn.commit()


def update_message_content(message_id: int, user_id: int, content: str,
                           truncate_after: bool = True):
    with closing(get_conn()) as conn:
        row = conn.execute(
            """
            SELECT messages.* FROM messages
        LEFT JOIN chats ON chats.id = messages.chat_id
        LEFT JOIN projects ON projects.id = COALESCE(chats.project_id, messages.project_id)
        WHERE messages.id = ? AND COALESCE(chats.user_id, projects.user_id) = ?
        """,
        (message_id, user_id),
    ).fetchone()
        
        if not row:
            return None
        if row["role"] != "user":
            return None
        
        content = (content or "").strip()
        if not content:
            return None
        
        conn.execute("UPDATE messages SET content = ? WHERE id = ?", (content, message_id))
        
        if truncate_after:
            if row["chat_id"] is not None:
                conn.execute(
                    "DELETE FROM messages WHERE chat_id = ? AND id > ?",
                    (row["chat_id"], message_id),
                )
            else:
                conn.execute(
                    "DELETE FROM messages WHERE project_id = ? AND id > ?",
                    (row["project_id"], message_id),
                )
        
        conn.commit()
        return dict(row)


def delete_message(message_id: int, user_id: int) -> bool:
    with closing(get_conn()) as conn:
        row = conn.execute(
            """
            SELECT messages.id FROM messages
            LEFT JOIN chats ON chats.id = messages.chat_id
            LEFT JOIN projects ON projects.id = COALESCE(chats.project_id, messages.project_id)
            WHERE messages.id = ? AND COALESCE(chats.user_id, projects.user_id) = ?
            """,
            (message_id, user_id),
        ).fetchone()
        
        if not row:
            return False
        
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        conn.commit()
        return True


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def add_file(project_id, user_id, filename, stored_path, mime, kind, text_extracted) -> int:
    with closing(get_conn()) as conn:
        cur = conn.execute(
            """
            INSERT INTO files(project_id, user_id, filename, stored_path, mime,
                              kind, text_extracted, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                user_id,
                filename,
                stored_path,
                mime or "",
                kind or "file",
                text_extracted or "",
                time.time(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_file_text(file_id: int, text: str) -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            "UPDATE files SET text_extracted = ? WHERE id = ?",
            (text[:200_000], file_id),
        )
        conn.commit()


def list_files(project_id: int):
    with closing(get_conn()) as conn:
        return conn.execute(
            "SELECT * FROM files WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()


def get_file(file_id: int):
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return dict(row) if row else None


def delete_file(file_id: int) -> None:
    with closing(get_conn()) as conn:
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()


def project_image_files(project_id: int, limit: int = 4):
    with closing(get_conn()) as conn:
        return conn.execute(
            """
            SELECT * FROM files
            WHERE project_id = ? AND kind = 'image'
            ORDER BY id DESC LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------
def upsert_model(model_id: str, **fields) -> None:
    fields.setdefault("added_at", time.time())
    keys = ["id"] + list(fields.keys())
    values = [model_id] + list(fields.values())
    updates = ", ".join(f"{k} = excluded.{k}" for k in fields.keys())
    
    with closing(get_conn()) as conn:
        conn.execute(
            f"INSERT INTO models({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            values,
        )
        conn.commit()


def set_model_status(model_id: str, status: str, progress: float = None, error: str = None,
                     size_bytes: int = None) -> None:
    parts, values = ["status = ?"], [status]
    
    if progress is not None:
        parts.append("progress = ?")
        values.append(progress)
    if error is not None:
        parts.append("error = ?")
        values.append(error)
    if size_bytes is not None:
        parts.append("size_bytes = ?")
        values.append(size_bytes)
    
    values.append(model_id)
    
    with closing(get_conn()) as conn:
        conn.execute(f"UPDATE models SET {', '.join(parts)} WHERE id = ?", values)
        conn.commit()


def list_model_registry():
    with closing(get_conn()) as conn:
        return conn.execute(
            "SELECT * FROM models ORDER BY last_used DESC, added_at DESC"
        ).fetchall()


def delete_model_row(model_id: str) -> None:
    with closing(get_conn()) as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()


def touch_model(model_id: str) -> None:
    with closing(get_conn()) as conn:
        conn.execute("UPDATE models SET last_used = ? WHERE id = ?", (time.time(), model_id))
        conn.commit()


# --------------------------------------------------------------------------
# generated images
# --------------------------------------------------------------------------
def add_generated_image(user_id, project_id, prompt, negative, filename,
                        width, height, steps, seed, model) -> int:
    with closing(get_conn()) as conn:
        cur = conn.execute(
            """
            INSERT INTO generated_images(user_id, project_id, prompt, negative, filename,
                                         width, height, steps, seed, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                project_id,
                prompt or "",
                negative or "",
                filename,
                width or 0,
                height or 0,
                steps or 0,
                seed or 0,
                model or "",
                time.time(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_generated_images(user_id: int, limit: int = 100):
    with closing(get_conn()) as conn:
        return conn.execute(
            "SELECT * FROM generated_images WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()


def delete_generated_image(user_id: int, image_id: int):
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM generated_images WHERE id = ? AND user_id = ?",
            (image_id, user_id),
        ).fetchone()
        
        if not row:
            return None
        
        conn.execute("DELETE FROM generated_images WHERE id = ?", (image_id,))
        conn.commit()
        
        path = GENERATED_DIR / row["filename"]
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        
        return dict(row)