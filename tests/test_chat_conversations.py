import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import db, main
from app.schemas import ChatRequest


class ChatConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "gpshap.db"
        self.db_path.touch()
        self.db_path.unlink()
        self.db_path_patch = patch.object(db, "DB_PATH", self.db_path)
        self.db_path_patch.start()

    def tearDown(self):
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    def test_legacy_project_messages_migrate_into_a_chat(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL, created_at REAL NOT NULL
            );
            CREATE TABLE projects (
                id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, name TEXT NOT NULL,
                system_prompt TEXT DEFAULT '', sort_order INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0, archived INTEGER DEFAULT 0, created_at REAL NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, meta TEXT DEFAULT '{}', number INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
            INSERT INTO users VALUES (1, 'alice', 'hash', 100);
            INSERT INTO projects VALUES (7, 1, 'Legacy project', '', 0, 0, 0, 100);
            INSERT INTO messages VALUES (11, 7, 'user', 'Preserve me', '{}', 0, 101);
            """
        )
        conn.commit()
        conn.close()

        db.init_db()

        chats = db.list_chats(1)
        self.assertEqual(len(chats), 1)
        self.assertEqual(chats[0]["project_id"], 7)
        messages = db.list_chat_messages(chats[0]["id"])
        self.assertEqual([(row["id"], row["content"]) for row in messages], [(11, "Preserve me")])
        with closing(db.get_conn()) as migrated:
            project_column = next(
                row for row in migrated.execute("PRAGMA table_info(messages)")
                if row["name"] == "project_id"
            )
            self.assertEqual(project_column["notnull"], 0)
            self.assertEqual(migrated.execute("PRAGMA foreign_key_check").fetchall(), [])

        db.init_db()
        self.assertEqual(len(db.list_chats(1)), 1)

    def test_general_and_project_chats_are_independent_and_user_owned(self):
        db.init_db()
        owner_id = db.create_user("alice", "password1")
        other_user_id = db.create_user("bob", "password2")
        project_id = db.create_project(owner_id, "Work")
        general_chat = db.create_chat(owner_id)
        first_project_chat = db.create_chat(owner_id, project_id)
        second_project_chat = db.create_chat(owner_id, project_id)

        db.add_chat_message(general_chat, None, "user", "General question")
        db.add_chat_message(first_project_chat, project_id, "user", "First discussion")
        db.add_chat_message(second_project_chat, project_id, "user", "Second discussion")

        chats = db.list_chats(owner_id)
        self.assertEqual(len(chats), 3)
        self.assertEqual(
            {chat["id"] for chat in chats if chat["project_id"] == project_id},
            {first_project_chat, second_project_chat},
        )
        self.assertEqual(
            db.list_chat_messages(general_chat)[0]["content"], "General question"
        )
        self.assertEqual(db.get_chat(other_user_id, general_chat), None)
        self.assertFalse(db.delete_chat(other_user_id, general_chat))

        self.assertTrue(db.update_message_content(
            db.list_chat_messages(first_project_chat)[0]["id"],
            owner_id,
            "Edited discussion",
        ))
        self.assertEqual(
            db.list_chat_messages(second_project_chat)[0]["content"], "Second discussion"
        )

        duplicated_project_id = db.duplicate_project(owner_id, project_id)
        duplicated_chats = [
            chat for chat in db.list_chats(owner_id)
            if chat["project_id"] == duplicated_project_id
        ]
        self.assertEqual(len(duplicated_chats), 2)
        copied_messages = {
            db.list_chat_messages(chat["id"])[0]["content"] for chat in duplicated_chats
        }
        self.assertEqual(copied_messages, {"Edited discussion", "Second discussion"})

    def test_chat_api_accepts_general_chat_and_enforces_ownership(self):
        db.init_db()
        owner_id = db.create_user("alice", "password1")
        other_user_id = db.create_user("bob", "password2")
        owner = {"id": owner_id}
        other_user = {"id": other_user_id}

        chat = main.create_chat({}, owner)
        self.assertIsNone(chat["project_id"])
        self.assertEqual(main.get_chat_messages(chat["id"], owner), {"messages": []})
        with self.assertRaises(HTTPException) as error:
            main.get_chat_messages(chat["id"], other_user)
        self.assertEqual(error.exception.status_code, 404)

        main._chat_context(chat["id"], ChatRequest(message="Hello"), owner)
        self.assertEqual(
            db.list_chat_messages(chat["id"])[0]["content"], "Hello"
        )


if __name__ == "__main__":
    unittest.main()
