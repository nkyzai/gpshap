import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app import db, main, tools


class LocalPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"

    def tearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_migration_removes_old_transcript_export_data(self):
        with closing(db.get_conn()) as conn:
            conn.executescript("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE projects (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    name TEXT NOT NULL
                );
                CREATE TABLE chat_exports (user_message TEXT);
                CREATE TABLE chat_export_consents (user_id INTEGER);
                INSERT INTO chat_exports VALUES ('private transcript');
            """)
            conn.commit()

        db.init_db()

        with closing(db.get_conn()) as conn:
            tables = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        self.assertNotIn("chat_exports", tables)
        self.assertNotIn("chat_export_consents", tables)

    def test_network_chat_tools_are_not_exposed_or_executable(self):
        exposed_names = {tool["name"] for tool in tools.MCP_TOOLS}
        self.assertNotIn("web_search", exposed_names)
        self.assertNotIn("fetch_url", exposed_names)
        routes = {getattr(route, "path", "") for route in main.app.routes}
        self.assertFalse(any("chat-export" in route for route in routes))
        self.assertEqual(
            tools.execute_tool("web_search", {"query": "private chat"}),
            {"error": "Unknown tool: web_search"},
        )
        self.assertEqual(
            tools.execute_tool("fetch_url", {"url": "https://example.com"}),
            {"error": "Unknown tool: fetch_url"},
        )


if __name__ == "__main__":
    unittest.main()
