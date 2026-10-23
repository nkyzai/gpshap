import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import auth, main


class PasswordRouteTests(unittest.TestCase):
    def setUp(self):
        main.app.dependency_overrides[auth.get_current_user] = lambda: {
            "id": 7,
            "password_hash": "stored-hash",
        }

    def tearDown(self):
        main.app.dependency_overrides.pop(auth.get_current_user, None)

    @patch("app.auth.db.delete_other_sessions")
    @patch("app.auth.db.update_user_password")
    @patch("app.auth.db.verify_password", return_value=True)
    def test_password_endpoint_is_mounted_and_updates_hash(
        self, verify_password, update_password, delete_other_sessions
    ):
        client = TestClient(main.app)
        client.cookies.set(auth.COOKIE_NAME, "current-session")
        response = client.post(
            "/api/password",
            json={"current_password": "old-password", "new_password": "new-password"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"ok": True})
        verify_password.assert_called_once_with("old-password", "stored-hash")
        update_password.assert_called_once_with(7, "new-password")
        delete_other_sessions.assert_called_once_with(7, "current-session")


if __name__ == "__main__":
    unittest.main()
