import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app import agent, main


class ChatStopTests(unittest.TestCase):
    def tearDown(self):
        agent.CANCELLED_PROJECTS.pop(42, None)

    @patch("app.main.db.get_project", return_value=None)
    def test_stop_rejects_projects_not_owned_by_user(self, get_project):
        with self.assertRaises(HTTPException) as error:
            main.stop_chat(42, {"id": 7})

        self.assertEqual(error.exception.status_code, 404)
        self.assertNotIn(42, agent.CANCELLED_PROJECTS)
        get_project.assert_called_once_with(7, 42)

    @patch("app.main.db.get_project", return_value={"id": 42})
    def test_stop_sets_cancellation_for_owned_project(self, get_project):
        result = main.stop_chat(42, {"id": 7})

        self.assertEqual(result, {"ok": True, "stopped": True})
        self.assertTrue(agent.CANCELLED_PROJECTS[42])
        get_project.assert_called_once_with(7, 42)


if __name__ == "__main__":
    unittest.main()
