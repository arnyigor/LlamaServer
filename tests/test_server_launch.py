"""Тесты ServerLaunchController: отложенный рестарт и env запуска."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.server_launch import ServerLaunchController


class _Settings:
    cuda_visible_devices = "0,1"
    cuda_module_loading = "LAZY"


class _EmptySettings:
    cuda_visible_devices = ""
    cuda_module_loading = ""


class TestServerLaunchController(unittest.TestCase):
    def test_request_and_poll_waits_for_server_stop(self):
        launcher = ServerLaunchController()
        launch = ("server.exe", ["-m", "m.gguf"], {"K": "V"})
        launcher.request_restart(launch)

        # Сервер ещё работает — ждём, запуск не отдаётся
        had, ready = launcher.poll_pending(server_running=True)
        self.assertTrue(had)
        self.assertIsNone(ready)
        self.assertTrue(launcher.is_pending)

        # Сервер остановился — запуск отдаётся ровно один раз
        had, ready = launcher.poll_pending(server_running=False)
        self.assertTrue(had)
        self.assertEqual(ready, launch)
        self.assertFalse(launcher.is_pending)

        had, ready = launcher.poll_pending(server_running=False)
        self.assertFalse(had)
        self.assertIsNone(ready)

    def test_cancel_pending(self):
        launcher = ServerLaunchController()
        launcher.request_restart(("e", [], None))
        self.assertTrue(launcher.cancel_pending())
        self.assertFalse(launcher.is_pending)
        # Повторная отмена без запроса — False
        self.assertFalse(launcher.cancel_pending())

        had, ready = launcher.poll_pending(server_running=False)
        self.assertFalse(had)
        self.assertIsNone(ready)

    def test_poll_without_request(self):
        launcher = ServerLaunchController()
        had, ready = launcher.poll_pending(server_running=False)
        self.assertFalse(had)
        self.assertIsNone(ready)

    def test_restart_needed_flag(self):
        launcher = ServerLaunchController()
        self.assertFalse(launcher.restart_needed)
        launcher.mark_restart_needed()
        self.assertTrue(launcher.restart_needed)
        launcher.clear_restart_needed()
        self.assertFalse(launcher.restart_needed)

    def test_env_from_settings(self):
        env = ServerLaunchController.env_from_settings(_Settings())
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "0,1", "CUDA_MODULE_LOADING": "LAZY"})
        self.assertEqual(ServerLaunchController.env_from_settings(_EmptySettings()), {})

    def test_pending_changed_signal(self):
        launcher = ServerLaunchController()
        events = []
        launcher.pending_changed.connect(lambda pending: events.append(pending))
        launcher.request_restart(("e", [], None))
        launcher.poll_pending(server_running=False)
        launcher.request_restart(("e", [], None))
        launcher.cancel_pending()
        self.assertEqual(events, [True, False, True, False])


if __name__ == "__main__":
    unittest.main()
