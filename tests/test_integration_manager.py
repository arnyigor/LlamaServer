# tests/test_integration_manager.py
"""Тесты IntegrationManager."""

import json
import tempfile
import unittest
from pathlib import Path

from src.services.integration_manager import IntegrationManager


class TestIntegrationManager(unittest.TestCase):

    def setUp(self):
        self.mgr = IntegrationManager("http://127.0.0.1:8080/v1")

    def _make_config(self, tmpdir: Path, data: dict) -> Path:
        p = tmpdir / "config.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_check_empty_path(self):
        r = self.mgr.check_models("", "opencode")
        self.assertFalse(r.success)
        self.assertIn("не указан", r.message)

    def test_check_nonexistent_file(self):
        r = self.mgr.check_models("/nonexistent/config.json", "opencode")
        self.assertFalse(r.success)

    def test_add_and_check_opencode(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._make_config(Path(d), {})
            r = self.mgr.add_model(str(cfg), "opencode", "my-model")
            self.assertTrue(r.success)
            self.assertIn("my-model", r.model_ids)

            r2 = self.mgr.check_models(str(cfg), "opencode")
            self.assertTrue(r2.success)
            self.assertIn("my-model", r2.model_ids)

    def test_add_and_remove_opencode(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._make_config(Path(d), {})
            self.mgr.add_model(str(cfg), "opencode", "my-model")
            r = self.mgr.remove_model(str(cfg), "opencode", "my-model")
            self.assertTrue(r.success)
            self.assertNotIn("my-model", r.model_ids)

    def test_add_and_check_pi(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._make_config(Path(d), {})
            r = self.mgr.add_model(str(cfg), "pi", "pi-model")
            self.assertTrue(r.success)
            self.assertIn("pi-model", r.model_ids)

    def test_add_duplicate_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._make_config(Path(d), {})
            self.mgr.add_model(str(cfg), "opencode", "model")
            r = self.mgr.add_model(str(cfg), "opencode", "model")
            self.assertTrue(r.success)
            self.assertEqual(r.model_ids.count("model"), 1)

    def test_remove_nonexistent_model(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._make_config(Path(d), {})
            r = self.mgr.remove_model(str(cfg), "opencode", "ghost")
            self.assertFalse(r.success)

    def test_invalid_target(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._make_config(Path(d), {})
            r = self.mgr.add_model(str(cfg), "unknown_target", "model")
            self.assertFalse(r.success)

    def test_add_check_and_remove_claude_code(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._make_config(Path(d), {"env": {"KEEP_ME": "yes"}})
            r = self.mgr.add_model(str(cfg), "claude", "local-model")
            self.assertTrue(r.success)
            self.assertEqual(r.model_ids, ["local-model"])

            data = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertEqual(data["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8080")
            self.assertEqual(data["env"]["ANTHROPIC_AUTH_TOKEN"], "llamacpp")
            self.assertEqual(data["env"]["ANTHROPIC_MODEL"], "local-model")
            self.assertEqual(data["env"]["ANTHROPIC_SMALL_FAST_MODEL"], "local-model")
            self.assertEqual(data["env"]["KEEP_ME"], "yes")

            checked = self.mgr.check_models(str(cfg), "claude")
            self.assertTrue(checked.success)
            self.assertEqual(checked.model_ids, ["local-model"])

            removed = self.mgr.remove_model(str(cfg), "claude", "local-model")
            self.assertTrue(removed.success)
            self.assertEqual(removed.model_ids, [])
            data = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertEqual(data["env"]["KEEP_ME"], "yes")
            self.assertNotIn("ANTHROPIC_BASE_URL", data["env"])
            self.assertNotIn("ANTHROPIC_AUTH_TOKEN", data["env"])

    def test_claude_code_settings_file_can_be_created(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / ".claude" / "settings.json"
            result = self.mgr.add_model(str(cfg), "claude", "local-model")

            self.assertTrue(result.success)
            self.assertTrue(cfg.exists())
            data = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertEqual(data["env"]["ANTHROPIC_MODEL"], "local-model")


# tests/test_log_manager.py
"""Тесты LogManager."""

import unittest
from unittest.mock import MagicMock, patch

from src.ui.log_manager import LogManager, LogEntry, _AUTO_LEVEL_PATTERNS


class TestLogManagerUnit(unittest.TestCase):

    def test_auto_level_bench(self):
        """tok/s определяется как bench."""
        text = "eval time: 42.3 tok/s"
        entry_level = "info"
        for pattern, level in _AUTO_LEVEL_PATTERNS:
            if pattern.search(text):
                entry_level = level
                break
        self.assertEqual(entry_level, "bench")

    def test_auto_level_error(self):
        text = "FATAL ERROR: out of memory"
        entry_level = "info"
        for pattern, level in _AUTO_LEVEL_PATTERNS:
            if pattern.search(text):
                entry_level = level
                break
        self.assertEqual(entry_level, "error")

    def test_speed_extraction(self):
        """Извлечение скорости из текста."""
        mock_edit = MagicMock()
        mock_edit.textCursor.return_value = MagicMock()
        mock_edit.document.return_value = MagicMock()
        mock_edit.document.return_value.blockCount.return_value = 0

        # Минимальный mock для QObject
        with patch("src.ui.log_manager.QTimer"):
            mgr = LogManager.__new__(LogManager)
            mgr._pp_speed = None
            mgr._tg_speed = None
            mgr.speed_updated = MagicMock()
            mgr.speed_updated.emit = MagicMock()

            mgr._extract_speed(
                "prompt eval time = 512.5 tokens/s, eval time = 23.1 tok/s"
            )
            self.assertIsNotNone(mgr._tg_speed)


if __name__ == "__main__":
    unittest.main()
