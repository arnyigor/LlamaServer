"""Характеризационные тесты MTP-fallback (Этап 2.3 плана).

Зафиксировано поведение LlamaGUI ДО выноса MTPFallbackController:
детекция ошибки draft-MTP, вырезание MTP-флагов и решение о повторном
запуске. После выноса эти тесты обязаны проходить без изменений —
GUI-методы остаются тонкими делегатами контроллеру.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import LlamaGUI
from src.core.config import ConfigManager
from src.core.mtp_fallback import MtpFallbackController


def _make_gui() -> LlamaGUI:
    gui = LlamaGUI.__new__(LlamaGUI)
    tmp = tempfile.TemporaryDirectory()
    gui._tmp = tmp  # держим ссылку, чтобы директория жила до конца теста
    gui.config = ConfigManager(str(Path(tmp.name) / "s.json"), str(Path(tmp.name) / "p.json"))
    gui.ui = MagicMock()
    gui.log_mgr = MagicMock()
    gui.mtp = MtpFallbackController()
    return gui


def _mtp_launch_args():
    return [
        "-m", "model.gguf",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "8",
        "--spec-draft-p-min", "0.8",
        "--model-draft", "draft.gguf",
    ]


class TestStripMtpArgs(unittest.TestCase):
    def test_removes_all_mtp_value_flags(self):
        gui = _make_gui()
        stripped = gui._strip_mtp_args(_mtp_launch_args())
        self.assertEqual(stripped, ["-m", "model.gguf"])

    def test_keeps_non_mtp_flags(self):
        gui = _make_gui()
        args = ["-m", "m.gguf", "-c", "8192", "--spec-draft-n-max", "8", "--jinja"]
        self.assertEqual(gui._strip_mtp_args(args), ["-m", "m.gguf", "-c", "8192", "--jinja"])

    def test_inline_equals_form(self):
        gui = _make_gui()
        args = ["-m", "m.gguf", "--spec-draft-n-max=8", "--spec-type=draft-mtp"]
        self.assertEqual(gui._strip_mtp_args(args), ["-m", "m.gguf"])

    def test_no_mtp_flags_returns_same_list(self):
        gui = _make_gui()
        args = ["-m", "m.gguf", "-c", "4096"]
        self.assertEqual(gui._strip_mtp_args(args), args)


class TestMtpModelRules(unittest.TestCase):
    def test_uses_embedded_mtp_mode_requires_arch_and_capable(self):
        gui = _make_gui()
        self.assertTrue(
            gui._uses_embedded_mtp_mode(
                {"architecture": "gemma4", "mtp_capable": True}
            )
        )
        self.assertTrue(
            gui._uses_embedded_mtp_mode({"architecture": "qwen3", "mtp_capable": True})
        )
        # Не mtp_capable — не встроенный режим
        self.assertFalse(
            gui._uses_embedded_mtp_mode({"architecture": "gemma4", "mtp_capable": False})
        )
        # QAT-сборки идут через отдельный draft
        self.assertFalse(
            gui._uses_embedded_mtp_mode(
                {"architecture": "gemma4", "mtp_capable": True, "is_qat": True}
            )
        )
        self.assertFalse(
            gui._uses_embedded_mtp_mode(
                {"architecture": "llama", "mtp_capable": True}
            )
        )

    def test_auto_mtp_supported_embedded(self):
        gui = _make_gui()
        self.assertTrue(
            gui._auto_mtp_supported({"architecture": "gemma4", "mtp_capable": True})
        )

    def test_auto_mtp_supported_requires_existing_draft(self):
        gui = _make_gui()
        self.assertFalse(gui._auto_mtp_supported({"mtp_draft_path": "no-such.gguf"}))

    def test_auto_mtp_supported_respects_disabled_models(self):
        gui = _make_gui()
        with patch("main.os.path.isfile", return_value=True):
            info = {"path": "C:/models/m.gguf", "mtp_draft_path": "C:/models/d.gguf"}
            self.assertTrue(gui._auto_mtp_supported(info))
            gui.config.settings.spec_draft_auto_disabled_models = [
                "c:\\models\\m.gguf"
            ]
            self.assertFalse(gui._auto_mtp_supported(info))

    def test_auto_mtp_draft_path_prefers_manual(self):
        gui = _make_gui()
        gui.config.settings.spec_draft_manual_paths = {
            "C:/MODELS/m.gguf": "C:/manual-draft.gguf"
        }
        with patch("main.os.path.isfile", return_value=True):
            info = {"path": "c:/models/m.gguf", "mtp_draft_path": "c:/models/d.gguf"}
            self.assertEqual(gui._auto_mtp_draft_path(info), "C:/manual-draft.gguf")

    def test_set_manual_draft_path_updates_both_maps(self):
        gui = _make_gui()
        info = {"path": "C:/models/m.gguf"}
        gui._set_mtp_manual_draft_path("C:/draft.gguf", info)
        settings = gui.config.settings
        self.assertEqual(
            settings.spec_draft_manual_paths["c:\\models\\m.gguf"], "C:/draft.gguf"
        )
        # Ручной путь задан — авто-подбор для модели не помечен отключённым
        # (ручной путь и так приоритетнее); очистка поля отключает авто-подбор.
        self.assertNotIn(
            "c:\\models\\m.gguf", settings.spec_draft_auto_disabled_models
        )
        gui._set_mtp_manual_draft_path("", info)
        self.assertIn("c:\\models\\m.gguf", settings.spec_draft_auto_disabled_models)
        self.assertEqual(settings.spec_draft_manual_paths, {})


class TestMtpFallbackFlow(unittest.TestCase):
    def test_mark_failed_fatal_schedules_abort_once(self):
        gui = _make_gui()
        with patch("main.QTimer.singleShot") as shot:
            gui._mark_mtp_launch_failed("reason", fatal=True)
            gui._mark_mtp_launch_failed("another", fatal=True)
        self.assertEqual(shot.call_count, 1)
        self.assertTrue(gui._mtp_draft_error_seen)
        self.assertEqual(gui._mtp_failure_reason, "another")

    def test_mark_failed_non_fatal_no_abort(self):
        gui = _make_gui()
        with patch("main.QTimer.singleShot") as shot:
            gui._mark_mtp_launch_failed("reason", fatal=False)
        shot.assert_not_called()
        self.assertTrue(gui._mtp_draft_error_seen)

    def test_retry_after_fatal_error(self):
        gui = _make_gui()
        gui.mtp.remember_launch("server.exe", _mtp_launch_args(), {"A": "1"}, is_retry=False)
        gui._mark_mtp_launch_failed("draft GGUF failed to load", fatal=False)
        launched = []
        with patch("main.QTimer.singleShot", side_effect=lambda ms, fn: launched.append(fn)) as shot:
            result = gui._retry_without_mtp_if_needed(1)
        self.assertTrue(result)
        self.assertTrue(gui._mtp_fallback_attempted)
        # Повторная попытка не выполняется
        self.assertFalse(gui._retry_without_mtp_if_needed(1))
        # UI: чекбокс MTP выключен
        gui.ui.speculative_mtp.setChecked.assert_called_once_with(False)
        # Отложенный запуск с очищенными аргументами
        self.assertEqual(len(launched), 1)

    def test_no_retry_without_mtp_error(self):
        gui = _make_gui()
        gui.mtp.remember_launch("server.exe", _mtp_launch_args(), None, is_retry=False)
        self.assertFalse(gui._retry_without_mtp_if_needed(1))

    def test_no_retry_without_mtp_flags(self):
        gui = _make_gui()
        gui.mtp.remember_launch("server.exe", ["-m", "m.gguf"], None, is_retry=False)
        gui._mark_mtp_launch_failed("x")
        self.assertFalse(gui._retry_without_mtp_if_needed(1))

    def test_no_retry_on_clean_exit(self):
        gui = _make_gui()
        gui.mtp.remember_launch("server.exe", _mtp_launch_args(), None, is_retry=False)
        gui._mark_mtp_launch_failed("x")
        self.assertFalse(gui._retry_without_mtp_if_needed(0))

    def test_no_retry_without_last_launch(self):
        gui = _make_gui()
        gui._mark_mtp_launch_failed("x")
        self.assertFalse(gui._retry_without_mtp_if_needed(1))


if __name__ == "__main__":
    unittest.main()
