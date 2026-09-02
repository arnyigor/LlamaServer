"""Тесты UI-логики главного окна (MainWindowUI)."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindowUI

_APP = QApplication.instance() or QApplication([])


class TestLogMaximize(unittest.TestCase):
    """Регрессия: кнопка Maximize дока логов должна раскрывать логи на весь
    экран, скрывая контент и панель запуска (раньше логи занимали лишь часть)."""

    def setUp(self):
        self.w = MainWindowUI()
        self.w.resize(900, 700)

    def test_maximize_hides_content_and_control(self):
        self.w._apply_log_maximize(True)
        self.assertTrue(self.w._log_maximized)
        # Контент намеренно НЕ скрывается (commit 9de2be5): прятать весь
        # nav-рейл и страницу выглядело как баг. Вместо этого контент
        # сжимается до минимума, а лог-док получает основную часть высоты.
        sizes = self.w.main_vsplit.sizes()
        self.assertGreater(sizes[0], 0)
        self.assertGreater(sizes[2], sizes[0])

    def test_restore_shows_content_and_control(self):
        self.w._apply_log_maximize(True)
        self.w._apply_log_maximize(False)
        self.assertFalse(self.w._log_maximized)
        self.assertFalse(self.w.content_splitter.isHidden())
        self.assertFalse(self.w.control_strip.isHidden())
        sizes = self.w.main_vsplit.sizes()
        self.assertGreater(sizes[0], 0)
        self.assertGreater(sizes[2], 0)

    def test_nav_to_autotune_syncs_models_without_error(self):
        # AutoTune — последний пункт NAV_PAGES (индекс 8). Переход не должен
        # падать с 'MainWindowUI' object has no attribute 'ui'.
        idx = next(
            i for i, (_, key) in enumerate(self.w.NAV_PAGES) if key == "autotune"
        )
        self.w._on_nav_selected(idx)
        self.assertTrue(hasattr(self.w, "autotune"))

    def _populate_launch_combo(self, n: int) -> None:
        self.w.model_combo.clear()
        for i in range(n):
            self.w.model_combo.addItem(f"model{i}", f"path{i}")

    def test_autotune_picker_syncs_all_models(self):
        self._populate_launch_combo(6)
        self.w.autotune._sync_model_items()
        self.assertEqual(self.w.autotune.model_combo.count(), 6)

    def test_autotune_picker_syncs_on_launch_combo_change(self):
        # Реальный путь сканирования: on_models_found -> set_model_list
        # перезаполняет launch-combo и синхронизирует пикер AutoTune целиком
        # (раньше currentIndexChanged стрелял при count==1, оставляя 1 модель).
        models = [{"display": f"model{i}", "path": f"path{i}"} for i in range(6)]
        self.w.set_model_list(models)
        self.assertEqual(self.w.autotune.model_combo.count(), 6)

    def test_autotune_kv_default_is_q8(self):
        # Q8 KV — практичный дефолт (почти F16 качество, вдвое меньше VRAM
        # на KV-кэш); большинству поисков не нужен потолок F16.
        self.assertEqual(self.w.autotune.kv_combo.currentText(), "q8_0/q8_0")
        self.assertEqual(self.w.autotune.options()["kv_k"], "q8_0")
        self.assertEqual(self.w.autotune.options()["kv_v"], "q8_0")

    def test_autotune_ctx_quick_buttons_set_context(self):
        buttons = self.w.autotune.ctx_quick_buttons
        self.assertEqual(len(buttons), 6)
        buttons[3].click()  # "64K"
        self.assertEqual(self.w.autotune.ctx_spin.value(), 65536)

    def test_autotune_search_strategy_is_fixed_not_user_facing(self):
        # Upstream llama_autotuner hard rule (SPECIFICATION.md "GUI one-click
        # workflow"): letting the user pick Quick/Balanced/Thorough before
        # search start routinely picked the wrong branch under VRAM pressure,
        # so mode/priority/budget must never be exposed as UI controls again.
        for removed in (
            "goal_combo",
            "mode_combo",
            "priority_combo",
            "degradation_combo",
            "min_pp_spin",
            "vram_floor_spin",
            "max_time_spin",
            "max_runs_spin",
        ):
            self.assertFalse(
                hasattr(self.w.autotune, removed),
                f"AutoTuneWidget must not expose {removed} (matches upstream's hard rule)",
            )
        options = self.w.autotune.options()
        self.assertEqual(options["mode"], "quick")
        self.assertEqual(options["priority"], "balanced")
        self.assertEqual(options["max_minutes"], 8)
        self.assertEqual(options["max_runs"], 12)
        self.assertEqual(options["absolute_vram_floor_mb"], 300)
        self.assertIsNone(options["min_pp_tps"])

    def test_autotune_strict_checkbox_maps_to_degradation_policy(self):
        self.assertFalse(self.w.autotune.strict_chk.isChecked())
        self.assertEqual(self.w.autotune.options()["degradation_policy"], "auto")
        self.w.autotune.strict_chk.setChecked(True)
        self.assertEqual(self.w.autotune.options()["degradation_policy"], "strict")

    def test_autotune_vision_checkbox_present(self):
        self.assertFalse(self.w.autotune.vision_chk.isChecked())
        self.w.autotune.vision_chk.setChecked(True)
        self.assertEqual(self.w.autotune.options()["vision"], "required")

    def test_autotune_vision_unchecked_sends_off_not_auto(self):
        # Regression: session.py silently promotes "auto" to REQUIRED
        # whenever an mmproj is auto-detected next to the model, regardless
        # of this checkbox. An unchecked box must mean "off", full stop —
        # matching upstream llama_autotuner's own GUI, which never sends
        # "auto" (that's a CLI-only power option).
        self.assertFalse(self.w.autotune.vision_chk.isChecked())
        self.w.autotune.mmproj_edit.setText("G:/models/mmproj-BF16.gguf")
        self.assertEqual(self.w.autotune.options()["vision"], "off")


if __name__ == "__main__":
    unittest.main()
