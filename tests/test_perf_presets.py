"""Тесты сохранения/загрузки performance presets и быстрых кнопок Context Size."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from src.core.benchmark_models import AutoTunePlan, BenchmarkCandidate, BenchmarkResult
from src.core.config import ConfigManager
from main import LlamaGUI


_APP = QApplication.instance() or QApplication([])


class TestPerfPresetsUI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.settings_path = str(root / "settings.json")
        self.profiles_path = str(root / "profiles.json")
        self.model_path = str(root / "model.gguf")
        Path(self.model_path).write_bytes(b"test")

        self.config = ConfigManager(self.settings_path, self.profiles_path)
        patches = [
            patch("main.ConfigManager", return_value=self.config),
            patch("main.QSystemTrayIcon.isSystemTrayAvailable", return_value=False),
            patch("main.QTimer.singleShot", return_value=None),
        ]
        self._patches = patches
        for p in patches:
            p.start()

        self.gui = LlamaGUI()
        self.ui = self.gui.ui
        self.ui.auto_params.setChecked(False)
        self.ui.models_by_path[self.model_path] = {"recommended_ctx": 8192}
        self.ui.model_combo.clear()
        self.ui.model_combo.addItem("model.gguf", self.model_path)
        self.ui.model_combo.setCurrentIndex(0)

    def tearDown(self):
        self.ui.close()
        for p in reversed(self._patches):
            p.stop()
        self.tmp.cleanup()

    def _set_saved_values(self):
        u = self.ui
        u.ctx_size.setValue(16384)
        u.gpu_auto.setChecked(False)
        u.gpu_layers.setValue(77)
        u.cpu_moe_layers.setValue(5)
        u.threads.setValue(7)
        u.threads_batch.setValue(3)
        u.cache_type_k.setCurrentText("q8_0")
        u.cache_type_v.setCurrentText("q4_0")
        u.batch_size.setValue(1024)
        u.ubatch_size.setValue(512)
        u.parallel_slots.setValue(2)
        u.flash_attn.setChecked(False)
        u.fit_off.setChecked(False)
        u.reasoning_mode.setCurrentText("on")
        u.ctx_checkpoints.setValue(4)
        u.cache_ram.setValue(2048)
        u.temperature.setValue(0.7)
        u.repeat_penalty.setValue(1.2)
        u.use_mmap.setChecked(False)
        u.use_mlock.setChecked(True)
        u.verbose.setChecked(True)
        u.log_timestamps.setChecked(True)
        u.cont_batching.setChecked(False)
        u.cache_prompt.setChecked(False)
        u.context_shift.setChecked(True)
        u.no_webui.setChecked(True)
        u.jinja.setChecked(True)
        u.use_mmproj.setChecked(False)
        u.mmproj_offload.setChecked(False)
        u.extra_args.setText("--seed 1")
        u.enable_thinking.setCurrentText("false")

    def _set_different_values(self):
        u = self.ui
        u.gpu_auto.setChecked(True)
        u.gpu_layers.setValue(1)
        u.cpu_moe_layers.setValue(-1)
        u.threads.setValue(1)
        u.threads_batch.setValue(0)
        u.cache_type_k.setCurrentText("f16")
        u.cache_type_v.setCurrentText("f16")
        u.batch_size.setValue(-1)
        u.ubatch_size.setValue(-1)
        u.parallel_slots.setValue(-1)
        u.flash_attn.setChecked(True)
        u.fit_off.setChecked(True)
        u.reasoning_mode.setCurrentText("off")
        u.ctx_checkpoints.setValue(-1)
        u.cache_ram.setValue(-2)
        u.temperature.setValue(-1.0)
        u.repeat_penalty.setValue(-1.0)
        u.use_mmap.setChecked(True)
        u.use_mlock.setChecked(False)
        u.verbose.setChecked(False)
        u.log_timestamps.setChecked(False)
        u.cont_batching.setChecked(True)
        u.cache_prompt.setChecked(True)
        u.context_shift.setChecked(False)
        u.no_webui.setChecked(False)
        u.jinja.setChecked(False)
        u.use_mmproj.setChecked(True)
        u.mmproj_offload.setChecked(True)
        u.extra_args.setText("")
        u.enable_thinking.setCurrentText("off")

    def _assert_saved_values_loaded(self):
        u = self.ui
        self.assertEqual(u.ctx_size.value(), 16384)
        self.assertFalse(u.gpu_auto.isChecked())
        self.assertEqual(u.gpu_layers.value(), 77)
        self.assertEqual(u.cpu_moe_layers.value(), 5)
        self.assertEqual(u.threads.value(), 7)
        self.assertEqual(u.threads_batch.value(), 3)
        self.assertEqual(u.cache_type_k.currentText(), "q8_0")
        self.assertEqual(u.cache_type_v.currentText(), "q4_0")
        self.assertEqual(u.batch_size.value(), 1024)
        self.assertEqual(u.ubatch_size.value(), 512)
        self.assertEqual(u.parallel_slots.value(), 2)
        self.assertFalse(u.flash_attn.isChecked())
        self.assertFalse(u.fit_off.isChecked())
        self.assertEqual(u.reasoning_mode.currentText(), "on")
        self.assertEqual(u.ctx_checkpoints.value(), 4)
        self.assertEqual(u.cache_ram.value(), 2048)
        self.assertAlmostEqual(u.temperature.value(), 0.7)
        self.assertAlmostEqual(u.repeat_penalty.value(), 1.2)
        self.assertFalse(u.use_mmap.isChecked())
        self.assertTrue(u.use_mlock.isChecked())
        self.assertTrue(u.verbose.isChecked())
        self.assertTrue(u.log_timestamps.isChecked())
        self.assertFalse(u.cont_batching.isChecked())
        self.assertFalse(u.cache_prompt.isChecked())
        self.assertTrue(u.context_shift.isChecked())
        self.assertTrue(u.no_webui.isChecked())
        self.assertTrue(u.jinja.isChecked())
        self.assertFalse(u.use_mmproj.isChecked())
        self.assertFalse(u.mmproj_offload.isChecked())
        self.assertEqual(u.extra_args.text(), "--seed 1")
        self.assertEqual(u.enable_thinking.currentText(), "false")

    def test_save_button_saves_and_context_input_loads_all_perf_params(self):
        self._set_saved_values()
        with patch.object(QMessageBox, "information", return_value=None):
            self.gui.save_preset()

        self._set_different_values()
        self.ui.ctx_size.setValue(8192)
        self.gui.on_ctx_changed(16384)

        self._assert_saved_values_loaded()

    def test_context_quick_button_loads_saved_preset(self):
        self._set_saved_values()
        with patch.object(QMessageBox, "information", return_value=None):
            self.gui.save_preset()

        self._set_different_values()
        self.ui.ctx_size.setValue(8192)
        button_16k = next(
            btn for btn in self.ui.ctx_quick_buttons if btn.property("ctx_value") == 16384
        )
        button_16k.click()

        self._assert_saved_values_loaded()

    def test_context_quick_button_reloads_when_value_is_already_selected(self):
        self._set_saved_values()
        with patch.object(QMessageBox, "information", return_value=None):
            self.gui.save_preset()

        self._set_different_values()
        self.ui.ctx_size.setValue(16384)
        button_16k = next(
            btn for btn in self.ui.ctx_quick_buttons if btn.property("ctx_value") == 16384
        )
        button_16k.click()

        self._assert_saved_values_loaded()

    def test_force_stop_is_available_even_without_owned_process(self):
        self.gui.update_action_buttons()

        self.assertTrue(self.ui.force_stop_btn.isEnabled())

    def test_left_autotune_button_opens_tab_and_builds_plan(self):
        self.ui.ctx_size.setValue(16384)
        self.assertTrue(hasattr(self.ui, "autotune_btn"))

        self.ui.autotune_btn.click()

        self.assertIs(self.ui.tabs.currentWidget(), self.ui.autotune)
        self.assertIsNotNone(self.gui.autotune_plan)
        self.assertEqual(self.gui.autotune_plan.ctx_size, 16384)
        self.assertGreater(self.ui.autotune.table.rowCount(), 0)

    def test_autotune_widget_shows_running_indicator(self):
        self.ui.ctx_size.setValue(16384)
        plan = self.gui.build_autotune_plan()
        self.assertIsNotNone(plan)

        self.ui.autotune.prepare_run(len(plan.candidates), 300)
        self.ui.autotune.set_running(True)
        self.ui.autotune.set_progress(1, len(plan.candidates))
        self.ui.autotune.mark_running(plan.candidates[0])

        self.assertFalse(self.ui.autotune.progress_bar.isHidden())
        self.assertEqual(self.ui.autotune.progress_bar.value(), 1)
        self.assertIn("AutoTune running", self.ui.autotune.status_label.text())
        self.assertIn("ETA", self.ui.autotune.progress_summary.text())
        self.assertIn(plan.candidates[0].id, self.ui.autotune.current_run_label.text())
        self.assertIn("START", self.ui.autotune.activity_log.toPlainText())
        self.assertEqual(self.ui.autotune.start_btn.text(), "AutoTune running...")

    def _make_autotune_candidate(self):
        return BenchmarkCandidate(
            "run_001",
            {
                "ngl": "auto",
                "ctx_size": 32768,
                "batch_size": 512,
                "ubatch_size": 512,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "threads": 14,
                "threads_batch": 0,
                "parallel_slots": 1,
                "flash_attn": True,
                "fit_off": True,
                "cache_prompt": False,
                "ncmoe": -1,
                "ctx_checkpoints": 0,
                "cache_ram": 0,
                "use_mmproj": False,
            },
            "safe baseline",
            "baseline",
        )

    def _set_autotune_best(self):
        candidate = self._make_autotune_candidate()
        self.gui.autotune_plan = AutoTunePlan(
            model_path=self.model_path,
            ctx_size=32768,
            mode="quick",
            target="balanced",
            engine="llama-bench",
            time_budget_sec=900,
            max_runs=1,
            repeat_top=1,
            candidates=[candidate],
        )
        self.gui.autotune_best_result = BenchmarkResult(
            candidate_id="run_001",
            status="success",
            score=100.0,
            prompt_tok_s=1000.0,
            generation_tok_s=100.0,
        )
        self.gui.autotune_results_dir = "benchmarks/test"

    def test_save_autotune_best_preset_roundtrip_loads_server_safe_params(self):
        self._set_autotune_best()
        with patch.object(QMessageBox, "information", return_value=None):
            self.gui.save_autotune_best_preset()

        self._set_different_values()
        self.ui.parallel_slots.setValue(2)
        self.ui.fit_off.setChecked(False)
        self.ui.cache_prompt.setChecked(True)
        loaded = self.config.load_perf_preset(self.model_path, 32768, self.ui)

        self.assertTrue(loaded)
        self.assertEqual(self.ui.ctx_size.value(), 32768)
        self.assertEqual(self.ui.parallel_slots.value(), 1)
        self.assertTrue(self.ui.fit_off.isChecked())
        self.assertFalse(self.ui.cache_prompt.isChecked())
        self.assertEqual(self.ui.cache_type_k.currentText(), "q8_0")
        self.assertEqual(self.ui.cache_type_v.currentText(), "q8_0")
        self.assertEqual(self.ui.ctx_checkpoints.value(), 0)
        self.assertEqual(self.ui.cache_ram.value(), 0)

    def test_manual_save_after_apply_best_roundtrip_loads_exact_autotune_params(self):
        self._set_autotune_best()
        with patch.object(QMessageBox, "information", return_value=None):
            applied = self.gui.apply_autotune_best(silent=True)
            self.gui.save_preset()

        self.assertTrue(applied)
        self._set_different_values()
        self.ui.parallel_slots.setValue(2)
        self.ui.fit_off.setChecked(False)
        self.ui.cache_prompt.setChecked(True)
        loaded = self.config.load_perf_preset(self.model_path, 32768, self.ui)

        self.assertTrue(loaded)
        self.assertEqual(self.ui.threads.value(), 14)
        self.assertEqual(self.ui.parallel_slots.value(), 1)
        self.assertTrue(self.ui.fit_off.isChecked())
        self.assertFalse(self.ui.cache_prompt.isChecked())
        self.assertEqual(self.ui.ctx_checkpoints.value(), 0)
        self.assertEqual(self.ui.cache_ram.value(), 0)


if __name__ == "__main__":
    unittest.main()
