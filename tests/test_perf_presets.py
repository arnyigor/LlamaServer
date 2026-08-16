"""Тесты сохранения/загрузки performance presets и быстрых кнопок Context Size."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
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
        u.speculative_mtp.setChecked(True)
        u.spec_draft_n_max.setValue(8)
        u.spec_draft_p_min.setValue(0.8)
        u.flash_attn.setChecked(False)
        u.fit_off.setChecked(False)
        u.reasoning_mode.setCurrentText("on")
        u.ctx_checkpoints.setValue(4)
        u.cache_ram.setValue(2048)
        u.temperature.setValue(0.7)
        u.top_k.setValue(20)
        u.top_p.setValue(0.9)
        u.min_p.setValue(0.05)
        u.typical_p.setValue(0.95)
        u.repeat_penalty.setValue(1.2)
        u.repeat_last_n.setValue(128)
        u.presence_penalty.setValue(0.2)
        u.frequency_penalty.setValue(0.1)
        u.seed.setValue(42)
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
        u.extra_args.setText("--dry-multiplier 0.8")
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
        u.speculative_mtp.setChecked(False)
        u.spec_draft_n_max.setValue(2)
        u.spec_draft_p_min.setValue(0.1)
        u.flash_attn.setChecked(True)
        u.fit_off.setChecked(True)
        u.reasoning_mode.setCurrentText("off")
        u.ctx_checkpoints.setValue(-1)
        u.cache_ram.setValue(-2)
        u.temperature.setValue(-1.0)
        u.top_k.setValue(-1)
        u.top_p.setValue(-1.0)
        u.min_p.setValue(-1.0)
        u.typical_p.setValue(-1.0)
        u.repeat_penalty.setValue(-1.0)
        u.repeat_last_n.setValue(-2)
        u.presence_penalty.setValue(-3.0)
        u.frequency_penalty.setValue(-3.0)
        u.seed.setValue(-2)
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
        self.assertTrue(u.speculative_mtp.isChecked())
        self.assertEqual(u.spec_draft_n_max.value(), 8)
        self.assertAlmostEqual(u.spec_draft_p_min.value(), 0.8)
        self.assertFalse(u.flash_attn.isChecked())
        self.assertFalse(u.fit_off.isChecked())
        self.assertEqual(u.reasoning_mode.currentText(), "on")
        self.assertEqual(u.ctx_checkpoints.value(), 4)
        self.assertEqual(u.cache_ram.value(), 2048)
        self.assertAlmostEqual(u.temperature.value(), 0.7)
        self.assertEqual(u.top_k.value(), 20)
        self.assertAlmostEqual(u.top_p.value(), 0.9)
        self.assertAlmostEqual(u.min_p.value(), 0.05)
        self.assertAlmostEqual(u.typical_p.value(), 0.95)
        self.assertAlmostEqual(u.repeat_penalty.value(), 1.2)
        self.assertEqual(u.repeat_last_n.value(), 128)
        self.assertAlmostEqual(u.presence_penalty.value(), 0.2)
        self.assertAlmostEqual(u.frequency_penalty.value(), 0.1)
        self.assertEqual(u.seed.value(), 42)
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
        self.assertEqual(u.extra_args.text(), "--dry-multiplier 0.8")
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

    def test_named_presets_keep_distinct_task_values_and_saved_context(self):
        self.ui.preset_name_combo.addItem("coding")
        self.ui.preset_name_combo.addItem("rag")
        self.ui.preset_name_combo.setCurrentText("coding")
        self._set_saved_values()
        with patch.object(QMessageBox, "information", return_value=None):
            self.gui.save_preset()

        if self.ui.preset_name_combo.findText("rag") < 0:
            self.ui.preset_name_combo.addItem("rag")
        self.ui.preset_name_combo.setCurrentText("rag")
        self._set_saved_values()
        self.ui.ctx_size.setValue(32768)
        self.ui.gpu_layers.setValue(55)
        self.ui.threads.setValue(9)
        self.ui.cache_type_k.setCurrentText("q4_1")
        with patch.object(QMessageBox, "information", return_value=None):
            self.gui.save_preset()

        self._set_different_values()
        self.ui.ctx_size.setValue(-1)
        self.ui.preset_name_combo.setCurrentText("coding")
        self.gui._on_preset_selected()

        self._assert_saved_values_loaded()

        self._set_different_values()
        self.ui.ctx_size.setValue(-1)
        self.ui.preset_name_combo.setCurrentText("rag")
        self.gui._on_preset_selected()

        self.assertEqual(self.ui.ctx_size.value(), 32768)
        self.assertEqual(self.ui.gpu_layers.value(), 55)
        self.assertEqual(self.ui.threads.value(), 9)
        self.assertEqual(self.ui.cache_type_k.currentText(), "q4_1")

    def test_add_and_delete_named_preset(self):
        with patch("main.QInputDialog.getText", return_value=("coding", True)):
            self.gui.add_preset()

        self.assertEqual(self.ui.preset_name_combo.currentText(), "coding")
        self.assertTrue(self.ui.delete_preset_btn.isEnabled())

        self._set_saved_values()
        with patch.object(QMessageBox, "information", return_value=None):
            self.gui.save_preset()

        self.assertIn("coding", self.config.list_perf_preset_names(self.model_path))

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.gui.delete_preset()

        self.assertNotIn("coding", self.config.list_perf_preset_names(self.model_path))
        self.assertEqual(self.ui.preset_name_combo.currentText(), "default")
        self.assertFalse(self.ui.delete_preset_btn.isEnabled())

    def test_force_stop_is_available_even_without_owned_process(self):
        self.gui.update_action_buttons()

        self.assertTrue(self.ui.force_stop_btn.isEnabled())

    def test_chat_template_controls_stay_editable_while_server_runs(self):
        with patch.object(self.gui.server, "is_server_running", return_value=True):
            self.gui.update_action_buttons()

        self.assertTrue(self.ui.use_chat_template.isEnabled())
        self.assertTrue(self.ui.chat_template_file.isEnabled())
        self.assertTrue(self.ui.chat_template_btn.isEnabled())
        self.assertFalse(self.ui.temperature.isEnabled())

    def test_mouse_wheel_does_not_change_numeric_fields(self):
        self.ui.port.setValue(8080)
        event = QWheelEvent(
            QPointF(5, 5),
            QPointF(5, 5),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(self.ui.port, event)
        self.assertEqual(self.ui.port.value(), 8080)

    def test_multiple_hf_models_start_as_parallel_independent_tasks(self):
        class FakeSignal:
            def __init__(self):
                self.callbacks = []

            def connect(self, callback):
                self.callbacks.append(callback)

        class FakeDownloader:
            instances = []

            def __init__(self, repo_id, files, model_dir):
                self.repo_id = repo_id
                self.files = files
                self.model_dir = model_dir
                self.progress = FakeSignal()
                self.percent = FakeSignal()
                self.completed = FakeSignal()
                self.finished = FakeSignal()
                self.running = False
                self.paused = False
                self.__class__.instances.append(self)

            def start(self):
                self.running = True

            def isRunning(self):
                return self.running

            def pause(self):
                self.paused = True

            def cancel_and_delete(self):
                self.running = False

        files = [
            {"name": "model-Q4.gguf", "rfilename": "model-Q4.gguf", "size": 100},
            {"name": "model-Q8.gguf", "rfilename": "model-Q8.gguf", "size": 200},
        ]
        self.ui.model_dir.setText(self.tmp.name)
        self.ui.hf_include_mmproj.setChecked(False)
        self.gui.hf_scan_result = {
            "repo_id": "author/model",
            "files": files,
            "projectors": [],
        }
        self.ui.hf_files.setRowCount(0)
        for file_info in files:
            self.gui._add_hf_file_row(file_info)
        selection = self.ui.hf_files.selectionModel()
        for row in range(len(files)):
            selection.select(
                self.ui.hf_files.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows,
            )

        with patch(
            "src.services.hf_download_coordinator.HfModelDownloader", FakeDownloader
        ), patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.gui.download_hf_selection()

        self.assertEqual(len(FakeDownloader.instances), 2)
        self.assertEqual(len(self.gui.hf.tasks()), 2)
        self.assertTrue(all(worker.running for worker in FakeDownloader.instances))
        self.assertEqual(self.ui.hf_downloads.rowCount(), 2)

        task_key = next(iter(self.gui.hf.tasks()))
        progress_text = (
            "model-Q4.gguf (1/1): 25 MiB / 100 MiB; total 25 MiB / 100 MiB, "
            "remaining 75 MiB, ETA 00:30; speed 2.5 MiB/s"
        )
        # Прогресс идёт через координатор: сигнал task_changed рендерит строку.
        self.gui.hf._on_progress(task_key, progress_text)
        row = self.gui.hf.task(task_key)["row"]
        self.assertIn("25 MiB / 100 MiB", self.ui.hf_downloads.item(row, 3).text())
        self.assertIn("2.5 MiB/s", self.ui.hf_downloads.item(row, 4).text())
        self.assertIn("00:30", self.ui.hf_downloads.item(row, 5).text())

        self.gui.pause_hf_download()
        self.assertTrue(all(worker.paused for worker in FakeDownloader.instances))

    def test_partial_download_is_visible_before_hf_scan(self):
        model_dir = Path(self.tmp.name) / "Models"
        partial = model_dir / "author" / "model" / "model-Q4.gguf.part"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"x" * 2048)
        self.ui.model_dir.setText(str(model_dir))

        self.gui._refresh_hf_partial_status()

        self.assertEqual(self.ui.hf_downloads.rowCount(), 1)
        self.assertIn("author/model / model-Q4.gguf", self.ui.hf_downloads.item(0, 0).text())
        self.assertIn("paused / resumable", self.ui.hf_downloads.item(0, 1).text())
        self.assertIn("2.05 KB", self.ui.hf_downloads.item(0, 3).text())
        self.assertIn(str(partial), self.ui.hf_downloads.item(0, 3).toolTip())

        file_info = {
            "name": "model-Q4.gguf",
            "rfilename": "model-Q4.gguf",
            "size": 4096,
        }
        self.gui._on_hf_scan_completed(
            {
                "repo_id": "author/model",
                "files": [file_info],
                "projectors": [],
                "all_files": [file_info],
            }
        )
        progress = self.ui.hf_downloads.cellWidget(0, 2)
        self.assertEqual(progress.value(), 50)
        self.assertIn("4.10 KB", self.ui.hf_downloads.item(0, 3).text())

    def test_auto_detected_separate_mtp_draft_can_be_manually_disabled(self):
        draft_path = Path(self.tmp.name) / "model-MTP-draft.gguf"
        draft_path.write_bytes(b"draft")
        info = {
            "path": self.model_path,
            "_model_path": self.model_path,
            "mtp_capable": True,
            "mtp_draft_path": str(draft_path),
            "architecture": "custommtp",
        }
        self.ui.models_by_path[self.model_path] = info

        self.gui._sync_mtp_controls_for_model(info)
        self.assertEqual(self.ui.spec_draft_model_path.text(), str(draft_path))
        self.assertTrue(self.ui.speculative_mtp.isChecked())
        self.gui.update_cli_preview(force=True)
        self.assertIn("--model-draft", self.ui.cli_preview.text())
        self.assertIn("--spec-type", self.ui.cli_preview.text())

        self.gui._set_mtp_manual_draft_path(str(draft_path), info)
        self.gui._sync_mtp_controls_for_model(info)
        self.assertEqual(self.ui.spec_draft_model_path.text(), str(draft_path))
        self.assertTrue(self.ui.speculative_mtp.isChecked())

        self.ui.spec_draft_model_path.clear()
        self.gui._on_mtp_draft_path_edited("")
        self.ui.speculative_mtp.setChecked(True)
        self.gui._sync_mtp_controls_for_model(info)

        self.assertEqual(self.ui.spec_draft_model_path.text(), "")
        self.assertTrue(self.ui.speculative_mtp.isChecked())
        self.assertFalse(self.gui._auto_mtp_supported(info))
        self.gui.update_cli_preview(force=True)
        self.assertNotIn("--model-draft", self.ui.cli_preview.text())
        self.assertIn("--spec-type", self.ui.cli_preview.text())

        reloaded = ConfigManager(self.settings_path, self.profiles_path)
        reloaded.load()
        self.assertIn(
            os.path.normcase(os.path.abspath(self.model_path)),
            reloaded.settings.spec_draft_auto_disabled_models,
        )
        self.assertNotIn(
            os.path.normcase(os.path.abspath(self.model_path)),
            reloaded.settings.spec_draft_manual_paths,
        )

    def test_manual_mtp_launch_is_not_disabled_for_plain_model_name(self):
        llama_base = Path(self.tmp.name) / "llamacpp"
        llama_build = llama_base / "llama-win-cuda-12.4-x64"
        llama_build.mkdir(parents=True)
        (llama_build / "llama-server.exe").write_bytes(b"exe")

        self.ui.exe_path.setText(str(llama_base))
        self.ui.auto_params.setChecked(True)
        self.ui.models_by_path[self.model_path] = {
            "path": self.model_path,
            "_model_path": self.model_path,
            "architecture": "qwen",
            "mtp_capable": False,
            "recommended_ctx": 8192,
        }
        self.ui.speculative_mtp.setChecked(True)
        self.ui.spec_draft_model_path.clear()
        self.ui.spec_draft_n_max.setValue(8)
        self.ui.spec_draft_p_min.setValue(0.8)
        self.ui.spec_draft_gpu_layers.setText("all")
        self.ui.spec_draft_device.setText("CUDA0")

        launch = self.gui._prepare_server_launch()

        self.assertIsNotNone(launch)
        _exe, args, _env = launch
        self.assertIn("--spec-type", args)
        self.assertEqual(args[args.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(args[args.index("--spec-draft-ngl") + 1], "all")
        self.assertEqual(args[args.index("--spec-draft-device") + 1], "CUDA0")

    def test_qwen_mtp_uses_embedded_speculation_without_neighbor_as_draft(self):
        other_quant = Path(self.tmp.name) / "Qwen3.6-27B-UD-IQ3_XXS.gguf"
        other_quant.write_bytes(b"another quant")
        info = {
            "path": self.model_path,
            "_model_path": self.model_path,
            "mtp_capable": True,
            "mtp_draft_path": str(other_quant),  # stale cache from an older scan
            "architecture": "qwen35moe",
            "block_count": 65,
        }
        self.ui.models_by_path[self.model_path] = info

        self.gui.apply_recommended_params(info)
        self.gui._sync_mtp_controls_for_model(info)
        self.gui.update_cli_preview(force=True)

        command = self.ui.cli_preview.text()
        self.assertTrue(self.ui.speculative_mtp.isChecked())
        self.assertEqual(self.ui.spec_draft_model_path.text(), "")
        self.assertIn("--spec-type draft-mtp", command)
        self.assertIn("--spec-draft-n-max 8", command)
        self.assertIn("--spec-draft-p-min 0.8", command)
        self.assertNotIn("--model-draft", command)

    def test_loading_draft_log_line_is_not_treated_as_mtp_failure(self):
        self.gui.mtp = type(self.gui.mtp)()

        self.gui._on_log_for_mem_viz(
            "common_speculative_init_result: loading draft model 'draft.gguf'",
            "info",
        )

        self.assertFalse(self.gui._mtp_draft_error_seen)

    def test_memory_log_handler_does_not_process_qt_events(self):
        with patch("main.QApplication.processEvents") as process_events:
            self.gui._on_log_for_mem_viz(
                "llama_model_load: model loaded",
                "info",
            )

        process_events.assert_not_called()
        self.assertTrue(self.gui._mem_viz_dirty)

    def test_missing_external_draft_is_not_added_but_qwen_embedded_mtp_remains_supported(self):
        missing_draft = Path(self.tmp.name) / "deleted-MTP-draft.gguf"
        info = {
            "path": self.model_path,
            "_model_path": self.model_path,
            "mtp_capable": True,
            "mtp_draft_path": str(missing_draft),
            "architecture": "qwen3",
        }

        self.gui._sync_mtp_controls_for_model(info)

        self.assertEqual(self.ui.spec_draft_model_path.text(), "")
        self.assertTrue(self.gui._auto_mtp_supported(info))

    def test_legacy_sampling_extra_args_are_migrated(self):
        legacy_path = Path(self.tmp.name) / "legacy-settings.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "extra_args": "--top-k 20 --top-p 0.9 --min-p 0.05 "
                    "--presence-penalty -0.2 --seed 42 --dry-multiplier 0.8"
                }
            ),
            encoding="utf-8",
        )
        manager = ConfigManager(str(legacy_path), self.profiles_path)

        manager.load()

        self.assertEqual(manager.settings.top_k, 20)
        self.assertEqual(manager.settings.top_p, 0.9)
        self.assertEqual(manager.settings.min_p, 0.05)
        self.assertEqual(manager.settings.presence_penalty, -0.2)
        self.assertEqual(manager.settings.seed, 42)
        self.assertEqual(manager.settings.extra_args, "--dry-multiplier 0.8")

    def test_left_autotune_button_opens_benchmark_section_and_builds_plan(self):
        self.ui.ctx_size.setValue(16384)
        self.assertTrue(hasattr(self.ui, "autotune_btn"))

        self.ui.autotune_btn.click()

        self.assertTrue(self.ui.bench_panel.toggle_btn.isChecked())
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

    def test_autotune_finish_does_not_apply_best_without_user_action(self):
        self._set_autotune_best()
        best = self.gui.autotune_best_result

        with patch.object(self.gui, "apply_autotune_best") as apply_best:
            self.gui._on_autotune_finished(best, "benchmarks/test")

        apply_best.assert_not_called()
        self.assertTrue(self.ui.autotune.apply_best_btn.isEnabled())

    def test_apply_selected_autotune_result_uses_selected_candidate_params(self):
        self._set_autotune_best()
        selected = BenchmarkCandidate(
            "run_002",
            {
                **self._make_autotune_candidate().params,
                "ngl": 22,
                "batch_size": 1024,
                "ubatch_size": 256,
                "cache_type_k": "q4_0",
                "cache_type_v": "q4_0",
            },
            "selected conservative run",
            "kv",
        )
        self.gui.autotune_plan.candidates.append(selected)
        selected_result = BenchmarkResult(
            candidate_id="run_002",
            status="success",
            score=95.0,
            prompt_tok_s=900.0,
            generation_tok_s=90.0,
        )
        self.gui.autotune = type("AutoTuneState", (), {"results": [selected_result]})()
        self.ui.autotune.verify_server_after_apply.setChecked(False)
        self.ui.autotune.set_plan(self.gui.autotune_plan)
        self.ui.autotune.update_result(selected_result)
        self.ui.autotune.table.selectRow(1)

        with (
            patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.StandardButton.Yes,
            ),
            patch.object(QMessageBox, "information", return_value=None),
        ):
            applied = self.gui.apply_autotune_selected("run_002")

        self.assertTrue(applied)
        self.assertEqual(self.gui.autotune_best_result.candidate_id, "run_002")
        self.assertEqual(self.ui.gpu_layers.value(), 22)
        self.assertEqual(self.ui.batch_size.value(), 1024)
        self.assertEqual(self.ui.ubatch_size.value(), 256)
        self.assertEqual(self.ui.cache_type_k.currentText(), "q4_0")

    def test_start_autotune_rebuilds_stale_plan_signature(self):
        self.ui.ctx_size.setValue(16384)
        plan = self.gui.build_autotune_plan()
        self.assertIsNotNone(plan)
        old_signature = self.gui._autotune_plan_signature

        self.ui.ctx_size.setValue(32768)
        new_signature = self.gui._current_autotune_plan_signature()

        self.assertNotEqual(old_signature, new_signature)


if __name__ == "__main__":
    unittest.main()
