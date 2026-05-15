"""LlamaServer GUI - точка входа."""

import json
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QSystemTrayIcon,
    QMenu,
)

from src.core.benchmark_plan import build_autotune_plan
from src.core.cli_builder import build_args
from src.core.config import ConfigManager
from src.core.gguf_parser import extract_model_info
from src.core.mem_viz_parser import MemoryData, parse_line
from src.core.server_manager import ServerManager
from src.services.autotune_manager import AutoTuneManager
from src.services.integration_manager import IntegrationManager
from src.services.threads import ModelScanner, LlamaCppUpdater
from src.ui.log_manager import LogManager
from src.ui.main_window import MainWindowUI
from src.ui.tooltips import build_ncmoe_tooltip, build_ctx_tooltip


class LlamaGUI:
    def __init__(self):
        self.ui = MainWindowUI()
        self.config = ConfigManager()
        self.server = ServerManager()
        self.scanner = None
        self.updater = None
        self.autotune = None
        self.autotune_plan = None
        self.autotune_results_dir = ""
        self.autotune_best_result = None
        self._autotune_running = False
        self._restart_pending = False
        self._pending_restart_launch = None
        self._restart_needed = False

        self.log_mgr = LogManager(self.ui.logs)
        self.log_mgr.speed_updated.connect(self.ui.speed_label.setText)
        self.ui.autoscroll_logs.toggled.connect(
            lambda checked: setattr(self.log_mgr, "autoscroll", checked)
        )

        self.config.load()
        self.config.apply_to_ui(self.ui)
        self._connect_signals()
        self._setup_tray()
        QTimer.singleShot(250, self.auto_scan_models)

    def _connect_signals(self):
        u = self.ui
        u.start_btn.clicked.connect(self.start_server)
        u.reload_btn.clicked.connect(self.restart_server)
        u.stop_btn.clicked.connect(self.stop_work)
        u.force_stop_btn.clicked.connect(self.force_stop_server)
        u.test_btn.clicked.connect(self.run_benchmark)
        u.scan_btn.clicked.connect(self.scan_models)
        u.model_combo.currentIndexChanged.connect(self.on_model_selected)
        u.ctx_size.valueChanged.connect(self.on_ctx_changed)
        for btn in getattr(u, "ctx_quick_buttons", []):
            btn.clicked.connect(
                lambda _checked=False, b=btn: self._set_context_size_from_button(
                    b.property("ctx_value")
                )
            )
        u.gpu_layers.valueChanged.connect(self._on_gpu_layers_changed)
        u.cache_type_k.currentIndexChanged.connect(self._on_param_changed)
        u.cache_type_v.currentIndexChanged.connect(self._on_param_changed)
        u.flash_attn.stateChanged.connect(self._on_param_changed)
        u.parallel_slots.valueChanged.connect(self._on_param_changed)
        u.cpu_moe_layers.valueChanged.connect(self._on_param_changed)
        u.gpu_auto.stateChanged.connect(self._on_param_changed)
        u.batch_size.valueChanged.connect(self._on_param_changed)
        u.ubatch_size.valueChanged.connect(self._on_param_changed)
        u.threads.valueChanged.connect(self._on_param_changed)
        u.threads_batch.valueChanged.connect(self._on_param_changed)
        u.fit_off.stateChanged.connect(self._on_param_changed)
        u.reasoning_mode.currentIndexChanged.connect(self._on_param_changed)
        u.port.valueChanged.connect(self._on_param_changed)
        u.ctx_checkpoints.valueChanged.connect(self._on_param_changed)
        u.cache_ram.valueChanged.connect(self._on_param_changed)
        u.temperature.valueChanged.connect(self._on_param_changed)
        u.repeat_penalty.valueChanged.connect(self._on_param_changed)
        u.use_mmap.stateChanged.connect(self._on_param_changed)
        u.use_mlock.stateChanged.connect(self._on_param_changed)
        u.verbose.stateChanged.connect(self._on_param_changed)
        u.log_timestamps.stateChanged.connect(self._on_param_changed)
        u.cont_batching.stateChanged.connect(self._on_param_changed)
        u.cache_prompt.stateChanged.connect(self._on_param_changed)
        u.context_shift.stateChanged.connect(self._on_param_changed)
        u.no_webui.stateChanged.connect(self._on_param_changed)
        u.use_mmproj.stateChanged.connect(self._on_param_changed)
        u.mmproj_offload.stateChanged.connect(self._on_param_changed)
        u.extra_args.textChanged.connect(self._on_param_changed)
        u.jinja.stateChanged.connect(self._on_param_changed)
        u.enable_thinking.currentIndexChanged.connect(self._on_param_changed)
        u.update_llama_btn.clicked.connect(self.update_llamacpp)
        u.integration_check_btn.clicked.connect(self.check_integration_models)
        u.integration_add_btn.clicked.connect(self.add_model_to_integration)
        u.integration_remove_btn.clicked.connect(self.remove_model_from_integration)
        u.integration_target.currentIndexChanged.connect(self.check_integration_models)
        u.opencode_config_path.editingFinished.connect(self._on_config_path_changed)
        u.pi_config_path.editingFinished.connect(self._on_config_path_changed)
        u.exe_path.textChanged.connect(self.auto_detect_bench)
        u._browse_exe_clicked = self.browse_exe
        u._browse_bench_clicked = self.browse_bench
        u._browse_model_dir_clicked = self.browse_model_dir
        u._browse_opencode_clicked = self.browse_opencode_config
        u._browse_pi_clicked = self.browse_pi_config
        u.save_preset_btn.clicked.connect(self.save_preset)
        u.autotune_btn.clicked.connect(self.open_autotune_tab)
        u.autotune.build_plan_requested.connect(self.build_autotune_plan)
        u.autotune.start_requested.connect(self.start_autotune)
        u.autotune.cancel_requested.connect(self.cancel_autotune)
        u.autotune.apply_best_requested.connect(self.apply_autotune_best)
        u.autotune.save_best_requested.connect(self.save_autotune_best_preset)
        u.autotune.export_report_requested.connect(self.show_autotune_report_path)
        u.autotune.open_results_requested.connect(self.open_autotune_results_folder)

        self.server.log_received.connect(
            lambda text, level: self.log_mgr.append(text, level)
        )
        self.server.state_changed.connect(self.update_action_buttons)
        self.server.bench_finished.connect(lambda _: self.update_action_buttons())

        # Парсинг логов для визуализации памяти
        self._mem_data = MemoryData()
        self.server.log_received.connect(self._on_log_for_mem_viz)
        self.server.server_stopped.connect(self._on_server_stopped)
        self.server.bench_finished.connect(self._on_bench_finished)

    def browse_opencode_config(self):
        f, _ = QFileDialog.getOpenFileName(
            self.ui, "Select opencode.json", "", "JSON (*.json);;All files (*.*)"
        )
        if f:
            self.ui.opencode_config_path.setText(f)
            self.save_settings()
            self.check_integration_models(silent=True)

    def browse_pi_config(self):
        f, _ = QFileDialog.getOpenFileName(
            self.ui, "Select PI config.json", "", "JSON (*.json);;All files (*.*)"
        )
        if f:
            self.ui.pi_config_path.setText(f)
            self.save_settings()
            self.check_integration_models(silent=True)

    def _setup_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(self.ui)
        self.tray.setToolTip("LlamaServer GUI")
        menu = QMenu()
        menu.addAction("Show", self.ui.showNormal)
        menu.addAction("Hide", self.ui.hide)
        menu.addSeparator()
        menu.addAction("Exit", self.quit_app)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda r: self.ui.hide()
            if self.ui.isVisible() and r == QSystemTrayIcon.DoubleClick
            else self.ui.showNormal()
        )
        self.tray.show()

    def save_settings(self):
        self.config.read_from_ui(self.ui)
        self.config.settings.model_cache = self.ui.models
        self.config.save()

    def _current_model_path(self):
        path = self.ui.model_combo.currentData()
        if path:
            return path

        text = self.ui.model_combo.currentText().strip()
        if text and os.path.exists(text) and text.lower().endswith(".gguf"):
            return text

        return None

    def save_preset(self):
        model_path = self._current_model_path()
        if not model_path:
            QMessageBox.warning(self.ui, "Error", "Select a model first")
            return

        ctx = self.ui.ctx_size.value()
        if ctx <= 0:
            QMessageBox.warning(
                self.ui,
                "Error",
                "Preset requires specific Context Size, not auto",
            )
            return

        try:
            self.config.save_perf_preset(model_path, ctx, self.ui)
        except ValueError as e:
            QMessageBox.warning(self.ui, "Error", str(e))
            return
        except OSError as e:
            QMessageBox.critical(
                self.ui,
                "Error",
                f"Failed to save preset: {e}",
            )
            return

        self.log_mgr.append(
            f"Preset saved: {os.path.basename(model_path)} | ctx={ctx:,}"
        )
        QMessageBox.information(
            self.ui,
            "Saved",
            f"Parameters for ctx={ctx:,} saved.",
        )

    def _try_load_perf_preset(self, model_path: str, ctx_size: int) -> bool:
        if not model_path or ctx_size <= 0:
            return False

        if getattr(self, "_loading_preset", False):
            return False

        self._loading_preset = True
        try:
            loaded = self.config.load_perf_preset(model_path, ctx_size, self.ui)
        finally:
            self._loading_preset = False

        if not loaded:
            return False

        self.log_mgr.append(
            f"Loaded preset: {os.path.basename(model_path)} | ctx={ctx_size:,}"
        )

        if hasattr(self.ui, "preset_status"):
            self.ui.preset_status.setText(f"Preset: loaded ctx={ctx_size:,}")
            self.ui.preset_status.setStyleSheet("color: #4CAF50;")

        info = self.ui.models_by_path.get(model_path)
        if info:
            self._refresh_tooltips(info)

        self.update_cli_preview()
        return True

    def auto_detect_bench(self):
        srv = self.ui.exe_path.text()
        if srv and os.path.exists(srv):
            bench = os.path.join(
                os.path.dirname(srv), os.path.basename(srv).replace("server", "bench")
            )
            if os.path.exists(bench):
                self.ui.bench_path.setText(bench)

    def browse_exe(self):
        f, _ = QFileDialog.getOpenFileName(
            self.ui, "Select llama-server", "", "Executable (*.exe)"
        )
        if f:
            self.ui.exe_path.setText(f)
            self.save_settings()

    def browse_bench(self):
        f, _ = QFileDialog.getOpenFileName(
            self.ui, "Select llama-bench", "", "Executable (*.exe)"
        )
        if f:
            self.ui.bench_path.setText(f)
            self.save_settings()

    def browse_model_dir(self):
        d = QFileDialog.getExistingDirectory(self.ui, "Select models folder")
        if d:
            self.ui.model_dir.setText(d)
            self.save_settings()
            self.scan_models()

    def auto_scan_models(self):
        if self.ui.models:
            self.ui.scan_status.setText(
                f"Model cache: {len(self.ui.models)}. Background check..."
            )
        bp = self.ui.model_dir.text()
        if bp and os.path.exists(bp):
            self.scan_models(silent=True)

    def scan_models(self, silent=False):
        bp = self.ui.model_dir.text()
        if not bp or not os.path.exists(bp):
            if not silent:
                QMessageBox.warning(self.ui, "Error", "Specify existing base folder")
            return
        if self.scanner and self.scanner.isRunning():
            if not silent:
                self.scanner.requestInterruption()
            return
        if self.scanner:
            self.scanner.deleteLater()
            self.scanner = None
        self.ui.scan_btn.setText("Cancel")
        self.ui.scan_progress.setVisible(True)
        self.ui.scan_status.setText("Scanning GGUF...")
        self.scanner = ModelScanner(bp)
        self.scanner.progress.connect(self.ui.scan_status.setText)
        self.scanner.models_found.connect(self.on_models_found)
        self.scanner.error.connect(lambda msg: self.log_mgr.append(msg, "error"))
        self.scanner.finished.connect(
            lambda: self.ui.scan_btn.setText("Scan")
            or self.ui.scan_progress.setVisible(False)
        )
        self.scanner.start()

    def on_models_found(self, models):
        self.ui.models = models
        self.ui.models_by_path = {m["path"]: m for m in models}
        self.ui.model_combo.clear()
        for m in models:
            self.ui.model_combo.addItem(m["display"], m["path"])
        last = self.config.settings.last_model_path
        idx = self.ui.model_combo.findData(last)
        if idx >= 0:
            self.ui.model_combo.setCurrentIndex(idx)
        elif models:
            self.ui.model_combo.setCurrentIndex(0)
        self.save_settings()
        self.log_mgr.append(f"Found models: {len(models)}")
        self.ui.scan_status.setText(f"Found models: {len(models)}")

    def on_model_selected(self, *_):
        path = self.ui.model_combo.currentData()
        if not path:
            txt = self.ui.model_combo.currentText().strip()
            if txt and os.path.exists(txt) and txt.lower().endswith(".gguf"):
                path = txt
                self.ui.model_combo.setItemData(
                    self.ui.model_combo.currentIndex(), path
                )
            else:
                self.ui.model_info.setText("Select model")
                return
        info = self.ui.models_by_path.get(path) or extract_model_info(path)
        self.ui.models_by_path[path] = info

        arch = info.get("architecture") or "?"
        quant = info.get("quant") or "?"
        size = info.get("size_gib", 0)
        block_count = info.get("block_count", 0)
        head_count = info.get("head_count", 0)
        emb_len = info.get("embedding_length", 0)
        expert_count = info.get("expert_count", 0)
        expert_used = info.get("expert_used", 0)
        ctx = info.get("context_length", 0)
        rec_ctx = info.get("recommended_ctx", 0)

        parts = [f"Architecture: {arch} | {quant} | {size:.2f} GiB"]

        layer_str = f"Layers: {block_count}" if block_count else "Layers: ?"
        if head_count:
            layer_str += f" | Heads: {head_count}"
        if emb_len:
            layer_str += f" | Emb: {emb_len}"
        parts.append(layer_str)

        if expert_count:
            moe_str = f"MoE: {expert_count} experts"
            if expert_used:
                moe_str += f", active: {expert_used}"
            parts.append(moe_str)

        if ctx:
            parts.append(f"Context: {ctx:,} | Rec: {rec_ctx:,}")

        if info.get("mmproj_path"):
            from pathlib import Path

            mmproj_name = Path(info["mmproj_path"]).name
            parts.append(f"mmproj: {mmproj_name}")

        if info.get("metadata_error"):
            parts.append(f"Warning: {info['metadata_error']}")

        self.ui.model_info.setText("\n".join(parts))

        self._refresh_tooltips(info)

        self.config.settings.last_model_path = path

        if self.ui.auto_params.isChecked() and not self.ui.loading_profile:
            self._loading_preset = True
            try:
                self.apply_recommended_params(info)
            finally:
                self._loading_preset = False

        ctx_size = self.ui.ctx_size.value()
        self._try_load_perf_preset(path, ctx_size)

        self.update_cli_preview()
        self._mark_restart_needed()

    def _refresh_tooltips(self, info):
        """Обновление tooltip для ncmoe и ctx при смене модели."""
        expert_count = info.get("expert_count", 0)

        if expert_count:
            gpu_layers_val = (
                999 if self.ui.gpu_auto.isChecked() else self.ui.gpu_layers.value()
            )
            tooltip = build_ncmoe_tooltip(
                info=info,
                ctx_size=self.ui.ctx_size.value(),
                gpu_layers=gpu_layers_val,
                cache_type_k=self.ui.cache_type_k.currentText(),
                cache_type_v=self.ui.cache_type_v.currentText(),
                flash_attn=self.ui.flash_attn.isChecked(),
                parallel_slots=self.ui.parallel_slots.value(),
                current_ncmoe=self.ui.cpu_moe_layers.value(),
            )
            self.ui.cpu_moe_layers.setToolTip(tooltip)
        else:
            self.ui.cpu_moe_layers.setToolTip("Model is not MoE")

        tooltip_ctx = build_ctx_tooltip(
            info=info,
            current_ctx=self.ui.ctx_size.value(),
            gpu_layers=self.ui.gpu_layers.value(),
            cache_type_k=self.ui.cache_type_k.currentText(),
            cache_type_v=self.ui.cache_type_v.currentText(),
            flash_attn=self.ui.flash_attn.isChecked(),
            parallel_slots=self.ui.parallel_slots.value(),
        )
        self.ui.ctx_size.setToolTip(tooltip_ctx)

    def apply_recommended_params(self, info):
        rec = info.get("recommended_ctx")
        if rec:
            self.ui.ctx_size.setValue(rec)
        q = (info.get("quant") or "").upper()
        if (
            q.startswith(("Q2", "Q3", "IQ1", "IQ2", "IQ3"))
            or info.get("recommended_ctx", 0) >= 16384
        ):
            self.ui.cache_type_k.setCurrentText("q8_0")
            self.ui.cache_type_v.setCurrentText("q8_0")
        else:
            self.ui.cache_type_k.setCurrentText("f16")
            self.ui.cache_type_v.setCurrentText("f16")
        self.ui.batch_size.setValue(2048)
        self.ui.ubatch_size.setValue(2048)

    def _set_context_size_from_button(self, ctx_size):
        ctx_size = int(ctx_size)
        if self.ui.ctx_size.value() == ctx_size:
            # setValue() не эмитит valueChanged для того же значения, поэтому
            # явно повторяем логику загрузки пресета по кнопке.
            self.on_ctx_changed(ctx_size)
        else:
            self.ui.ctx_size.setValue(ctx_size)

    def on_ctx_changed(self, ctx_size):
        if getattr(self, "_loading_preset", False) or self.ui.loading_profile:
            return

        info = self.ui.models_by_path.get(self.ui.model_combo.currentData())
        if not info:
            return

        gpu_layers_val = (
            999 if self.ui.gpu_auto.isChecked() else self.ui.gpu_layers.value()
        )
        tooltip = build_ncmoe_tooltip(
            info=info,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers_val,
            cache_type_k=self.ui.cache_type_k.currentText(),
            cache_type_v=self.ui.cache_type_v.currentText(),
            flash_attn=self.ui.flash_attn.isChecked(),
            parallel_slots=self.ui.parallel_slots.value(),
            current_ncmoe=self.ui.cpu_moe_layers.value(),
        )
        self.ui.cpu_moe_layers.setToolTip(tooltip)

        tooltip_ctx = build_ctx_tooltip(
            info=info,
            current_ctx=ctx_size,
            gpu_layers=self.ui.gpu_layers.value(),
            cache_type_k=self.ui.cache_type_k.currentText(),
            cache_type_v=self.ui.cache_type_v.currentText(),
            flash_attn=self.ui.flash_attn.isChecked(),
            parallel_slots=self.ui.parallel_slots.value(),
        )
        self.ui.ctx_size.setToolTip(tooltip_ctx)

        preset_loaded = False
        model_path = self._current_model_path()
        if model_path:
            preset_loaded = self._try_load_perf_preset(model_path, ctx_size)

        if not preset_loaded:
            self.update_cli_preview()

        self._mark_restart_needed()

    def _on_gpu_layers_changed(self, value):
        if getattr(self, "_loading_preset", False) or self.ui.loading_profile:
            return

        info = self.ui.models_by_path.get(self.ui.model_combo.currentData())
        if info:
            self._refresh_tooltips(info)

        self.update_cli_preview()
        self._mark_restart_needed()

    def _on_param_changed(self, _value=None):
        if getattr(self, "_loading_preset", False) or self.ui.loading_profile:
            return

        info = self.ui.models_by_path.get(self.ui.model_combo.currentData())
        if info:
            self._refresh_tooltips(info)

        self.update_cli_preview()
        self._mark_restart_needed()

    def update_cli_preview(self):
        try:
            self.config.read_from_ui(self.ui)
            args = build_args(self.config.settings, self.ui.model_combo.currentData())
            exe = self.ui.exe_path.text() or "llama-server.exe"
            self.ui.cli_preview.setText(f"{exe} {' '.join(args)}" if args else "")
        except Exception:
            self.ui.cli_preview.setText("")

    def _mark_restart_needed(self):
        """Подсвечивает, что запущенному серверу нужен рестарт для новых параметров."""
        if not self.server.is_server_running():
            return
        self._restart_needed = True
        self.ui.start_btn.setVisible(False)
        self.ui.reload_btn.setVisible(True)
        self.ui.reload_btn.setText("Restart to apply")
        self.ui.reload_btn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
        )
        self.ui.reload_btn.setEnabled(True)

    def _reset_restart_indicator(self):
        self._restart_needed = False
        self.ui.reload_btn.setText("Restart")
        self.ui.reload_btn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
        )

    def _on_log_for_mem_viz(self, text: str, level: str):
        """Обработка логов для визуализации памяти."""
        for line in text.splitlines():
            parse_line(line, self._mem_data)
        # Принудительно обновляем UI после каждого блока логов
        self.ui.mem_viz.update_from_data(self._mem_data)
        # Обрабатываем события Qt чтобы UI не зависал
        QApplication.processEvents()

    def _reset_mem_viz(self, status: str | None = None):
        """Сброс визуализации памяти."""
        self._mem_data = MemoryData()
        self.ui.mem_viz.clear()
        if status:
            self.ui.mem_viz.status_label.setText(status)

    def _finalize_mem_viz_after_stop(self, exit_code: int | None, status: str):
        """Обновляет вкладку Memory после выгрузки модели/остановки процесса."""
        if self._mem_data.fatal_error:
            # При ошибке оставляем диагностические данные, чтобы было видно,
            # какой компонент и сколько памяти пытались выделить.
            self._mem_data.process_exit_code = exit_code
            self.ui.mem_viz.update_from_data(self._mem_data)
        else:
            # Нормальная выгрузка освобождает RAM/VRAM — старые allocations
            # больше неактуальны, поэтому полностью очищаем вкладку.
            self._reset_mem_viz(status)
        self.ui.tabs.setCurrentIndex(1)

    def _on_server_stopped(self):
        """Обработчик остановки сервера."""
        if self._restart_pending:
            self._reset_mem_viz("Сервер остановлен, перезапуск с новыми параметрами...")
            QTimer.singleShot(150, self._start_pending_restart)
            return
        self._finalize_mem_viz_after_stop(
            self.server.server_proc.exitCode(),
            "Сервер остановлен",
        )

    def _on_bench_finished(self, exit_code: int):
        self._finalize_mem_viz_after_stop(exit_code, "Benchmark завершён")

    def _prepare_server_launch(self):
        if self.server.is_bench_running():
            QMessageBox.warning(
                self.ui,
                "Benchmark running",
                "Stop benchmark before starting server",
            )
            return None
        exe = self.ui.exe_path.text()
        if not exe or not os.path.exists(exe):
            QMessageBox.critical(self.ui, "Error", "Specify path to llama-server.exe")
            return None
        self.config.read_from_ui(self.ui)
        # resolve mmproj
        info = self.ui.models_by_path.get(self.ui.model_combo.currentData()) or {}
        self.config.settings.mmproj_path = info.get("mmproj_path", "")
        try:
            args = build_args(self.config.settings, self.ui.model_combo.currentData())
        except ValueError as e:
            QMessageBox.warning(self.ui, "Error", str(e))
            return None
        if not args:
            return None
        return exe, args

    def _launch_server(self, exe: str, args: list[str], action: str = "Starting server"):
        self.log_mgr.append(f"{action}: {exe}\n   Args: {' '.join(args)}")
        self._reset_mem_viz()
        self.server.start_server(exe, args)
        self._reset_restart_indicator()
        self.ui.start_btn.setVisible(False)
        self.ui.reload_btn.setVisible(True)
        self.ui.start_btn.setEnabled(False)
        self.ui.reload_btn.setEnabled(True)
        self.ui.test_btn.setEnabled(False)
        self.ui.stop_btn.setEnabled(True)
        self.ui.force_stop_btn.setEnabled(True)
        if hasattr(self, "tray"):
            self.tray.setToolTip(
                f"LlamaServer GUI - Running on port {self.ui.port.value()}"
            )

    def _start_pending_restart(self):
        if not self._restart_pending:
            return
        if self.server.is_server_running():
            QTimer.singleShot(150, self._start_pending_restart)
            return
        launch = self._pending_restart_launch
        self._restart_pending = False
        self._pending_restart_launch = None
        if launch:
            exe, args = launch
            self._launch_server(exe, args, action="Restarting server")

    def restart_server(self):
        """Перезапускает llama-server с текущими параметрами UI."""
        if not self.server.is_server_running():
            self.start_server()
            return
        launch = self._prepare_server_launch()
        if not launch:
            return
        self._pending_restart_launch = launch
        self._restart_pending = True
        self.log_mgr.append("Restart requested: stopping current server...")
        self._reset_mem_viz("Остановка сервера для перезапуска...")
        self.server.stop_server()
        self.update_action_buttons()

    def start_server(self):
        if self.server.is_server_running():
            self.restart_server()
            return
        launch = self._prepare_server_launch()
        if not launch:
            return
        exe, args = launch
        self._launch_server(exe, args)

    def run_benchmark(self):
        if self.server.is_server_running():
            QMessageBox.warning(
                self.ui, "Server running", "Stop server before running benchmark"
            )
            return
        self.auto_detect_bench()
        bexe = self.ui.bench_path.text()
        if not bexe or not os.path.exists(bexe):
            QMessageBox.critical(self.ui, "Error", "Specify path to llama-bench.exe")
            return
        self.config.read_from_ui(self.ui)
        try:
            args = build_args(
                self.config.settings,
                self.ui.model_combo.currentData(),
                for_benchmark=True,
            )
        except ValueError as e:
            QMessageBox.warning(self.ui, "Error", str(e))
            return
        if not args:
            return
        self.log_mgr.append(
            f"Running benchmark: {os.path.basename(bexe)}\n   Params: {' '.join(args)}"
        )
        self._reset_mem_viz()
        self.server.start_bench(bexe, args)
        self.ui.test_btn.setEnabled(False)
        self.ui.test_btn.setText("Testing...")
        self.ui.start_btn.setEnabled(False)
        self.ui.stop_btn.setEnabled(True)

    def _current_model_info(self):
        model_path = self._current_model_path()
        if not model_path:
            return {}
        info = self.ui.models_by_path.get(model_path)
        if not info:
            info = extract_model_info(model_path)
            self.ui.models_by_path[model_path] = info
        return info

    def open_autotune_tab(self):
        """Открывает вкладку AutoTune и строит свежий план из текущих настроек."""
        self.ui.tabs.setCurrentWidget(self.ui.autotune)
        self.build_autotune_plan()

    def build_autotune_plan(self):
        model_path = self._current_model_path()
        if not model_path:
            QMessageBox.warning(self.ui, "AutoTune", "Select a GGUF model first")
            return None
        self.config.read_from_ui(self.ui)
        options = self.ui.autotune.options()
        plan = build_autotune_plan(
            self.config.settings,
            model_path,
            self._current_model_info(),
            mode=options["mode"],
            target=options["target"],
            engine=options["engine"],
            time_budget_sec=options["time_budget_sec"],
            max_runs=options["max_runs"],
            repeat_top=options["repeat_top"],
        )
        self.autotune_plan = plan
        self.autotune_best_result = None
        self.autotune_results_dir = ""
        self.ui.autotune.set_plan(plan)
        self.ui.tabs.setCurrentWidget(self.ui.autotune)
        self.log_mgr.append(
            f"AutoTune plan built: {len(plan.candidates)} candidates | ctx={plan.ctx_size:,} | {plan.mode}/{plan.target}"
        )
        return plan

    def _external_llama_processes(self):
        """Ищет orphan/external llama.cpp процессы, которые ломают benchmark."""
        if not sys.platform.startswith("win"):
            return []
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process | "
                    "Where-Object { $_.Name -match '^llama-(server|bench)(\\.exe)?$' } | "
                    "Select-Object ProcessId,Name,ExecutablePath | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                encoding="utf-8",
                errors="ignore",
            )
            text = (proc.stdout or "").strip()
            if not text:
                return []
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
            return [p for p in data if isinstance(p, dict)]
        except Exception:
            return []

    def _warn_if_external_llama_processes(self) -> bool:
        processes = self._external_llama_processes()
        if not processes:
            return False
        details = "\n".join(
            f"PID {p.get('ProcessId')}: {p.get('Name')} — {p.get('ExecutablePath') or ''}"
            for p in processes[:8]
        )
        QMessageBox.warning(
            self.ui,
            "AutoTune blocked",
            "Found already running llama.cpp process. It can occupy VRAM/GPU and make AutoTune results invalid.\n\n"
            "Stop it first, then start AutoTune again.\n\n"
            f"{details}",
        )
        self.log_mgr.append(
            "AutoTune blocked: external llama.cpp process is running. Stop it before benchmarking.",
            "warn",
        )
        return True

    def start_autotune(self):
        if self.server.is_server_running() or self.server.is_bench_running():
            QMessageBox.warning(
                self.ui, "AutoTune", "Stop server/manual benchmark before AutoTune"
            )
            return
        if self._warn_if_external_llama_processes():
            return
        if self.autotune and self.autotune.isRunning():
            return
        self.auto_detect_bench()
        bexe = self.ui.bench_path.text().strip()
        if not bexe or not os.path.exists(bexe):
            QMessageBox.critical(self.ui, "AutoTune", "Specify path to llama-bench.exe")
            return
        plan = self.autotune_plan or self.build_autotune_plan()
        if not plan:
            return
        options = self.ui.autotune.options()
        if str(options.get("engine", "llama-bench")) == "llama-server":
            QMessageBox.information(
                self.ui,
                "AutoTune",
                "Server AutoTune is planned for a later version. MVP uses llama-bench.",
            )
            return
        self.ui.autotune.clear_results()
        self.ui.autotune.set_plan(plan)
        self.autotune_best_result = None
        self.autotune_results_dir = ""
        self.autotune = AutoTuneManager(
            bexe,
            plan,
            model_info=self._current_model_info(),
            prompt_tokens=self.ui.bench_prompt.value(),
            generation_tokens=self.ui.bench_gen.value(),
            per_run_timeout_sec=options["per_run_timeout_sec"],
        )
        self.ui.autotune.prepare_run(len(plan.candidates), options["per_run_timeout_sec"])
        self.autotune.log.connect(lambda text, level: self.log_mgr.append(text, level))
        self.autotune.log.connect(lambda text, _level: self.ui.autotune.append_activity(text))
        self.autotune.progress.connect(self.ui.autotune.set_progress)
        self.autotune.run_started.connect(self.ui.autotune.mark_running)
        self.autotune.run_finished.connect(self.ui.autotune.update_result)
        self.autotune.autotune_finished.connect(self._on_autotune_finished)
        self.autotune.finished.connect(self.update_action_buttons)
        self._autotune_running = True
        self.ui.autotune.set_running(True)
        self.autotune.start()
        self.update_action_buttons()

    def cancel_autotune(self):
        if self.autotune and self.autotune.isRunning():
            self.log_mgr.append("AutoTune cancel requested", "warn")
            self.autotune.cancel()

    def _best_autotune_params(self):
        if not self.autotune_plan or not self.autotune_best_result:
            return {}
        for candidate in self.autotune_plan.candidates:
            if candidate.id == self.autotune_best_result.candidate_id:
                return dict(candidate.params)
        return {}

    def _on_autotune_finished(self, best, output_dir):
        self.autotune_best_result = best
        self.autotune_results_dir = output_dir
        self._autotune_running = False
        self.ui.autotune.set_running(False)
        self.ui.autotune.show_best(best, self._best_autotune_params(), output_dir)
        if best:
            self.log_mgr.append(
                f"AutoTune finished: best={best.candidate_id}, score={best.score:.3f}, results={output_dir}"
            )
        else:
            self.log_mgr.append(f"AutoTune finished: no successful result, results={output_dir}", "warn")
        self.update_action_buttons()

    def apply_autotune_best(self, silent=False):
        params = self._best_autotune_params()
        if not params:
            if not silent:
                QMessageBox.warning(self.ui, "AutoTune", "No best result to apply")
            return False

        self._loading_preset = True
        try:
            ngl = params.get("ngl", "auto")
            is_auto_ngl = str(ngl).lower() == "auto"
            self.ui.gpu_auto.setChecked(is_auto_ngl)
            if not is_auto_ngl:
                self.ui.gpu_layers.setValue(int(ngl))
            self.ui.ctx_size.setValue(int(params.get("ctx_size", self.ui.ctx_size.value())))
            self.ui.batch_size.setValue(int(params.get("batch_size", self.ui.batch_size.value())))
            self.ui.ubatch_size.setValue(int(params.get("ubatch_size", self.ui.ubatch_size.value())))
            self.ui.cache_type_k.setCurrentText(str(params.get("cache_type_k", self.ui.cache_type_k.currentText())))
            self.ui.cache_type_v.setCurrentText(str(params.get("cache_type_v", self.ui.cache_type_v.currentText())))
            self.ui.threads.setValue(int(params.get("threads", self.ui.threads.value())))
            self.ui.threads_batch.setValue(int(params.get("threads_batch", self.ui.threads_batch.value())))
            self.ui.parallel_slots.setValue(int(params.get("parallel_slots", self.ui.parallel_slots.value())))
            self.ui.flash_attn.setChecked(bool(params.get("flash_attn", self.ui.flash_attn.isChecked())))
            self.ui.fit_off.setChecked(bool(params.get("fit_off", self.ui.fit_off.isChecked())))
            self.ui.cache_prompt.setChecked(bool(params.get("cache_prompt", self.ui.cache_prompt.isChecked())))
            self.ui.cpu_moe_layers.setValue(int(params.get("ncmoe", self.ui.cpu_moe_layers.value())))
            self.ui.ctx_checkpoints.setValue(int(params.get("ctx_checkpoints", self.ui.ctx_checkpoints.value())))
            self.ui.cache_ram.setValue(int(params.get("cache_ram", self.ui.cache_ram.value())))
            self.ui.use_mmproj.setChecked(bool(params.get("use_mmproj", self.ui.use_mmproj.isChecked())))
        finally:
            self._loading_preset = False

        self.update_cli_preview()
        self._mark_restart_needed()
        self.save_settings()
        self.log_mgr.append(
            f"AutoTune best applied: {self.autotune_best_result.candidate_id if self.autotune_best_result else ''}"
        )
        if not silent:
            QMessageBox.information(self.ui, "AutoTune", "Best parameters applied to UI")
        return True

    def save_autotune_best_preset(self):
        model_path = self._current_model_path()
        if not model_path or not self.autotune_best_result:
            QMessageBox.warning(self.ui, "AutoTune", "No best result to save")
            return
        if not self.apply_autotune_best(silent=True):
            return
        ctx = self.ui.ctx_size.value()
        metadata = {
            "source": "autotune",
            "run_id": self.autotune_best_result.candidate_id,
            "score": self.autotune_best_result.score,
            "prompt_tok_s": self.autotune_best_result.prompt_tok_s,
            "generation_tok_s": self.autotune_best_result.generation_tok_s,
            "load_time_sec": self.autotune_best_result.load_time_sec,
            "vram_used_mib": self.autotune_best_result.vram_used_mib,
            "ram_used_mib": self.autotune_best_result.ram_used_mib,
            "results_dir": self.autotune_results_dir,
        }
        try:
            self.config.save_perf_preset(model_path, ctx, self.ui, metadata=metadata)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self.ui, "AutoTune", str(e))
            return
        self.log_mgr.append(f"AutoTune preset saved: {Path(model_path).name} | ctx={ctx:,}")
        QMessageBox.information(self.ui, "AutoTune", "Best AutoTune preset saved")

    def show_autotune_report_path(self):
        if not self.autotune_results_dir:
            QMessageBox.information(self.ui, "AutoTune", "No report yet")
            return
        QMessageBox.information(
            self.ui,
            "AutoTune Report",
            f"results.json and report.md are saved in:\n{self.autotune_results_dir}",
        )

    def open_autotune_results_folder(self):
        if not self.autotune_results_dir or not os.path.isdir(self.autotune_results_dir):
            QMessageBox.information(self.ui, "AutoTune", "No results folder yet")
            return
        if sys.platform.startswith("win"):
            os.startfile(self.autotune_results_dir)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", self.autotune_results_dir])
        else:
            subprocess.Popen(["xdg-open", self.autotune_results_dir])

    def stop_work(self):
        if self._restart_pending:
            self._restart_pending = False
            self._pending_restart_launch = None
            self.log_mgr.append("Restart cancelled")
        if self.server.is_server_running():
            self.server.stop_server()
        if self.server.is_bench_running():
            self.server.stop_bench()
        if self.autotune and self.autotune.isRunning():
            self.autotune.cancel()
        if self.scanner and self.scanner.isRunning():
            self.scanner.requestInterruption()
        self.update_action_buttons()

    def force_stop_server(self):
        if self._restart_pending:
            self._restart_pending = False
            self._pending_restart_launch = None
            self.log_mgr.append("Restart cancelled")
        if not self.server.is_server_running():
            self.log_mgr.append("Force stop skipped: server is not running", "warn")
            self.update_action_buttons()
            return
        self.log_mgr.append("Force stop requested: killing llama-server now", "error")
        self.server.force_stop_server()
        self.update_action_buttons()

    def update_action_buttons(self, busy=False):
        srv = self.server.is_server_running()
        bnch = self.server.is_bench_running()
        scan = self.scanner and self.scanner.isRunning()
        upd = self.updater and self.updater.isRunning()
        tune = self._autotune_running or (self.autotune and self.autotune.isRunning())
        busy = srv or bnch or scan or tune or self._restart_pending
        show_reload = srv or self._restart_pending
        self.ui.start_btn.setVisible(not show_reload)
        self.ui.reload_btn.setVisible(show_reload)
        self.ui.stop_btn.setEnabled(busy)
        self.ui.force_stop_btn.setEnabled(srv)
        self.ui.update_llama_btn.setEnabled(not busy and not upd)
        self.ui.start_btn.setEnabled(not busy and not upd)
        self.ui.reload_btn.setEnabled(
            srv
            and not bnch
            and not scan
            and not upd
            and not tune
            and not self._restart_pending
            and not self.server.server_stop_requested
        )
        self.ui.test_btn.setEnabled(not busy and not upd)
        self.ui.autotune_btn.setEnabled(not busy and not upd)
        self.ui.autotune.start_btn.setEnabled(not busy and not upd)
        self.ui.autotune.build_plan_btn.setEnabled(not busy and not upd)
        self.ui.autotune.cancel_btn.setEnabled(bool(tune))
        if not srv and not self._restart_pending:
            self._reset_restart_indicator()

    def update_llamacpp(self):
        if self.server.is_server_running() or self.server.is_bench_running():
            QMessageBox.warning(self.ui, "Updater", "Stop processes before updating.")
            return
        exe = self.ui.exe_path.text().strip()
        if not exe or not os.path.exists(exe):
            QMessageBox.critical(
                self.ui, "Updater", "Select existing llama-server.exe first."
            )
            return
        self.ui.update_progress.setValue(0)
        self.ui.update_progress.setVisible(True)
        self.updater = LlamaCppUpdater(exe)
        self.updater.progress.connect(self.ui.update_status.setText)
        self.updater.percent.connect(self.ui.update_progress.setValue)
        self.updater.completed.connect(
            lambda ch, msg: self.ui.update_status.setText(msg)
            or self.auto_detect_bench()
            or self.save_settings()
        )
        self.updater.finished.connect(
            lambda: self.ui.update_progress.setVisible(False)
            or self.update_action_buttons()
        )
        self.updater.start()
        self.update_action_buttons()

    # Integration & Profiles logic
    def check_integration_models(self, silent=False):
        target = self.ui.current_config_target()
        config_path = self.ui.current_config_path()
        mgr = IntegrationManager(base_url=self.ui.current_base_url())
        result = mgr.check_models(config_path, target)
        self.ui.integration_status.setText(result.message)
        self.ui.integration_models_list.clear()
        if result.model_ids:
            self.ui.integration_models_list.addItems(result.model_ids)
        if not silent and not result.success:
            QMessageBox.warning(self.ui, "Integration Error", result.message)

    def _on_config_path_changed(self):
        self.save_settings()
        self.check_integration_models(silent=True)

    def add_model_to_integration(self):
        model_id = self.ui.current_model_id()
        if not model_id:
            QMessageBox.warning(self.ui, "Error", "Select a model first")
            return
        target = self.ui.current_config_target()
        config_path = self.ui.current_config_path()
        mgr = IntegrationManager(base_url=self.ui.current_base_url())
        result = mgr.add_model(config_path, target, model_id)
        self.ui.integration_status.setText(result.message)
        if result.success:
            self.ui.integration_models_list.clear()
            self.ui.integration_models_list.addItems(result.model_ids)
        else:
            QMessageBox.warning(self.ui, "Integration Error", result.message)

    def remove_model_from_integration(self):
        selected = self.ui.integration_models_list.currentItem()
        if not selected:
            QMessageBox.warning(self.ui, "Error", "Select a model to remove")
            return
        model_id = selected.text()
        target = self.ui.current_config_target()
        config_path = self.ui.current_config_path()
        mgr = IntegrationManager(base_url=self.ui.current_base_url())
        result = mgr.remove_model(config_path, target, model_id)
        self.ui.integration_status.setText(result.message)
        if result.success:
            self.ui.integration_models_list.clear()
            self.ui.integration_models_list.addItems(result.model_ids)
        else:
            QMessageBox.warning(self.ui, "Integration Error", result.message)

    def quit_app(self):
        self.save_settings()
        self.ui.save_ui_state()
        self.log_mgr.stop()
        self.server.terminate_all()
        if self.scanner and self.scanner.isRunning():
            self.scanner.requestInterruption()
            self.scanner.wait(1000)
        if self.autotune and self.autotune.isRunning():
            self.autotune.cancel()
            self.autotune.wait(2000)
        if hasattr(self, "tray"):
            self.tray.hide()
        QApplication.instance().quit()


def main():
    app = QApplication(sys.argv)
    gui = LlamaGUI()
    gui.ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
