"""LlamaServer GUI - точка входа."""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon
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
from src.core.mem_viz_parser import COMPONENT_META, MemoryData, fmt_mem, parse_line
from src.core.metrics_poller import MetricsPoller
from src.core.server_manager import ServerManager
from src.core.vram_estimator import full_vram_estimate
from src.services.autotune_manager import AutoTuneManager
from src.services.integration_manager import IntegrationManager
from src.services.hf_downloader import (
    HfModelDownloader,
    HfRepoScanner,
    delete_file_safely,
    find_partial_downloads,
    format_bytes,
    list_all_local_model_entries,
    list_local_repo_files,
    lmstudio_repo_dir,
    normalize_hf_repo_id,
    partial_download_info,
)
from src.services.threads import ModelScanner, LlamaCppUpdater
from src.ui.log_manager import LogManager
from src.ui.main_window import MainWindowUI
from src.ui.tooltips import build_ncmoe_tooltip, build_ctx_tooltip
from src.utils.subprocess_utils import no_console_kwargs


class LlamaGUI:
    def __init__(self):
        self.ui = MainWindowUI()
        self.config = ConfigManager()
        self.server = ServerManager()
        self.scanner = None
        self.updater = None
        self.hf_scanner = None
        self.hf_downloader = None
        self.hf_scan_result = None
        self.autotune = None
        self.autotune_plan = None
        self.autotune_results_dir = ""
        self.autotune_best_result = None
        self._autotune_best_applied = False
        self._autotune_running = False
        self._restart_pending = False
        self._pending_restart_launch = None
        self._restart_needed = False
        self._pending_server_verify = None
        self._last_server_launch = None
        self._mtp_draft_error_seen = False
        self._mtp_failure_reason = ""
        self._mtp_fallback_attempted = False
        self._mtp_auto_abort_requested = False
        self._memory_summary_logged = False

        self.log_mgr = LogManager(self.ui.logs)
        self.log_mgr.speed_updated.connect(self._on_log_speed_updated)
        self.ui.autoscroll_logs.toggled.connect(
            lambda checked: setattr(self.log_mgr, "autoscroll", checked)
        )
        self.metrics = MetricsPoller(poll_interval_ms=250)
        self.metrics.slot_metrics_updated.connect(self._on_slot_metrics_updated)
        self.metrics.server_metrics_updated.connect(self._on_server_metrics_updated)
        self._latest_token_total = 0
        self._latest_prompt_total = 0
        self._latest_predicted_total = 0
        self._token_baseline_total = 0
        self._saved_token_total = 0
        self._slot_prompt_total = 0
        self._slot_predicted_total = 0
        self._slot_token_seen = {}
        self._metrics_total_seen = False


        self.config.load()
        self.config.apply_to_ui(self.ui)
        self._normalize_llamacpp_path_ui()
        self.auto_detect_bench()
        self._connect_signals()
        self._setup_tray()
        QTimer.singleShot(250, self.auto_scan_models)
        QTimer.singleShot(350, self._refresh_hf_partial_status)

    def _connect_signals(self):
        u = self.ui
        u.start_btn.clicked.connect(self.start_server)
        u.reload_btn.clicked.connect(self.restart_server)
        u.stop_btn.clicked.connect(self.stop_work)
        u.force_stop_btn.clicked.connect(self.force_stop_server)
        u.tokens_reset_btn.clicked.connect(self.reset_task_tokens)
        u.test_btn.clicked.connect(self.run_benchmark)
        u.scan_btn.clicked.connect(self.scan_models)
        u.hf_scan_btn.clicked.connect(self.scan_hf_repo)
        u.hf_download_btn.clicked.connect(self.download_hf_selection)
        u.hf_pause_btn.clicked.connect(self.pause_hf_download)
        u.hf_cancel_btn.clicked.connect(self.cancel_hf_download)
        u.hf_refresh_local_btn.clicked.connect(self.refresh_hf_local_files)
        u.hf_delete_local_folder_btn.clicked.connect(self.delete_hf_local_folder)
        u.local_models_refresh_btn.clicked.connect(self.refresh_local_model_manager)
        u.local_models_delete_btn.clicked.connect(self.delete_selected_local_model)
        u.local_models_list.itemSelectionChanged.connect(self._update_local_model_delete_button)
        u.hf_files.itemSelectionChanged.connect(self._update_hf_download_button)
        u.model_combo.currentIndexChanged.connect(self.on_model_selected)
        u.ctx_size.valueChanged.connect(self.on_ctx_changed)
        for btn in getattr(u, "ctx_quick_buttons", []):
            btn.clicked.connect(
                lambda _checked=False, b=btn: self._set_context_size_from_button(
                    b.property("ctx_value")
                )
            )
        u.gpu_layers.valueChanged.connect(self._on_gpu_layers_changed)
        u.gpu_layers_all.stateChanged.connect(self._on_param_changed)
        u.cache_type_k.currentIndexChanged.connect(self._on_param_changed)
        u.cache_type_v.currentIndexChanged.connect(self._on_param_changed)
        u.flash_attn.stateChanged.connect(self._on_param_changed)
        u.parallel_slots.valueChanged.connect(self._on_param_changed)
        u.kv_unified.stateChanged.connect(self._on_param_changed)
        u.speculative_mtp.stateChanged.connect(self._on_param_changed)
        u.spec_draft_n_max.valueChanged.connect(self._on_param_changed)
        u.spec_draft_gpu_layers.textChanged.connect(self._on_param_changed)
        u.spec_draft_model_path.textChanged.connect(self._on_param_changed)
        u.cuda_device.textChanged.connect(self._on_param_changed)
        u.spec_draft_device.textChanged.connect(self._on_param_changed)
        u.split_mode.currentIndexChanged.connect(self._on_param_changed)
        u.main_gpu.valueChanged.connect(self._on_param_changed)
        u.cpu_moe_layers.valueChanged.connect(self._on_param_changed)
        u.gpu_auto.stateChanged.connect(self._on_param_changed)
        u.batch_size.valueChanged.connect(self._on_param_changed)
        u.ubatch_size.valueChanged.connect(self._on_param_changed)
        u.threads.valueChanged.connect(self._on_param_changed)
        u.threads_batch.valueChanged.connect(self._on_param_changed)
        u.fit_off.stateChanged.connect(self._on_param_changed)
        u.reasoning_mode.currentIndexChanged.connect(self._on_param_changed)
        u.host.textChanged.connect(self._on_param_changed)
        u.port.valueChanged.connect(self._on_param_changed)
        u.ctx_checkpoints.valueChanged.connect(self._on_param_changed)
        u.cache_ram.valueChanged.connect(self._on_param_changed)
        u.temperature.valueChanged.connect(self._on_param_changed)
        u.repeat_penalty.valueChanged.connect(self._on_param_changed)
        u.use_mmap.stateChanged.connect(self._on_param_changed)
        u.use_mlock.stateChanged.connect(self._on_param_changed)
        u.verbose.stateChanged.connect(self._on_param_changed)
        u.log_timestamps.stateChanged.connect(self._on_param_changed)
        u.cuda_visible_devices.textChanged.connect(self._on_param_changed)
        u.cuda_module_loading.textChanged.connect(self._on_param_changed)
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
        u.cuda_version_combo.currentIndexChanged.connect(
            self._on_llamacpp_version_changed
        )
        u.integration_check_btn.clicked.connect(self.check_integration_models)
        u.integration_add_btn.clicked.connect(self.add_model_to_integration)
        u.integration_remove_btn.clicked.connect(self.remove_model_from_integration)
        u.integration_target.currentIndexChanged.connect(self.check_integration_models)
        u.opencode_config_path.editingFinished.connect(self._on_config_path_changed)
        u.pi_config_path.editingFinished.connect(self._on_config_path_changed)
        u.exe_path.textChanged.connect(self.auto_detect_bench)
        u.exe_path.textChanged.connect(self.update_cli_preview)
        u.copy_model_btn.clicked.connect(self._copy_model_path)
        u._browse_exe_clicked = self.browse_exe
        u._browse_bench_clicked = self.browse_bench
        u._browse_model_dir_clicked = self.browse_model_dir
        u._browse_opencode_clicked = self.browse_opencode_config
        u._browse_pi_clicked = self.browse_pi_config
        u._browse_mtp_draft_clicked = self.browse_mtp_draft_model
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

    def _fmt_counter(self, value) -> str:
        return f"{max(int(value or 0), 0):,}"

    def _server_metrics_url(self) -> str:
        host = str(self.ui.host.text() or "127.0.0.1").strip() or "127.0.0.1"
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{self.ui.port.value()}"

    def _on_log_speed_updated(self, text: str):
        if not getattr(self.metrics, "_is_running", False):
            self.ui.speed_label.setText(text)

    def _start_metrics_polling(self):
        self.metrics.set_url(self._server_metrics_url())
        self._slot_prompt_total = 0
        self._slot_predicted_total = 0
        self._slot_token_seen = {}
        self._metrics_total_seen = False
        self.metrics.start()
        self.ui.speed_label.setText("Speed: waiting for /slots...")
        self.ui.request_tokens_label.setText("Request: -")

    def _stop_metrics_polling(self):
        self.metrics.stop()
        self.ui.speed_label.setText("Speed: -")
        self.ui.request_tokens_label.setText("Request: -")

    def _refresh_token_label(self):
        task_total = max(self._latest_token_total - self._token_baseline_total, 0)
        self.ui.tokens_label.setText(
            "Tokens: "
            f"total {self._fmt_counter(self._latest_token_total)} | "
            f"task {self._fmt_counter(task_total)} | "
            f"prompt {self._fmt_counter(self._latest_prompt_total)} | "
            f"generated {self._fmt_counter(self._latest_predicted_total)}"
        )

    def _slot_speed(self, slot, token_attr: str, ms_attr: str, speed_attr: str) -> float:
        speed = float(getattr(slot, speed_attr, 0.0) or 0.0)
        if speed > 0:
            return speed
        token_count = int(getattr(slot, token_attr, 0) or 0)
        elapsed_ms = float(getattr(slot, ms_attr, 0.0) or 0.0)
        if token_count > 0 and elapsed_ms > 0:
            return token_count / elapsed_ms * 1000.0
        return 0.0

    def _accumulate_slot_tokens(self, slots):
        if self._metrics_total_seen:
            return
        for slot in slots:
            slot_id = int(getattr(slot, "id", 0) or 0)
            prompt_tokens = int(getattr(slot, "prompt_n", 0) or 0)
            predicted_tokens = int(getattr(slot, "predicted_n", 0) or 0)
            previous_prompt, previous_predicted = self._slot_token_seen.get(slot_id, (0, 0))
            prompt_delta = prompt_tokens - previous_prompt if prompt_tokens >= previous_prompt else prompt_tokens
            predicted_delta = (
                predicted_tokens - previous_predicted
                if predicted_tokens >= previous_predicted
                else predicted_tokens
            )
            self._slot_prompt_total += max(prompt_delta, 0)
            self._slot_predicted_total += max(predicted_delta, 0)
            self._slot_token_seen[slot_id] = (prompt_tokens, predicted_tokens)
        self._latest_prompt_total = self._slot_prompt_total
        self._latest_predicted_total = self._slot_predicted_total
        self._latest_token_total = self._latest_prompt_total + self._latest_predicted_total
        self._refresh_token_label()

    def _on_slot_metrics_updated(self, slots):
        if not slots:
            return
        active = [slot for slot in slots if getattr(slot, "is_processing", False)]
        visible = active or [
            slot
            for slot in slots
            if getattr(slot, "prompt_per_second", 0.0)
            or getattr(slot, "predicted_per_second", 0.0)
            or getattr(slot, "prompt_n", 0)
            or getattr(slot, "predicted_n", 0)
        ]
        if not visible:
            self.ui.speed_label.setText("Speed: -")
            self.ui.request_tokens_label.setText("Request: -")
            return

        prompt_speed = sum(
            self._slot_speed(slot, "prompt_n", "prompt_ms", "prompt_per_second")
            for slot in visible
        )
        predicted_speed = sum(
            self._slot_speed(slot, "predicted_n", "predicted_ms", "predicted_per_second")
            for slot in visible
        )
        prompt_tokens = sum(int(getattr(slot, "prompt_n", 0) or 0) for slot in visible)
        predicted_tokens = sum(int(getattr(slot, "predicted_n", 0) or 0) for slot in visible)

        parts = []
        if prompt_speed > 0:
            parts.append(f"PP {prompt_speed:.2f} tok/s")
        if predicted_speed > 0:
            parts.append(f"TG {predicted_speed:.2f} tok/s")
        self.ui.speed_label.setText("Speed: " + (" | ".join(parts) if parts else "-"))
        self.ui.request_tokens_label.setText(
            f"Request: prompt {self._fmt_counter(prompt_tokens)} | generated {self._fmt_counter(predicted_tokens)}"
        )
        self._accumulate_slot_tokens(slots)

    def _on_server_metrics_updated(self, metrics):
        prompt_total = int(getattr(metrics, "prompt_tokens_total", 0) or 0)
        predicted_total = int(getattr(metrics, "tokens_predicted_total", 0) or 0)
        total = prompt_total + predicted_total
        if total <= 0:
            return
        self._metrics_total_seen = True
        if total < self._token_baseline_total:
            self._token_baseline_total = 0
        self._latest_prompt_total = prompt_total
        self._latest_predicted_total = predicted_total
        self._latest_token_total = total
        self._refresh_token_label()

    def reset_task_tokens(self):
        task_total = max(self._latest_token_total - self._token_baseline_total, 0)
        self._saved_token_total += task_total
        self._token_baseline_total = self._latest_token_total
        self.ui.tokens_saved_label.setText(
            f"Saved: last {self._fmt_counter(task_total)} | total {self._fmt_counter(self._saved_token_total)}"
        )
        self._refresh_token_label()
        self.log_mgr.append(f"Token counter reset: saved {task_total:,} tokens")
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
        if not self.ui.windowIcon().isNull():
            self.tray.setIcon(self.ui.windowIcon())
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
        self.auto_detect_bench()
        self.config.read_from_ui(self.ui)
        self.config.settings.model_cache = self.ui.models
        self.config.save()

    def _selected_cuda_version(self) -> str:
        return str(self.ui.cuda_version_combo.currentData() or "12")

    def _llamacpp_cuda_major(self, path: Path) -> str:
        match = re.search(r"cuda-(\d+)(?:[.-]|$)", path.name.lower())
        return match.group(1) if match else ""

    def _llamacpp_version_key(self, path: Path):
        match = re.search(r"cuda-(\d+)\.(\d+)", path.name.lower())
        if not match:
            return (0, 0)
        return (int(match.group(1)), int(match.group(2)))

    def _matching_llamacpp_dirs(self, base: Path, cuda_version: str):
        if not base.is_dir():
            return []
        pattern = re.compile(
            rf"cuda-{re.escape(str(cuda_version))}(?:\.|-|$)", re.IGNORECASE
        )
        try:
            dirs = [
                p
                for p in base.iterdir()
                if p.is_dir() and pattern.search(p.name) and "x64" in p.name.lower()
            ]
        except OSError:
            return []
        return sorted(dirs, key=self._llamacpp_version_key, reverse=True)

    def _normalize_llamacpp_path_ui(self):
        """Миграция старого пути к exe в новый базовый путь, если формат узнаваем."""
        text = self.ui.exe_path.text().strip().strip('"')
        if not text:
            return
        path = Path(text)
        if path.name.lower() != "llama-server.exe":
            return
        parent = path.parent
        if self._llamacpp_cuda_major(parent) and parent.parent:
            self.ui.exe_path.setText(str(parent.parent))

    def _resolve_llamacpp_executable(self, kind: str = "server") -> str:
        exe_name = f"llama-{kind}.exe"
        raw = self.ui.exe_path.text().strip().strip('"')
        if not raw:
            return ""

        selected_cuda = self._selected_cuda_version()
        path = Path(raw)
        candidate_dirs = []

        if path.suffix.lower() == ".exe":
            parent = path.parent
            parent_cuda = self._llamacpp_cuda_major(parent)
            if parent_cuda and parent_cuda != selected_cuda:
                candidate_dirs.extend(
                    self._matching_llamacpp_dirs(parent.parent, selected_cuda)
                )
            candidate_dirs.append(parent)
        else:
            path_cuda = self._llamacpp_cuda_major(path)
            if path_cuda:
                if path_cuda == selected_cuda:
                    candidate_dirs.append(path)
                candidate_dirs.extend(
                    self._matching_llamacpp_dirs(path.parent, selected_cuda)
                )
                candidate_dirs.append(path)
            else:
                candidate_dirs.extend(self._matching_llamacpp_dirs(path, selected_cuda))
                candidate_dirs.append(path)

        seen = set()
        for directory in candidate_dirs:
            key = os.path.normcase(str(directory))
            if key in seen:
                continue
            seen.add(key)
            exe = directory / exe_name
            if exe.exists():
                return str(exe)
        return ""

    def _on_llamacpp_version_changed(self):
        self.auto_detect_bench()
        self.update_cli_preview()
        self.save_settings()
        self._mark_restart_needed()

    def _copy_model_path(self):
        path = self._current_model_path()
        if path:
            QApplication.clipboard().setText(path)
            self.log_mgr.append(f"Model path copied: {path}")

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

        autotune_params = None
        if (
            self._autotune_best_applied
            and self.autotune_plan
            and self.autotune_best_result
            and os.path.normcase(os.path.abspath(self.autotune_plan.model_path))
            == os.path.normcase(os.path.abspath(model_path))
            and int(self.autotune_plan.ctx_size) == int(ctx)
        ):
            autotune_params = self._best_autotune_params()

        try:
            self.config.save_perf_preset(
                model_path,
                ctx,
                self.ui,
                autotune_params=autotune_params,
            )
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
        bench = self._resolve_llamacpp_executable("bench")
        if bench:
            self.ui.bench_path.setText(bench)
        return bench

    def browse_exe(self):
        d = QFileDialog.getExistingDirectory(
            self.ui,
            "Select llama.cpp base folder",
            self.ui.exe_path.text().strip() or "",
        )
        if d:
            self.ui.exe_path.setText(d)
            self.auto_detect_bench()
            self.save_settings()

    def browse_bench(self):
        # Kept for backward compatibility; bench is normally auto-detected from base folder.
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

    def browse_mtp_draft_model(self):
        start_dir = self.ui.model_dir.text().strip() or ""
        current = self.ui.spec_draft_model_path.text().strip()
        if current:
            start_dir = str(Path(current).parent)
        f, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Select separate MTP/draft GGUF",
            start_dir,
            "GGUF (*.gguf);;All files (*.*)",
        )
        if f:
            self.ui.spec_draft_model_path.setText(f)
            self.ui.speculative_mtp.setChecked(True)
            self.save_settings()

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

    def _update_local_model_delete_button(self):
        self.ui.local_models_delete_btn.setEnabled(
            bool(self.ui.local_models_list.selectedItems())
        )

    def refresh_local_model_manager(self, silent=False):
        self.ui.local_models_list.clear()
        model_dir = self.ui.model_dir.text().strip()
        if not model_dir or not os.path.exists(model_dir):
            self.ui.local_models_delete_btn.setEnabled(False)
            if not silent:
                self.ui.local_models_status.setText("Specify existing Models folder")
            return

        info = list_all_local_model_entries(Path(model_dir))
        entries = info.get("entries") or []
        for entry in entries:
            examples = ", ".join(entry.get("examples") or [])
            suffix = f" | {examples}" if examples else ""
            item_text = (
                f"{entry.get('relative')}  |  {entry.get('type')}  |  "
                f"GGUF: {entry.get('gguf_count')}  |  {entry.get('size_text')}{suffix}"
            )
            self.ui.local_models_list.addItem(item_text)
            item = self.ui.local_models_list.item(self.ui.local_models_list.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(str(entry.get("path") or ""))

        self.ui.local_models_delete_btn.setEnabled(False)
        if entries:
            self.ui.local_models_status.setText(
                f"Local models: {len(entries)}, total {info.get('total_size_text')} | root: {info.get('root')}"
            )
        elif not silent:
            self.ui.local_models_status.setText(f"No local GGUF models found in {model_dir}")

    def _selected_local_model_entry(self):
        selected = self.ui.local_models_list.selectedItems()
        if not selected:
            return None
        return selected[0].data(Qt.ItemDataRole.UserRole)

    def _path_is_inside(self, path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except (OSError, ValueError):
            return False

    def _current_model_uses_path(self, target: Path) -> bool:
        current = self._current_model_path()
        if not current:
            return False
        current_path = Path(current)
        try:
            if target.is_dir():
                current_path.resolve().relative_to(target.resolve())
                return True
            return current_path.resolve() == target.resolve()
        except (OSError, ValueError):
            return False

    def delete_selected_local_model(self):
        entry = self._selected_local_model_entry()
        if not entry:
            return
        if self.hf_downloader and self.hf_downloader.isRunning():
            QMessageBox.warning(
                self.ui,
                "Delete local model",
                "Stop the Hugging Face download before deleting local models.",
            )
            return

        model_dir = self.ui.model_dir.text().strip()
        target = Path(str(entry.get("path") or ""))
        base = Path(model_dir) if model_dir else Path()
        if not target.exists() or not model_dir or not self._path_is_inside(target, base):
            QMessageBox.warning(self.ui, "Delete local model", "Selected path is invalid or outside Models folder.")
            self.refresh_local_model_manager(silent=True)
            return
        if target.resolve() == base.resolve():
            QMessageBox.warning(self.ui, "Delete local model", "Refusing to delete the Models root folder.")
            return
        if self.server.is_server_running() and self._current_model_uses_path(target):
            QMessageBox.warning(
                self.ui,
                "Delete local model",
                "This model is currently loaded. Stop the server first to release RAM/VRAM, then delete it.",
            )
            return

        delete_kind = "folder" if target.is_dir() else "file"
        reply = QMessageBox.question(
            self.ui,
            "Delete local model",
            f"Delete selected local model {delete_kind}?\n\n"
            f"{target}\n\n"
            "This cannot be undone. For folders, all GGUF/mmproj/.part files inside are removed.",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except OSError as exc:
            QMessageBox.critical(self.ui, "Delete failed", str(exc))
            return

        self.log_mgr.append(f"Local model {delete_kind} deleted: {target}", "warn")
        self.ui.local_models_status.setText(f"Deleted: {target}")
        self.refresh_local_model_manager(silent=True)
        self.refresh_hf_local_files(silent=True)
        self.scan_models(silent=True)

    def scan_hf_repo(self):
        if self.hf_scanner and self.hf_scanner.isRunning():
            self.hf_scanner.requestInterruption()
            self.ui.hf_status.setText("Отмена запроса Hugging Face...")
            return

        repo = self.ui.hf_repo.text().strip()
        if not repo:
            QMessageBox.warning(self.ui, "Hugging Face", "Вставьте repo id или URL модели")
            return

        self.save_settings()
        self.ui.hf_files.clear()
        self.ui.hf_progress.setValue(0)
        self.ui.hf_download_btn.setEnabled(False)
        self.ui.hf_progress.setVisible(True)
        self.ui.hf_progress.setRange(0, 0)
        self.ui.hf_status.setText("Сканирование Hugging Face...")
        self.ui.hf_scan_btn.setText("Cancel")

        self.hf_scanner = HfRepoScanner(repo, self.ui.hf_quant_filter.text().strip())
        self.hf_scanner.progress.connect(self.ui.hf_status.setText)
        self.hf_scanner.completed.connect(self._on_hf_scan_completed)
        self.hf_scanner.error.connect(self._on_hf_scan_error)
        self.hf_scanner.finished.connect(self._on_hf_scan_finished)
        self.hf_scanner.start()

    def _on_hf_scan_finished(self):
        self.ui.hf_scan_btn.setText("Scan HF")
        self.ui.hf_progress.setVisible(False)
        self.ui.hf_progress.setRange(0, 100)

    def _on_hf_scan_error(self, message):
        self.hf_scan_result = None
        self.ui.hf_status.setText(message)
        self.log_mgr.append(f"Hugging Face scan failed: {message}", "error")

    def _on_hf_scan_completed(self, result):
        self.hf_scan_result = result
        self.ui.hf_files.clear()
        files = result.get("files") or []
        projectors = result.get("projectors") or []
        partial_count = 0
        for file_info in files:
            partial = self._hf_partial_info(file_info)
            if partial:
                partial_count += 1
            self.ui.hf_files.addItem(self._hf_file_display(file_info))
            item = self.ui.hf_files.item(self.ui.hf_files.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, file_info)
            if partial:
                item.setToolTip(f"Partial file: {partial.get('partial_path')}")

        if files:
            self.ui.hf_files.setCurrentRow(0)

        target_text = ""
        model_dir = self.ui.model_dir.text().strip()
        if model_dir:
            target_text = f" → {lmstudio_repo_dir(Path(model_dir), result.get('repo_id', ''))}"
        projector = self._select_hf_projector()
        projector_text = f", vision: {projector.get('name')}" if projector else ""
        total_size = sum(int(f.get("size") or 0) for f in files)
        total_text = f", shown size: {format_bytes(total_size)}" if total_size else ""
        partial_text = f", partial/resumable: {partial_count}" if partial_count else ""
        self.ui.hf_status.setText(
            f"Найдено GGUF: {len(files)} из {len(result.get('all_files') or [])}"
            f"{total_text}, mmproj: {len(projectors)}{projector_text}"
            f"{partial_text}{target_text}"
        )
        self._update_hf_download_button()
        self.refresh_hf_local_files(silent=True)
        self.save_settings()

    def _hf_file_display(self, file_info):
        name = file_info.get("name") or file_info.get("rfilename") or ""
        parts = [str(name)]
        quant = str(file_info.get("quant") or "").strip()
        if quant:
            parts.append(quant)
        size = int(file_info.get("size") or 0)
        parts.append(format_bytes(size) if size else "size unknown")
        partial = self._hf_partial_info(file_info)
        if partial:
            parts.append(
                f"partial {partial.get('partial_size_text')} / "
                f"{partial.get('expected_size_text')}"
            )
        return "  |  ".join(parts)

    def _hf_partial_info(self, file_info):
        if not self.hf_scan_result:
            return {}
        model_dir = self.ui.model_dir.text().strip()
        repo_id = self.hf_scan_result.get("repo_id") or ""
        filename = file_info.get("rfilename") or file_info.get("name") or ""
        if not model_dir or not repo_id or not filename:
            return {}
        return partial_download_info(
            Path(model_dir), repo_id, filename, int(file_info.get("size") or 0)
        )

    def _refresh_hf_partial_status(self):
        model_dir = self.ui.model_dir.text().strip()
        repo_text = self.ui.hf_repo.text().strip()
        if not model_dir or not repo_text:
            return
        try:
            repo_id = normalize_hf_repo_id(repo_text)
            partials = find_partial_downloads(Path(model_dir), repo_id)
        except Exception:
            return
        if not partials:
            return
        total = sum(int(p.get("partial_size") or 0) for p in partials)
        self.ui.hf_status.setText(
            f"Найдены незавершённые HF загрузки: {len(partials)}, "
            f"сохранено {format_bytes(total)}. Нажмите Scan HF, затем Download selected — "
            "загрузка продолжится с .part."
        )
        self.refresh_hf_local_files(silent=True)

    def _current_hf_repo_id(self):
        if self.hf_scan_result and self.hf_scan_result.get("repo_id"):
            return self.hf_scan_result.get("repo_id")
        repo_text = self.ui.hf_repo.text().strip()
        if not repo_text:
            return ""
        try:
            return normalize_hf_repo_id(repo_text)
        except Exception:
            return ""

    def refresh_hf_local_files(self, silent=False):
        self.ui.hf_local_files.clear()
        model_dir = self.ui.model_dir.text().strip()
        repo_id = self._current_hf_repo_id()
        if not model_dir or not repo_id:
            self.ui.hf_delete_local_folder_btn.setEnabled(False)
            if not silent:
                self.ui.hf_status.setText("Local files: specify Models folder and HF repo")
            return

        info = list_local_repo_files(Path(model_dir), repo_id)
        files = info.get("files") or []
        for file_info in files:
            marker = "partial" if file_info.get("is_partial") else "local"
            self.ui.hf_local_files.addItem(
                f"{file_info.get('relative')}  |  {marker}  |  {file_info.get('size_text')}"
            )
            self.ui.hf_local_files.item(self.ui.hf_local_files.count() - 1).setToolTip(
                str(file_info.get("path") or "")
            )
        self.ui.hf_delete_local_folder_btn.setEnabled(bool(info.get("exists")))
        if files:
            self.ui.hf_status.setText(
                f"Local folder: {info.get('root')} | files: {len(files)}, total {info.get('total_size_text')}"
            )
        elif info.get("exists"):
            self.ui.hf_status.setText(f"Local folder exists but is empty: {info.get('root')}")
        elif not silent:
            self.ui.hf_status.setText(f"Local folder not found: {info.get('root')}")

    def delete_hf_local_folder(self):
        if self.hf_downloader and self.hf_downloader.isRunning():
            QMessageBox.warning(self.ui, "Hugging Face", "Stop the download before deleting local files")
            return
        model_dir = self.ui.model_dir.text().strip()
        repo_id = self._current_hf_repo_id()
        if not model_dir or not repo_id:
            return
        target_root = lmstudio_repo_dir(Path(model_dir), repo_id)
        if not target_root.exists():
            self.refresh_hf_local_files(silent=True)
            return
        reply = QMessageBox.question(
            self.ui,
            "Delete local model folder",
            "Удалить всю локальную папку модели, включая main GGUF, vision/mmproj и .part?\n\n"
            f"{target_root}",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            shutil.rmtree(target_root)
        except OSError as exc:
            QMessageBox.critical(self.ui, "Delete failed", str(exc))
            return
        self.ui.hf_status.setText(f"Local model folder deleted: {target_root}")
        self.refresh_hf_local_files(silent=True)
        self.refresh_local_model_manager(silent=True)
        self.scan_models(silent=True)

    def _selected_hf_file_info(self):
        selected = self.ui.hf_files.selectedItems()
        if not selected:
            return None
        return selected[0].data(Qt.ItemDataRole.UserRole)

    def _selected_hf_partial_info(self):
        file_info = self._selected_hf_file_info()
        return self._hf_partial_info(file_info) if file_info else {}

    def _update_hf_download_button(self):
        is_running = bool(self.hf_downloader and self.hf_downloader.isRunning())
        has_selection = bool(self.ui.hf_files.selectedItems())
        has_partial = bool(self._selected_hf_partial_info())
        self.ui.hf_download_btn.setText(
            "Resume selected" if has_partial and not is_running else "Download selected"
        )
        self.ui.hf_download_btn.setEnabled(has_selection and not is_running)
        self.ui.hf_pause_btn.setEnabled(is_running)
        self.ui.hf_cancel_btn.setEnabled(is_running or (has_partial and not is_running))
        self.ui.hf_cancel_btn.setToolTip(
            "Cancel running download and delete .part"
            if is_running
            else "Delete saved .part for selected file"
            if has_partial
            else "No partial download selected"
        )

    def _set_hf_download_controls_locked(self, locked):
        for widget in (
            self.ui.hf_repo,
            self.ui.hf_quant_filter,
            self.ui.hf_scan_btn,
            self.ui.hf_include_mmproj,
            self.ui.hf_files,
        ):
            widget.setEnabled(not locked)
        if locked:
            self.ui.hf_download_btn.setText("Download selected")
            self.ui.hf_download_btn.setEnabled(False)
            self.ui.hf_pause_btn.setEnabled(True)
            self.ui.hf_cancel_btn.setEnabled(True)
            self.ui.hf_cancel_btn.setToolTip("Cancel running download and delete .part")
        else:
            self._update_hf_download_button()

    def _select_hf_projector(self):
        if not self.hf_scan_result:
            return None
        projectors = list(self.hf_scan_result.get("projectors") or [])
        if not projectors:
            return None
        filter_text = self.ui.hf_quant_filter.text().upper()
        preferred = []
        for key in ("BF16", "F16", "F32"):
            if key in filter_text:
                preferred.append(key)
        preferred.extend(["BF16", "F16", "F32"])
        for key in preferred:
            for item in projectors:
                if key in str(item.get("name") or "").upper():
                    return item
        projectors.sort(key=lambda item: (int(item.get("size") or 0), str(item.get("name") or "").lower()))
        return projectors[0]

    def download_hf_selection(self):
        if self.hf_downloader and self.hf_downloader.isRunning():
            return

        if not self.hf_scan_result:
            QMessageBox.warning(self.ui, "Hugging Face", "Сначала просканируйте репозиторий")
            return
        selected = self.ui.hf_files.selectedItems()
        if not selected:
            QMessageBox.warning(self.ui, "Hugging Face", "Выберите GGUF файл для скачивания")
            return
        selected_partial = self._selected_hf_partial_info()
        model_dir = self.ui.model_dir.text().strip()
        if not model_dir:
            QMessageBox.warning(self.ui, "Hugging Face", "Укажите базовую папку Models")
            return

        main_file = selected[0].data(Qt.ItemDataRole.UserRole)
        files = [main_file]
        if self.ui.hf_include_mmproj.isChecked():
            projector = self._select_hf_projector()
            if projector and projector.get("rfilename") != main_file.get("rfilename"):
                files.append(projector)

        repo_id = self.hf_scan_result.get("repo_id") or ""
        target_root = lmstudio_repo_dir(Path(model_dir), repo_id)
        total_size = sum(int(f.get("size") or 0) for f in files)
        names = "\n".join(f"• {self._hf_file_display(f)}" for f in files)
        size_line = f"\nTotal: {format_bytes(total_size)}" if total_size else ""
        action_title = "Resume GGUF" if selected_partial else "Download GGUF"
        action_text = "Продолжить загрузку" if selected_partial else "Скачать"
        resume_line = (
            f"\nResume from: {selected_partial.get('partial_size_text')}"
            if selected_partial
            else ""
        )
        reply = QMessageBox.question(
            self.ui,
            action_title,
            f"{action_text} в LM Studio-compatible папку:\n{target_root}\n\n"
            f"{names}{size_line}{resume_line}",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.ui.hf_download_btn.setText("Download selected")
        self._set_hf_download_controls_locked(True)
        self.ui.hf_progress.setRange(0, 100)
        self.ui.hf_progress.setValue(0)
        self.ui.hf_progress.setVisible(True)
        self.ui.hf_status.setText(
            f"Начало скачивания: {len(files)} файл(а), total {format_bytes(total_size)}"
        )

        self.hf_downloader = HfModelDownloader(repo_id, files, model_dir)
        self.hf_downloader.progress.connect(self.ui.hf_status.setText)
        self.hf_downloader.percent.connect(self.ui.hf_progress.setValue)
        self.hf_downloader.completed.connect(self._on_hf_download_completed)
        self.hf_downloader.finished.connect(self._on_hf_download_finished)
        self.hf_downloader.start()

    def pause_hf_download(self):
        if self.hf_downloader and self.hf_downloader.isRunning():
            self.hf_downloader.pause()
            self.ui.hf_pause_btn.setEnabled(False)
            self.ui.hf_cancel_btn.setEnabled(False)
            self.ui.hf_status.setText("Пауза: сохраняю .part для докачки...")

    def cancel_hf_download(self):
        if self.hf_downloader and self.hf_downloader.isRunning():
            reply = QMessageBox.question(
                self.ui,
                "Cancel download",
                "Прервать текущую загрузку и удалить частичный .part файл?\n\n"
                "Если хотите продолжить позже — нажмите Pause вместо Cancel.",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.hf_downloader.cancel_and_delete()
            self.ui.hf_pause_btn.setEnabled(False)
            self.ui.hf_cancel_btn.setEnabled(False)
            self.ui.hf_status.setText("Отмена: удаляю частичный .part файл...")
            return

        partial = self._selected_hf_partial_info()
        if not partial:
            return
        reply = QMessageBox.question(
            self.ui,
            "Cancel partial download",
            "Удалить сохранённый .part и начать этот файл заново при следующей загрузке?\n\n"
            f"{partial.get('partial_path')}\n"
            f"Saved: {partial.get('partial_size_text')}",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        delete_file_safely(Path(partial.get("partial_path") or ""))
        item = self.ui.hf_files.currentItem()
        file_info = self._selected_hf_file_info()
        if item and file_info:
            item.setText(self._hf_file_display(file_info))
            item.setToolTip("")
        self.ui.hf_status.setText("Частичный .part удалён. Следующая загрузка начнётся заново.")
        self.refresh_hf_local_files(silent=True)
        self._update_hf_download_button()

    def _on_hf_download_finished(self):
        item = self.ui.hf_files.currentItem()
        file_info = self._selected_hf_file_info()
        if item and file_info:
            partial = self._hf_partial_info(file_info)
            item.setText(self._hf_file_display(file_info))
            item.setToolTip(f"Partial file: {partial.get('partial_path')}" if partial else "")
        self._set_hf_download_controls_locked(False)
        self.refresh_hf_local_files(silent=True)
        self.refresh_local_model_manager(silent=True)
        self._update_hf_download_button()

    def _on_hf_download_completed(self, ok, message):
        if ok:
            self.ui.hf_progress.setValue(100)
            QTimer.singleShot(1500, self._reset_hf_progress_after_complete)
        self.ui.hf_status.setText(message)
        self.log_mgr.append(message, "info" if ok else "error")
        self.refresh_hf_local_files(silent=True)
        self.refresh_local_model_manager(silent=True)
        if ok:
            self.scan_models(silent=True)

    def _reset_hf_progress_after_complete(self):
        if self.hf_downloader and self.hf_downloader.isRunning():
            return
        self.ui.hf_progress.setValue(0)
        self.ui.hf_progress.setVisible(False)

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
        self.refresh_local_model_manager(silent=True)

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
        info.setdefault("path", path)
        info.setdefault("_model_path", path)
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

        tags = []
        if info.get("is_qat"):
            tags.append("QAT")
        if info.get("mtp_capable"):
            tags.append("MTP")
        tag_text = f" | {'/'.join(tags)}" if tags else ""
        parts = [f"Architecture: {arch} | {quant} | {size:.2f} GiB{tag_text}"]

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
            mmproj_name = Path(info["mmproj_path"]).name
            parts.append(f"mmproj: {mmproj_name}")

        if info.get("mtp_draft_path"):
            parts.append(f"MTP draft: {Path(info['mtp_draft_path']).name}")

        if info.get("metadata_error"):
            parts.append(f"Warning: {info['metadata_error']}")

        self.ui.model_info.setText("\n".join(parts))
        model_name = Path(path).name if path else ""
        self.ui.model_id_label.setText(f"{model_name}\n{path or ''}")

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
        self._sync_mtp_controls_for_model(info)

        self.update_cli_preview()
        self._mark_restart_needed()

    def _sync_mtp_controls_for_model(self, info):
        is_mtp = self._is_mtp_model_info(info)
        draft_path = str(info.get("mtp_draft_path") or "").strip()
        for widget in (
            self.ui.spec_draft_n_max,
            self.ui.spec_draft_gpu_layers,
            self.ui.spec_draft_model_path,
            self.ui.spec_draft_model_btn,
            self.ui.spec_draft_device,
        ):
            widget.setEnabled(is_mtp)
        self.ui.speculative_mtp.setEnabled(is_mtp)
        if is_mtp:
            current_draft = self.ui.spec_draft_model_path.text().strip()
            if self._uses_embedded_mtp_mode(info):
                if current_draft == draft_path or "mtp-gemma" in current_draft.lower():
                    self.ui.spec_draft_model_path.clear()
                self.ui.spec_draft_model_path.setPlaceholderText(
                    "Auto: embedded/package MTP mode, no --model-draft"
                )
            elif draft_path and (not current_draft or not os.path.exists(current_draft)):
                self.ui.spec_draft_model_path.setText(draft_path)
            self.ui.speculative_mtp.setToolTip(
                "Enable llama.cpp MTP speculative decoding automatically. Gemma 4 regular GGUF uses package/embedded mode; QAT/manual modes can use a separate draft GGUF."
            )
            return

        if self.ui.speculative_mtp.isChecked():
            self.ui.speculative_mtp.setChecked(False)
            self.log_mgr.append(
                "MTP speculative disabled: selected model does not contain MTP layers",
                "warn",
            )
        self.ui.spec_draft_model_path.clear()
        self.ui.spec_draft_model_path.setPlaceholderText(
            "Auto-detected, or browse for separate MTP GGUF"
        )
        self.ui.speculative_mtp.setToolTip(
            "Disabled: selected GGUF/package has no detected MTP draft support. Use Extra params only if you know this model supports it."
        )

    def _refresh_tooltips(self, info):
        """Обновление tooltip для ncmoe и ctx при смене модели."""
        expert_count = info.get("expert_count", 0)

        if expert_count:
            gpu_layers_val = (
                999
                if self.ui.gpu_auto.isChecked() or self.ui.gpu_layers_all.isChecked()
                else self.ui.gpu_layers.value()
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

    def _is_mtp_model_info(self, info):
        if info.get("mtp_capable") or info.get("mtp_draft_path"):
            return True
        text = " ".join(
            str(info.get(k) or "")
            for k in ("path", "name", "display", "architecture", "_model_path")
        ).lower()
        return "mtp" in text

    def _uses_embedded_mtp_mode(self, info):
        """True when llama.cpp should use --spec-type draft-mtp without --model-draft."""
        arch = str(info.get("architecture") or "").lower()
        name_text = " ".join(
            str(info.get(k) or "") for k in ("path", "name", "display", "_model_path")
        ).lower()
        return (
            arch.startswith("gemma4")
            and bool(info.get("mtp_capable"))
            and not info.get("is_qat")
            and "qat" not in name_text
        )

    def _auto_mtp_supported(self, info):
        """Auto-enable MTP only for known-safe embedded mode or a local draft.

        Для LM Studio layout ищем draft только внутри папки выбранной модели.
        Соседние модели не используются, чтобы не подставить несовместимый
        Qwen/Gemma draft и не получить GGML_ASSERT по ширине embedding.
        Если ручной draft всё же несовместим, приложение сделает fallback без MTP.
        """
        return bool(self._uses_embedded_mtp_mode(info) or info.get("mtp_draft_path"))

    def _auto_mtp_draft_path(self, info):
        if not self._auto_mtp_supported(info) or self._uses_embedded_mtp_mode(info):
            return ""
        return str(info.get("mtp_draft_path") or "").strip()

    def _apply_mtp_recommended_params(self, info):
        block_count = int(info.get("block_count") or 0)
        self.ui.gpu_auto.setChecked(False)
        self.ui.gpu_layers_all.setChecked(True)
        if block_count > 0:
            self.ui.gpu_layers.setValue(block_count)
        self.ui.cache_type_k.setCurrentText("q8_0")
        self.ui.cache_type_v.setCurrentText("q8_0")
        self.ui.batch_size.setValue(512)
        self.ui.ubatch_size.setValue(256)
        self.ui.parallel_slots.setValue(1)
        self.ui.kv_unified.setChecked(False)
        self.ui.speculative_mtp.setChecked(True)
        draft_path = self._auto_mtp_draft_path(info)
        if draft_path:
            self.ui.spec_draft_model_path.setText(draft_path)
        else:
            self.ui.spec_draft_model_path.clear()
        self.ui.spec_draft_n_max.setValue(2)
        self.ui.spec_draft_gpu_layers.setText("all")
        self.ui.cuda_device.setText("CUDA0")
        self.ui.spec_draft_device.setText("CUDA0")
        self.ui.split_mode.setCurrentText("none")
        self.ui.main_gpu.setValue(0)
        self.ui.cuda_visible_devices.setText("0")
        self.ui.cuda_module_loading.setText("LAZY")
        logical = max(os.cpu_count() or 4, 1)
        self.ui.threads.setValue(min(8, logical))
        self.ui.threads_batch.setValue(min(16, logical))
        self.ui.fit_off.setChecked(True)
        self.ui.reasoning_mode.setCurrentText("off")
        self.ui.enable_thinking.setCurrentText("false")
        self.ui.cache_prompt.setChecked(True)
        self.ui.use_mmproj.setChecked(False)
        self.ui.jinja.setChecked(True)
        self.ui.context_shift.setChecked(False)

    def apply_recommended_params(self, info):
        rec = info.get("recommended_ctx")
        if rec:
            self.ui.ctx_size.setValue(rec)
        if self._auto_mtp_supported(info):
            self._apply_mtp_recommended_params(info)
            return
        if self._is_mtp_model_info(info):
            self.ui.speculative_mtp.setChecked(False)
            self.ui.spec_draft_model_path.clear()
            self.log_mgr.append(
                "MTP auto: not enabled. No embedded MTP metadata and no nearby MTP draft GGUF was found.",
                "warn",
            )
        if str(info.get("architecture") or "").lower().startswith("gemma4") or info.get("is_qat"):
            self.ui.flash_attn.setChecked(True)
            self.ui.jinja.setChecked(True)
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

        self._autotune_best_applied = False
        info = self.ui.models_by_path.get(self.ui.model_combo.currentData())
        if not info:
            return

        gpu_layers_val = (
            999
            if self.ui.gpu_auto.isChecked() or self.ui.gpu_layers_all.isChecked()
            else self.ui.gpu_layers.value()
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

        self._autotune_best_applied = False
        info = self.ui.models_by_path.get(self.ui.model_combo.currentData())
        if info:
            self._refresh_tooltips(info)

        self.update_cli_preview()
        self._mark_restart_needed()

    def update_cli_preview(self):
        try:
            self.config.read_from_ui(self.ui)
            args = build_args(self.config.settings, self.ui.model_combo.currentData())
            exe = self._resolve_llamacpp_executable("server") or "llama-server.exe"
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

    def _system_ram_snapshot_mib(self):
        """Returns (used_mib, total_mib, percent) for system RAM without extra deps."""
        if sys.platform.startswith("win"):
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total = stat.ullTotalPhys / (1024**2)
                    used = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024**2)
                    pct = (used / total * 100.0) if total else 0.0
                    return used, total, pct
            except Exception:
                return None
        return None

    def _gpu_total_snapshots(self) -> list[dict]:
        """Returns total GPU usage from nvidia-smi. Includes all processes."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
                **no_console_kwargs(),
            )
        except Exception:
            return []
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                used = float(parts[2])
                total = float(parts[3])
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "used_mib": used,
                        "total_mib": total,
                        "util_pct": float(parts[4]),
                        "temp_c": float(parts[5]),
                        "used_pct": (used / total * 100.0) if total else 0.0,
                    }
                )
            except ValueError:
                continue
        return gpus

    def _server_working_set_mib(self) -> float | None:
        """Возвращает Working Set llama-server процесса на Windows, если доступно."""
        pid = int(self.server.server_proc.processId() or 0)
        if not pid or not sys.platform.startswith("win"):
            return None
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
                **no_console_kwargs(),
            )
        except Exception:
            return None
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if not line or "INFO:" in line:
            return None
        try:
            import csv

            row = next(csv.reader([line]))
            mem_text = row[4]
            digits = "".join(ch for ch in mem_text if ch.isdigit())
            return int(digits) / 1024.0 if digits else None
        except Exception:
            return None

    def _server_gpu_memory_mib(self) -> float | None:
        """Returns llama-server GPU memory from nvidia-smi, if available."""
        pid = int(self.server.server_proc.processId() or 0)
        if not pid:
            return None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
                **no_console_kwargs(),
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        total = 0.0
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                if int(parts[0]) == pid:
                    total += float(parts[1])
            except ValueError:
                continue
        return total or None

    def _parsed_memory_without_process_fallback(self) -> float:
        return sum(
            mib
            for comps in self._mem_data.raw_devices.values()
            for comp, mib in comps.items()
            if comp not in {"process_working_set", "process_gpu_memory"}
        )

    def _update_process_memory_fallbacks(self) -> tuple[float | None, float | None]:
        """Adds process RAM/VRAM fallback when llama.cpp prints no buffers."""
        ws = self._server_working_set_mib()
        gpu = self._server_gpu_memory_mib()
        # Process memory overlaps with parsed buffers, so use it only as a fallback
        # when llama.cpp did not print detailed RAM/VRAM allocations.
        if self._parsed_memory_without_process_fallback() <= 0:
            if ws is not None:
                self._mem_data.raw_devices.setdefault("PROCESS", {})[
                    "process_working_set"
                ] = ws
            if gpu is not None:
                self._mem_data.raw_devices.setdefault("CUDA_PROCESS", {})[
                    "process_gpu_memory"
                ] = gpu
        return ws, gpu

    def _memory_estimate_lines(self) -> list[str]:
        model_path = self._current_model_path()
        if not model_path:
            return []
        info = self.ui.models_by_path.get(model_path) or extract_model_info(model_path)
        block_count = int(info.get("block_count") or 0)
        gpu_layers = block_count or 999
        if not getattr(self.config.settings, "gpu_auto", True) and not getattr(
            self.config.settings, "gpu_layers_all", False
        ):
            gpu_layers = int(getattr(self.config.settings, "gpu_layers", gpu_layers) or gpu_layers)
        ctx_size = int(getattr(self.config.settings, "ctx_size", 0) or 0)
        if ctx_size <= 0:
            ctx_size = int(info.get("recommended_ctx") or info.get("context_length") or 4096)
        parallel_slots = int(getattr(self.config.settings, "parallel_slots", 1) or 1)
        if parallel_slots <= 0:
            parallel_slots = 1
        ncmoe = int(getattr(self.config.settings, "cpu_moe_layers", 0) or 0)
        if ncmoe < 0:
            ncmoe = 0
        est = full_vram_estimate(
            info,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            cache_type_k=getattr(self.config.settings, "cache_type_k", "f16"),
            cache_type_v=getattr(self.config.settings, "cache_type_v", "f16"),
            flash_attn=bool(getattr(self.config.settings, "flash_attn", True)),
            parallel_slots=parallel_slots,
            ncmoe=ncmoe,
        )
        gpu_text = "all" if getattr(self.config.settings, "gpu_layers_all", False) else str(gpu_layers)
        return [
            "  Estimated VRAM allocation (heuristic, not measured):",
            f"    Model weights: {est.model_vram_gib:.2f} GiB",
            f"    KV cache: {est.kv_cache_gib:.2f} GiB ({est.kv_per_1k_ctx_mib:.1f} MiB / 1K ctx)",
            f"    Runtime/compute overhead: {est.overhead_gib:.2f} GiB",
            f"    Total estimated VRAM: {est.total_gib:.2f} GiB",
            "    Settings: "
            f"ctx={ctx_size:,}, KV={getattr(self.config.settings, 'cache_type_k', 'f16')}/"
            f"{getattr(self.config.settings, 'cache_type_v', 'f16')}, "
            f"flash-attn={'on' if getattr(self.config.settings, 'flash_attn', True) else 'off'}, "
            f"slots={parallel_slots}, gpu-layers={gpu_text}, ncmoe={ncmoe}",
            f"    Model metadata: layers={info.get('block_count') or '?'}, heads={info.get('head_count') or '?'}, "
            f"kv-heads={info.get('head_count_kv') or 'same/unknown'}, emb={info.get('embedding_length') or '?'}",
        ]

    def _memory_summary_text(self) -> str:
        ws, gpu = self._update_process_memory_fallbacks()
        system_ram = self._system_ram_snapshot_mib()
        gpu_snapshots = self._gpu_total_snapshots()
        agg = self._mem_data.get_aggregated()
        lines = ["📊 Memory after load:"]
        if gpu_snapshots or system_ram:
            lines.append("  Measured system snapshot:")
            for gpu_info in gpu_snapshots:
                lines.append(
                    f"    GPU{gpu_info['index']} {gpu_info['name']}: "
                    f"{fmt_mem(gpu_info['used_mib'])} / {fmt_mem(gpu_info['total_mib'])} "
                    f"({gpu_info['used_pct']:.1f}%), util {gpu_info['util_pct']:.0f}%, "
                    f"temp {gpu_info['temp_c']:.0f}°C"
                )
            if system_ram:
                used, total, pct = system_ram
                lines.append(f"    System RAM: {fmt_mem(used)} / {fmt_mem(total)} ({pct:.1f}%)")
            if gpu_snapshots:
                lines.append("    Note: GPU total includes desktop and other processes.")
        for cat in ("VRAM", "RAM"):
            total = self._mem_data.total(cat)
            cap = self._mem_data.system_memory.get(cat)
            if total <= 0 and not cap:
                continue
            cap_text = f" / {fmt_mem(cap)}" if cap else ""
            util = self._mem_data.utilization(cat)
            util_text = f" ({util:.1f}%)" if util is not None else ""
            lines.append(f"  {cat}: {fmt_mem(total)}{cap_text}{util_text}")
            comps = agg.get(cat, {})
            comp_parts = []
            for comp, mib in sorted(comps.items(), key=lambda item: item[1], reverse=True):
                label = COMPONENT_META.get(comp, {}).get("label", comp)
                comp_parts.append(f"{label} {fmt_mem(mib, short=True)}")
            if comp_parts:
                lines.append(f"    {'; '.join(comp_parts)}")

        parsed_detail = self._parsed_memory_without_process_fallback() > 0
        if not parsed_detail:
            if ws is not None and "process_working_set" not in agg.get("RAM", {}):
                lines.append(f"  RAM Process Working Set: {fmt_mem(ws)}")
            if gpu is not None and "process_gpu_memory" not in agg.get("VRAM", {}):
                lines.append(f"  VRAM Process GPU memory: {fmt_mem(gpu)}")
            lines.append(
                "  Detail: llama.cpp did not print per-buffer RAM/VRAM sizes; "
                "using OS process counters plus an estimate below."
            )
            lines.extend(self._memory_estimate_lines())
        elif ws is not None:
            lines.append(f"  Process RAM working set: {fmt_mem(ws)}")
        if len(lines) == 1:
            lines.append(
                "  llama.cpp did not print buffer sizes and process counters are unavailable. "
                "Increase log verbosity (-lv) or check nvidia-smi/Task Manager."
            )
            lines.extend(self._memory_estimate_lines())
        return "\n".join(lines)

    def _maybe_log_memory_summary(self):
        if self._memory_summary_logged or not self._mem_data.server_ready:
            return
        self._memory_summary_logged = True
        self.log_mgr.append(self._memory_summary_text())

    def _abort_bad_mtp_launch(self):
        """Immediately kill a broken MTP launch so model loading cannot keep VRAM busy."""
        if not self.server.is_server_running():
            return
        self.log_mgr.append(
            "⛔ Fatal MTP error detected during load: killing llama-server now to free RAM/VRAM. "
            "The app will retry once without MTP.",
            "error",
        )
        self.server.force_stop_server()

    def _mark_mtp_launch_failed(self, reason: str, fatal: bool = False):
        self._mtp_draft_error_seen = True
        self._mtp_failure_reason = reason
        if fatal and not self._mtp_auto_abort_requested:
            self._mtp_auto_abort_requested = True
            QTimer.singleShot(0, self._abort_bad_mtp_launch)

    def _on_log_for_mem_viz(self, text: str, level: str):
        """Обработка логов для визуализации памяти."""
        lower_text = text.lower()
        if "model doesn't contain mtp layers" in lower_text:
            self._mark_mtp_launch_failed("main GGUF does not contain MTP layers", fatal=True)
        elif "failed to create mtp context" in lower_text:
            self._mark_mtp_launch_failed("failed to create MTP context", fatal=True)
        elif (
            "failed to load draft model" in lower_text
            or "common_speculative_init_result" in lower_text
            or "invalid vector subscript" in lower_text
        ):
            self._mark_mtp_launch_failed("draft GGUF failed to load", fatal=True)
        elif (
            "mtp input row width must match" in lower_text
            or ("ggml_assert" in lower_text and "mtp" in lower_text)
            or ("speculative.cpp" in lower_text and "mtp" in lower_text)
        ):
            self._mark_mtp_launch_failed(
                "draft GGUF is incompatible with the main model (embedding width mismatch)",
                fatal=True,
            )
        for line in text.splitlines():
            parse_line(line, self._mem_data)
        if self._mem_data.server_ready:
            self._update_process_memory_fallbacks()
        self._maybe_log_memory_summary()
        # Принудительно обновляем UI после каждого блока логов
        self.ui.mem_viz.update_from_data(self._mem_data)
        # Обрабатываем события Qt чтобы UI не зависал
        QApplication.processEvents()

    def _reset_mem_viz(self, status: str | None = None):
        """Сброс визуализации памяти."""
        self._mem_data = MemoryData()
        self._memory_summary_logged = False
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

    def _strip_mtp_args(self, args: list[str]) -> list[str]:
        value_flags = {
            "-md",
            "--model-draft",
            "--spec-draft-device",
            "--spec-type",
            "--spec-draft-n-max",
            "--spec-draft-n-min",
            "--spec-draft-p-min",
            "--spec-draft-ngl",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
        }
        stripped = []
        i = 0
        while i < len(args):
            arg = args[i]
            base = arg.split("=", 1)[0] if str(arg).startswith("-") else arg
            if base in value_flags:
                if "=" not in str(arg) and i + 1 < len(args):
                    i += 2
                else:
                    i += 1
                continue
            stripped.append(arg)
            i += 1
        return stripped

    def _retry_without_mtp_if_needed(self, exit_code: int) -> bool:
        if exit_code == 0 or self._mtp_fallback_attempted:
            return False
        if not self._mtp_draft_error_seen or not self._last_server_launch:
            return False

        exe, args, env = self._last_server_launch
        if "--spec-type" not in args and "--model-draft" not in args and "-md" not in args:
            return False

        fallback_args = self._strip_mtp_args(args)
        if fallback_args == args:
            return False

        self._mtp_fallback_attempted = True
        reason = self._mtp_failure_reason or "MTP initialization failed"
        self._mtp_draft_error_seen = False
        self._mtp_failure_reason = ""
        self._mtp_auto_abort_requested = False
        self.ui.speculative_mtp.setChecked(False)
        self.log_mgr.append(
            f"⚠️ MTP disabled: {reason}. Retrying once without MTP so the main model can start. "
            "For automatic MTP use a main GGUF/package that actually contains MTP layers, "
            "or a matching supported draft GGUF with a new llama.cpp build.",
            "warn",
        )
        self._reset_mem_viz("MTP failed, retrying without MTP...")
        QTimer.singleShot(
            150,
            lambda: self._launch_server(
                exe,
                fallback_args,
                env=env,
                action="Retry without MTP (draft failed)",
            ),
        )
        return True

    def _on_server_stopped(self):
        """Обработчик остановки сервера."""
        self._stop_metrics_polling()
        if self._restart_pending:
            self._reset_mem_viz("Сервер остановлен, перезапуск с новыми параметрами...")
            QTimer.singleShot(150, self._start_pending_restart)
            return
        exit_code = self.server.server_proc.exitCode()
        if self._retry_without_mtp_if_needed(exit_code):
            return
        self._finalize_mem_viz_after_stop(
            exit_code,
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
        exe = self._resolve_llamacpp_executable("server")
        if not exe or not os.path.exists(exe):
            QMessageBox.critical(
                self.ui,
                "Error",
                "Select llama.cpp base folder with the requested CUDA build.",
            )
            return None
        self.config.read_from_ui(self.ui)
        # resolve mmproj
        model_path = self.ui.model_combo.currentData()
        info = self.ui.models_by_path.get(model_path) or {}
        if model_path:
            info.setdefault("path", model_path)
            info.setdefault("_model_path", model_path)
        is_mtp_model = self._is_mtp_model_info(info)
        if not is_mtp_model:
            # Сохранённый MTP-чекбокс/пресет не должен ломать обычные GGUF.
            # Пользовательские эксперименты всё ещё можно задать вручную в Extra params.
            self.config.settings.speculative_mtp = False
        if (
            self.ui.auto_params.isChecked()
            and is_mtp_model
            and not self._auto_mtp_supported(info)
        ):
            self.config.settings.speculative_mtp = False
            self.config.settings.spec_draft_model_path = ""
            self.log_mgr.append(
                "MTP auto: skipped because no embedded MTP metadata and no nearby MTP draft GGUF was found.",
                "warn",
            )
        if self.ui.auto_params.isChecked() and self._auto_mtp_supported(info):
            block_count = int(info.get("block_count") or 0)
            self.config.settings.gpu_auto = False
            self.config.settings.gpu_layers_all = True
            if block_count > 0:
                self.config.settings.gpu_layers = block_count
            self.config.settings.cache_type_k = "q8_0"
            self.config.settings.cache_type_v = "q8_0"
            self.config.settings.batch_size = 512
            self.config.settings.ubatch_size = 256
            self.config.settings.parallel_slots = 1
            self.config.settings.kv_unified = False
            self.config.settings.speculative_mtp = True
            auto_draft_path = self._auto_mtp_draft_path(info)
            if auto_draft_path:
                self.config.settings.spec_draft_model_path = (
                    self.config.settings.spec_draft_model_path or auto_draft_path
                )
            elif self._uses_embedded_mtp_mode(info):
                self.config.settings.spec_draft_model_path = ""
            self.config.settings.spec_draft_n_max = 2
            self.config.settings.spec_draft_gpu_layers = "all"
            self.config.settings.cuda_device = (
                self.config.settings.cuda_device or "CUDA0"
            )
            self.config.settings.spec_draft_device = (
                self.config.settings.spec_draft_device
                or self.config.settings.cuda_device
            )
            self.config.settings.split_mode = self.config.settings.split_mode or "none"
            if self.config.settings.main_gpu < 0:
                self.config.settings.main_gpu = 0
            self.config.settings.cuda_visible_devices = (
                self.config.settings.cuda_visible_devices or "0"
            )
            self.config.settings.cuda_module_loading = (
                self.config.settings.cuda_module_loading or "LAZY"
            )
            self.config.settings.fit_off = True
            self.config.settings.reasoning_mode = "off"
            self.config.settings.enable_thinking = "false"
            self.config.settings.cache_prompt = True
            self.config.settings.use_mmproj = False
            self.config.settings.jinja = True
            self.config.settings.context_shift = False
        self.config.settings.mmproj_path = info.get("mmproj_path", "")
        try:
            args = build_args(self.config.settings, self.ui.model_combo.currentData())
        except ValueError as e:
            QMessageBox.warning(self.ui, "Error", str(e))
            return None
        if not args:
            return None
        if getattr(self.config.settings, "speculative_mtp", False):
            if self.config.settings.spec_draft_model_path:
                self.log_mgr.append(
                    "MTP auto: using separate draft GGUF (--model-draft). "
                    "If it fails, the app will retry without MTP.",
                    "info",
                )
            else:
                self.log_mgr.append(
                    "MTP auto: using embedded/package mode (--spec-type draft-mtp, "
                    "no --model-draft). This is the preferred mode for Gemma 4 regular GGUF.",
                    "info",
                )
        env = self._server_env_from_settings()
        return exe, args, env

    def _server_env_from_settings(self):
        env = {}
        cuda_visible = str(
            getattr(self.config.settings, "cuda_visible_devices", "") or ""
        ).strip()
        cuda_loading = str(
            getattr(self.config.settings, "cuda_module_loading", "") or ""
        ).strip()
        if cuda_visible:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        if cuda_loading:
            env["CUDA_MODULE_LOADING"] = cuda_loading
        return env

    def _launch_server(
        self,
        exe: str,
        args: list[str],
        env: dict | None = None,
        action: str = "Starting server",
    ):
        cuda_ver = self._selected_cuda_version()
        is_retry = action.lower().startswith("retry without mtp")
        if not is_retry:
            self.log_mgr.clear()
        env_text = ""
        if env:
            env_text = "\n   Env: " + " ".join(f"{k}={v}" for k, v in env.items())
        self.log_mgr.append(
            f"{action} [CUDA {cuda_ver}]: {exe}\n   Args: {' '.join(args)}{env_text}"
        )
        self._last_server_launch = (exe, list(args), dict(env or {}))
        if "--spec-type" in args or "--model-draft" in args or "-md" in args:
            self._mtp_draft_error_seen = False
            self._mtp_failure_reason = ""
            self._mtp_auto_abort_requested = False
            if not action.lower().startswith("retry without mtp"):
                self._mtp_fallback_attempted = False
        self._reset_mem_viz()
        self.server.start_server(exe, args, env=env)
        self._start_metrics_polling()
        self._reset_restart_indicator()
        self.ui.start_btn.setVisible(False)
        self.ui.reload_btn.setVisible(True)
        self.ui.start_btn.setEnabled(False)
        self.ui.reload_btn.setEnabled(True)
        self.ui.test_btn.setEnabled(False)
        self.ui.stop_btn.setEnabled(True)
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
            exe, args, env = launch
            self._launch_server(exe, args, env=env, action="Restarting server")

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
        exe, args, env = launch
        self._launch_server(exe, args, env=env)

    def run_benchmark(self):
        if self.server.is_server_running():
            QMessageBox.warning(
                self.ui, "Server running", "Stop server before running benchmark"
            )
            return
        bexe = self.auto_detect_bench()
        if not bexe or not os.path.exists(bexe):
            QMessageBox.critical(
                self.ui,
                "Error",
                "llama-bench.exe was not found in the selected CUDA build folder.",
            )
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
        env = self._server_env_from_settings()
        env_text = ""
        if env:
            env_text = "\n   Env: " + " ".join(f"{k}={v}" for k, v in env.items())
        self.log_mgr.append(
            f"Running benchmark [CUDA {self._selected_cuda_version()}]: {os.path.basename(bexe)}\n"
            f"   Params: {' '.join(args)}{env_text}"
        )
        self._reset_mem_viz()
        self.server.start_bench(bexe, args, env=env)
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
        """Открывает секцию Benchmark/AutoTune и строит свежий план."""
        if not self.ui.bench_panel.toggle_btn.isChecked():
            self.ui.bench_panel.toggle_btn.setChecked(True)
            self.ui.bench_panel.toggle_visibility()
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
            early_stop_on_peak=bool(options.get("early_stop_on_peak", False)),
        )
        self.autotune_plan = plan
        self.autotune_best_result = None
        self._autotune_best_applied = False
        self.autotune_results_dir = ""
        self.ui.autotune.set_plan(plan)
        if not self.ui.bench_panel.toggle_btn.isChecked():
            self.ui.bench_panel.toggle_btn.setChecked(True)
            self.ui.bench_panel.toggle_visibility()
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
                **no_console_kwargs(),
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
        bexe = self.auto_detect_bench()
        if not bexe or not os.path.exists(bexe):
            QMessageBox.critical(
                self.ui,
                "AutoTune",
                "llama-bench.exe was not found in the selected CUDA build folder.",
            )
            return
        plan = self.autotune_plan or self.build_autotune_plan()
        if not plan:
            return
        options = self.ui.autotune.options()
        plan = self.ui.autotune.apply_table_edits_to_plan(plan)
        self.autotune_plan = plan
        plan.early_stop_on_peak = bool(options.get("early_stop_on_peak", False))
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
        self._autotune_best_applied = False
        self.autotune_results_dir = ""
        self.autotune = AutoTuneManager(
            bexe,
            plan,
            model_info=self._current_model_info(),
            prompt_tokens=self.ui.bench_prompt.value(),
            generation_tokens=self.ui.bench_gen.value(),
            per_run_timeout_sec=options["per_run_timeout_sec"],
        )
        self.ui.autotune.prepare_run(
            len(plan.candidates), options["per_run_timeout_sec"]
        )
        self.autotune.log.connect(lambda text, level: self.log_mgr.append(text, level))
        self.autotune.log.connect(
            lambda text, _level: self.ui.autotune.append_activity(text)
        )
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
                params = dict(candidate.params)
                if str(params.get("ngl", "")).strip().lower() == "auto":
                    info = self._current_model_info()
                    block_count = int(info.get("block_count") or 0)
                    params["ngl"] = block_count if block_count > 0 else 99
                return params
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
            # Auto-apply best + server verification если включен чекбокс
            try:
                options = self.ui.autotune.options()
                if options.get("verify_server_after_apply"):
                    self.log_mgr.append(
                        "AutoTune auto-apply: applying best and starting server verification...",
                        "info",
                    )
                    if self.apply_autotune_best(silent=True):
                        self._start_or_restart_server_with_verification()
            except Exception as e:
                self.log_mgr.append(
                    f"Auto-apply after autotune failed (non-critical): {e}", "warn"
                )
        else:
            self.log_mgr.append(
                f"AutoTune finished: no successful result, results={output_dir}", "warn"
            )
        self.update_action_buttons()

    def apply_autotune_best(self, silent=False):
        params = self._best_autotune_params()
        if not params:
            if not silent:
                QMessageBox.warning(self.ui, "AutoTune", "No best result to apply")
            return False

        # Убеждаемся что ngl — число, не "auto", для reproducibility
        ngl_raw = params.get("ngl", "auto")
        if str(ngl_raw).strip().lower() == "auto":
            info = self._current_model_info()
            block_count = int(info.get("block_count") or 0)
            ngl_val = block_count if block_count > 0 else 99
            params["ngl"] = ngl_val
        else:
            ngl_val = int(ngl_raw)

        self._loading_preset = True
        try:
            self.ui.gpu_auto.setChecked(False)
            self.ui.gpu_layers.setValue(ngl_val)
            self.ui.ctx_size.setValue(
                int(params.get("ctx_size", self.ui.ctx_size.value()))
            )
            self.ui.batch_size.setValue(
                int(params.get("batch_size", self.ui.batch_size.value()))
            )
            self.ui.ubatch_size.setValue(
                int(params.get("ubatch_size", self.ui.ubatch_size.value()))
            )
            self.ui.cache_type_k.setCurrentText(
                str(params.get("cache_type_k", self.ui.cache_type_k.currentText()))
            )
            self.ui.cache_type_v.setCurrentText(
                str(params.get("cache_type_v", self.ui.cache_type_v.currentText()))
            )
            self.ui.threads.setValue(
                int(params.get("threads", self.ui.threads.value()))
            )
            self.ui.threads_batch.setValue(
                int(params.get("threads_batch", self.ui.threads_batch.value()))
            )
            self.ui.parallel_slots.setValue(
                int(params.get("parallel_slots", self.ui.parallel_slots.value()))
            )
            self.ui.kv_unified.setChecked(
                bool(params.get("kv_unified", self.ui.kv_unified.isChecked()))
            )
            self.ui.speculative_mtp.setChecked(
                bool(params.get("speculative_mtp", self.ui.speculative_mtp.isChecked()))
            )
            self.ui.spec_draft_n_max.setValue(
                int(params.get("spec_draft_n_max", self.ui.spec_draft_n_max.value()))
            )
            self.ui.flash_attn.setChecked(
                bool(params.get("flash_attn", self.ui.flash_attn.isChecked()))
            )
            self.ui.fit_off.setChecked(
                bool(params.get("fit_off", self.ui.fit_off.isChecked()))
            )
            self.ui.cache_prompt.setChecked(
                bool(params.get("cache_prompt", self.ui.cache_prompt.isChecked()))
            )
            self.ui.cpu_moe_layers.setValue(
                int(params.get("ncmoe", self.ui.cpu_moe_layers.value()))
            )
            self.ui.ctx_checkpoints.setValue(
                int(params.get("ctx_checkpoints", self.ui.ctx_checkpoints.value()))
            )
            self.ui.cache_ram.setValue(
                int(params.get("cache_ram", self.ui.cache_ram.value()))
            )
            self.ui.use_mmproj.setChecked(
                bool(params.get("use_mmproj", self.ui.use_mmproj.isChecked()))
            )

            # Санитизируем extra_args от флагов, управляемых UI/AutoTune
            current_extra = self.ui.extra_args.text()
            if current_extra.strip():
                from src.core.config import _sanitize_extra_args

                sanitized = _sanitize_extra_args(current_extra)
                self.ui.extra_args.setText(sanitized)
        finally:
            self._loading_preset = False

        self.update_cli_preview()
        self._mark_restart_needed()
        self._autotune_best_applied = True
        self.save_settings()
        self.log_mgr.append(
            f"AutoTune best applied: {self.autotune_best_result.candidate_id if self.autotune_best_result else ''}; "
            f"ngl={ngl_val} KV {params.get('cache_type_k', '?')}/{params.get('cache_type_v', '?')} "
            f"b={params.get('batch_size', '?')} ub={params.get('ubatch_size', '?')} "
            f"t={params.get('threads', '?')} tb={params.get('threads_batch', '?')}"
        )
        if not silent:
            options = self.ui.autotune.options()
            if options.get("verify_server_after_apply"):
                self._start_or_restart_server_with_verification()
                QMessageBox.information(
                    self.ui,
                    "AutoTune",
                    "Best parameters applied. Server verification has been started.",
                )
            else:
                QMessageBox.information(
                    self.ui, "AutoTune", "Best parameters applied to UI"
                )
        return True

    def _start_or_restart_server_with_verification(self):
        launch = self._prepare_server_launch()
        if not launch:
            return
        exe, args, env = launch
        expected_cli = f"{exe} {' '.join(args)}"
        self._pending_server_verify = {
            "started_at": time.monotonic(),
            "attempts": 0,
            "expected_cli": expected_cli,
        }
        self.log_mgr.append(
            "AutoTune server verification: launching llama-server with applied best params\n"
            f"   Expected CLI: {expected_cli}",
            "info",
        )
        if self.server.is_server_running():
            self._pending_restart_launch = (exe, args, env)
            self._restart_pending = True
            self.server.stop_server()
        else:
            self._launch_server(
                exe, args, env=env, action="Starting verified AutoTune server"
            )
        QTimer.singleShot(1200, self._poll_verified_server)

    def _poll_verified_server(self):
        state = getattr(self, "_pending_server_verify", None)
        if not state:
            return
        state["attempts"] += 1
        if state["attempts"] > 60:
            self.log_mgr.append("AutoTune server verification timed out", "warn")
            self._pending_server_verify = None
            return
        if not self.server.is_server_running():
            QTimer.singleShot(1000, self._poll_verified_server)
            return

        base_url = self.ui.current_base_url().rstrip("/")
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=2) as resp:
                if resp.status >= 400:
                    raise urllib.error.URLError(f"HTTP {resp.status}")
            self._send_verified_server_request(base_url, state)
        except Exception:
            QTimer.singleShot(1000, self._poll_verified_server)

    def _send_verified_server_request(self, base_url: str, state: dict):
        payload = json.dumps(
            {
                "model": self.ui.current_model_id(),
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 8,
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
            elapsed = time.monotonic() - started
            tokens = 0
            try:
                data = json.loads(text)
                tokens = int(data.get("usage", {}).get("completion_tokens") or 0)
            except Exception:
                pass
            speed = f", ~{tokens / elapsed:.1f} tok/s" if tokens and elapsed > 0 else ""
            total_elapsed = time.monotonic() - float(state.get("started_at", started))
            self.log_mgr.append(
                f"AutoTune server verification OK: test request completed in {elapsed:.1f}s{speed}; "
                f"server ready after {total_elapsed:.1f}s",
                "info",
            )

            # Сравнение benchmark TG с реальной server TG
            if tokens and elapsed > 0:
                server_tg = tokens / elapsed
                bench_tg = (
                    float(self.autotune_best_result.generation_tok_s)
                    if self.autotune_best_result
                    else 0.0
                )
                if bench_tg > 0:
                    ratio = server_tg / bench_tg
                    if ratio < 0.8:
                        msg = (
                            f"⚠️ Server TG ({server_tg:.1f}) is {int((1 - ratio) * 100)}% slower than "
                            f"benchmark TG ({bench_tg:.1f}). "
                            f"Consider reviewing KV cache, flash_attn or ctx_checkpoints settings."
                        )
                        self.log_mgr.append(msg, "warn")
                    elif ratio < 0.95:
                        self.log_mgr.append(
                            f"📊 Server TG {server_tg:.1f} vs benchmark TG {bench_tg:.1f} "
                            f"({int(ratio * 100)}% — acceptable difference)",
                            "info",
                        )
                    else:
                        self.log_mgr.append(
                            f"✓ Server TG {server_tg:.1f} matches benchmark TG {bench_tg:.1f} "
                            f"({int(ratio * 100)}% — excellent reproducibility)",
                            "info",
                        )

            # Auto-save preset на успешной verification
            if tokens > 0:
                try:
                    model_path = self._current_model_path()
                    if model_path and self.autotune_best_result and self.autotune_plan:
                        ctx = self.ui.ctx_size.value()
                        metadata = {
                            "source": "autotune_verified",
                            "run_id": self.autotune_best_result.candidate_id,
                            "bench_tg": self.autotune_best_result.generation_tok_s,
                            "server_tg": tokens / elapsed if elapsed > 0 else 0,
                            "score": self.autotune_best_result.score,
                            "results_dir": self.autotune_results_dir,
                        }
                        self.config.save_perf_preset(
                            model_path,
                            ctx,
                            self.ui,
                            metadata=metadata,
                            autotune_params=self._best_autotune_params(),
                        )
                        self.log_mgr.append(
                            f"AutoTune preset auto-saved after verification: ctx={ctx:,}",
                            "info",
                        )
                except Exception as e:
                    self.log_mgr.append(
                        f"Auto-save preset failed (non-critical): {e}", "warn"
                    )
        except Exception as exc:
            self.log_mgr.append(
                f"AutoTune server verification request failed: {exc}", "warn"
            )
        finally:
            self._pending_server_verify = None

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
            self.config.save_perf_preset(
                model_path,
                ctx,
                self.ui,
                metadata=metadata,
                autotune_params=self._best_autotune_params(),
            )
        except (ValueError, OSError) as e:
            QMessageBox.warning(self.ui, "AutoTune", str(e))
            return
        self.log_mgr.append(
            f"AutoTune preset saved: {Path(model_path).name} | ctx={ctx:,}"
        )
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
        if not self.autotune_results_dir or not os.path.isdir(
            self.autotune_results_dir
        ):
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
        if self.server.is_server_running():
            self.log_mgr.append(
                "Force stop requested: killing llama-server now", "error"
            )
            self.server.force_stop_server()
            self.update_action_buttons()
            return
        # Нет своего процесса — попробуем убить внешние llama процессы
        external = self._external_llama_processes()
        if external:
            self.log_mgr.append(
                f"Force stop: killing {len(external)} external llama process(es)...",
                "error",
            )
            if sys.platform.startswith("win"):
                for proc in external:
                    pid = proc.get("ProcessId")
                    if pid:
                        subprocess.run(
                            ["taskkill", "/PID", str(pid), "/T", "/F"],
                            capture_output=True,
                            timeout=10,
                            **no_console_kwargs(),
                        )
            self.update_action_buttons()
            return
        self.log_mgr.append("Force stop skipped: no llama processes found", "warn")
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
        self.ui.force_stop_btn.setEnabled(True)
        self.ui.update_llama_btn.setEnabled(not busy and not upd)
        self.ui.start_btn.setEnabled(not busy and not upd)
        self.ui.cuda_version_combo.setEnabled(not busy and not upd)
        self.ui.exe_path.setEnabled(not busy and not upd)
        # Lock model & all params while server/bench/autotune is running
        lock = busy or upd
        for w in getattr(self.ui, "_runtime_lockable", []):
            try:
                w.setEnabled(not lock)
            except RuntimeError:
                pass
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
        if not exe:
            QMessageBox.critical(
                self.ui, "Updater", "Select llama.cpp base folder first."
            )
            return
        self.ui.update_progress.setValue(0)
        self.ui.update_progress.setVisible(True)
        cuda_version = self.ui.cuda_version_combo.currentData() or "12"
        self.updater = LlamaCppUpdater(exe, cuda_version=cuda_version)
        self.updater.progress.connect(self.ui.update_status.setText)
        self.updater.percent.connect(self.ui.update_progress.setValue)
        self.updater.completed.connect(
            lambda ch, msg: (
                self.ui.update_status.setText(msg),
                self.auto_detect_bench(),
                self.save_settings(),
            )
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
        self.metrics.stop()
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
    icon_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    icon_path = icon_root / "assets" / "llama_server_icon.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    gui = LlamaGUI()
    gui.ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
