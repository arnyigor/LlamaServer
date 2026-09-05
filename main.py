"""LlamaServer GUI - точка входа."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from PySide6.QtCore import (
    QItemSelectionModel,
    QSettings,
    Qt,
    QTimer,
    QtMsgType,
    QTranslator,
    qInstallMessageHandler,
)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QProgressBar,
    QSystemTrayIcon,
    QMenu,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from src.core.cli_builder import build_args, merge_extra_args
from src.core.cli_parser import parse_llama_server_command
from src.core.config import ConfigManager, candidate_to_settings_values
from src.core.diagnostics import (
    analyze_server_failure,
    consume_previous_native_crash,
    diagnostics_dir,
    finish_native_crash_capture,
    format_diagnostic_summary,
    start_native_crash_capture,
    write_app_exception_report,
    write_server_report,
)
from src.core.constants import (
    STATUS_COLOR_BENCH,
    STATUS_COLOR_ERROR,
    STATUS_COLOR_MUTED,
    STATUS_COLOR_MUTED_DARK,
    STATUS_COLOR_PENDING,
    STATUS_COLOR_READY,
    STATUS_COLOR_RUNNING,
    STATUS_COLOR_WARNING,
    STAT_COLOR_CAPTION,
    STAT_COLOR_GENERATED,
    STAT_COLOR_PROMPT,
    STAT_COLOR_SAVED,
    STAT_COLOR_TASK,
    STAT_COLOR_TIME,
    STAT_COLOR_TOTAL,
    format_duration,
    format_speed,
    stat_kv,
    stat_sep,
)
from src.core.gguf_parser import extract_model_info, is_mtp_draft_file
from src.core.help_detector import is_spec_supported, probe_supported_flags
from src.core.mem_viz_parser import COMPONENT_META, MemoryData, fmt_mem, parse_line
from src.core.model_load_progress import progress_from_load_line
from src.core.mtp_fallback import (
    MtpFallbackController,
    MtpModelRules,
    strip_mtp_args,
)
from src.core.metrics_poller import MetricsPoller
from src.core.param_registry import PARAM_REGISTRY
from src.core.runtime_stats import RuntimeStatsController, format_runtime_stats_markdown
from src.core.server_launch import ServerLaunchController
from src.core.server_manager import ServerManager
from src.core.vram_estimator import full_vram_estimate
from llama_autotuner.session import SessionConfig
from llama_autotuner.llama.library import preferred_mmproj
from src.services.autotune_manager import AutoTuneManager
from src.services.hf_download_coordinator import HfDownloadCoordinator
from src.services.integration_manager import IntegrationManager
from src.services.hf_downloader import (
    HfModelDownloader,
    HfRepoScanner,
    delete_file_safely,
    format_bytes,
    list_all_local_model_entries,
    list_all_partial_downloads,
    list_local_repo_files,
    lmstudio_repo_dir,
    normalize_hf_repo_id,
    partial_download_info,
)
from src.services.threads import ModelScanner, LlamaCppUpdater, LlamaCppUpdateChecker
from src.ui.dialogs import confirm_destructive_action
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
        self.hf = HfDownloadCoordinator()
        self.hf.task_changed.connect(self._set_hf_task_display)
        self.hf.percent_changed.connect(
            lambda _key: self._refresh_hf_download_summary()
        )
        self.hf.task_completed.connect(self._on_hf_task_completed)
        self.hf.task_finished.connect(self._on_hf_task_finished)
        self.hf_scan_result = None
        self.autotune = None
        self._autotune_running = False
        self.autotune_session_result = None
        self._syncing_model_combo = False
        # Координация запуска (отложенные рестарты, env) — в контроллере.
        self.launcher = ServerLaunchController()
        self._pending_server_verify = None
        self._last_force_stop_confirmed_at = 0.0
        # Состояние MTP-fallback (детекция ошибок draft, retry без MTP) —
        # в контроллере; GUI-обёртки делегируют ему.
        self.mtp = MtpFallbackController()
        self._memory_summary_logged = False
        self._shutting_down = False
        self.last_diagnostic_path = ""
        self.last_diagnostic_summary = ""

        self.log_mgr = LogManager(self.ui.logs)
        self.log_mgr.speed_updated.connect(self._on_log_speed_updated)
        self.log_mgr.timing_updated.connect(self._on_log_timing_updated)
        self.log_mgr.toast_requested.connect(self.ui.toast_overlay.show_message)
        self.ui.autoscroll_logs.toggled.connect(
            lambda checked: setattr(self.log_mgr, "autoscroll", checked)
        )
        self.ui.copy_last_error_btn.clicked.connect(self.copy_last_error)
        self.ui.open_diagnostics_btn.clicked.connect(self.open_diagnostics_folder)
        self.metrics = MetricsPoller(poll_interval_ms=250)
        self.metrics.slot_metrics_updated.connect(self._on_slot_metrics_updated)
        self.metrics.server_metrics_updated.connect(self._on_server_metrics_updated)
        # Накопление runtime-статистики вынесено в контроллер; лейблы
        # обновляются через сигналы (см. _connect_stats_signals).
        self.stats = RuntimeStatsController()
        self._connect_stats_signals()

        self._llamacpp_update_info = {}
        self._update_checker = None

        self.config.load()
        self.config.apply_to_ui(self.ui)
        self._normalize_llamacpp_path_ui()
        self.auto_detect_bench()
        self._connect_signals()
        self._setup_tray()
        self._update_cuda_status()
        self._refresh_overview()
        QTimer.singleShot(250, self.auto_scan_models)
        QTimer.singleShot(350, self._refresh_hf_partial_status)
        QTimer.singleShot(600, self.auto_check_llamacpp_updates)

    @property
    def _mtp_draft_error_seen(self) -> bool:
        return self.mtp.draft_error_seen

    @property
    def _mtp_failure_reason(self) -> str:
        return self.mtp.failure_reason

    @property
    def _mtp_fallback_attempted(self) -> bool:
        return self.mtp.fallback_attempted

    @property
    def _last_server_launch(self):
        return self.mtp.last_launch

    def _connect_stats_signals(self):
        self.stats.tokens_changed.connect(self._render_tokens_label)
        self.stats.active_time_changed.connect(self._render_active_time_label)
        self.stats.current_time_changed.connect(self._render_current_time_label)
        self.stats.saved_changed.connect(self._render_saved_label)

    def _connect_signals(self):
        u = self.ui
        u.start_btn.clicked.connect(self.start_server)
        u.reload_btn.clicked.connect(self.restart_server)
        u.stop_btn.clicked.connect(self.stop_work)
        u.force_stop_btn.clicked.connect(self.force_stop_server)
        u.tokens_reset_btn.clicked.connect(self.reset_task_tokens)
        u.export_stats_btn.clicked.connect(self.export_runtime_stats)
        u.copy_stats_md_btn.clicked.connect(self.copy_runtime_stats_markdown)
        u.reset_session_btn.clicked.connect(self.reset_session)
        u.reset_saved_btn.clicked.connect(self.reset_saved_total)
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
        u.local_models_list.itemSelectionChanged.connect(
            self._update_local_model_delete_button
        )
        u.hf_files.itemSelectionChanged.connect(self._update_hf_download_button)
        u.hf_downloads.itemSelectionChanged.connect(self._update_hf_download_button)
        u.model_combo.currentIndexChanged.connect(self.on_model_selected)
        u.model_combo.currentIndexChanged.connect(self._sync_autotune_model_from_main)
        u.autotune.model_combo.currentIndexChanged.connect(
            self._sync_main_model_from_autotune
        )
        u.ctx_size.valueChanged.connect(self.on_ctx_changed)
        u.preset_name_combo.activated.connect(lambda _index: self._on_preset_selected())
        u.preset_name_combo.currentIndexChanged.connect(
            lambda _index: self._update_preset_buttons()
        )
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
        u.speculative_mtp.toggled.connect(self._on_mtp_checkbox_toggled)
        u.spec_draft_n_max.valueChanged.connect(self._on_param_changed)
        u.spec_draft_p_min.valueChanged.connect(self._on_param_changed)
        u.spec_draft_gpu_layers.textChanged.connect(self._on_param_changed)
        u.spec_draft_model_path.textChanged.connect(self._on_param_changed)
        u.spec_draft_model_path.textEdited.connect(self._on_mtp_draft_path_edited)
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
        u.reasoning_effort.currentIndexChanged.connect(self._on_param_changed)
        u.reasoning_preserve.currentIndexChanged.connect(self._on_param_changed)
        u.reasoning_budget.valueChanged.connect(self._on_param_changed)
        u.reasoning_budget_message.textChanged.connect(self._on_param_changed)
        u.host.textChanged.connect(self._on_param_changed)
        u.port.valueChanged.connect(self._on_param_changed)
        u.ctx_checkpoints.valueChanged.connect(self._on_param_changed)
        u.cache_ram.valueChanged.connect(self._on_param_changed)
        u.temperature.valueChanged.connect(self._on_param_changed)
        u.top_k.valueChanged.connect(self._on_param_changed)
        u.top_p.valueChanged.connect(self._on_param_changed)
        u.min_p.valueChanged.connect(self._on_param_changed)
        u.typical_p.valueChanged.connect(self._on_param_changed)
        u.repeat_penalty.valueChanged.connect(self._on_param_changed)
        u.repeat_last_n.valueChanged.connect(self._on_param_changed)
        u.presence_penalty.valueChanged.connect(self._on_param_changed)
        u.frequency_penalty.valueChanged.connect(self._on_param_changed)
        u.seed.valueChanged.connect(self._on_param_changed)
        u.use_mlock.stateChanged.connect(self._on_param_changed)
        u.verbose.stateChanged.connect(self._on_param_changed)
        u.log_timestamps.stateChanged.connect(self._on_param_changed)
        u.cuda_visible_devices.textChanged.connect(self._on_param_changed)
        u.cuda_module_loading.textChanged.connect(self._on_param_changed)
        u.context_shift.stateChanged.connect(self._on_param_changed)
        u.no_webui.stateChanged.connect(self._on_param_changed)
        u.use_mmproj.stateChanged.connect(self._on_param_changed)
        u.mmproj_offload.stateChanged.connect(self._on_param_changed)
        u.extra_args.textChanged.connect(self._on_param_changed)
        u.jinja.stateChanged.connect(self._on_param_changed)
        u.use_chat_template.stateChanged.connect(self._on_param_changed)
        u.chat_template_file.textChanged.connect(self._on_param_changed)
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
        u.cli_manual_mode.toggled.connect(self._on_cli_manual_mode_toggled)
        u.cli_apply_btn.clicked.connect(self.apply_cli_preview)
        u.cli_copy_btn.clicked.connect(self.copy_cli_preview)
        u.cli_import_btn.clicked.connect(self.import_cli_from_clipboard)
        u._browse_exe_clicked = self.browse_exe
        u._browse_bench_clicked = self.browse_bench
        u._browse_model_dir_clicked = self.browse_model_dir
        u._browse_opencode_clicked = self.browse_opencode_config
        u._browse_pi_clicked = self.browse_pi_config
        u._browse_chat_template_clicked = self.browse_chat_template
        u._browse_mtp_draft_clicked = self.browse_mtp_draft_model
        u.add_preset_btn.clicked.connect(self.add_preset)
        u.delete_preset_btn.clicked.connect(self.delete_preset)
        u.save_preset_btn.clicked.connect(self.save_preset)
        u.autotune.start_requested.connect(self.start_autotune)
        u.autotune.cancel_requested.connect(self.cancel_autotune)
        u.autotune.apply_requested.connect(self.apply_autotune_profile)
        u.autotune.open_results_requested.connect(self.open_autotune_results_folder)
        u.ctx_help_btn.clicked.connect(
            lambda: self._show_parameter_help(
                "Context Size Help", getattr(self, "_ctx_help_text", "")
            )
        )
        u.ncmoe_help_btn.clicked.connect(
            lambda: self._show_parameter_help(
                "CPU MoE Help", getattr(self, "_ncmoe_help_text", "")
            )
        )

        self.server.log_received.connect(
            lambda text, level: self.log_mgr.append(text, level)
        )
        self.server.state_changed.connect(self.update_action_buttons)
        self.server.bench_finished.connect(lambda _: self.update_action_buttons())

        # Парсинг логов для визуализации памяти
        self._mem_data = MemoryData()
        self._mem_viz_dirty = False
        self._mem_viz_timer = QTimer(self.ui)
        self._mem_viz_timer.setInterval(100)
        self._mem_viz_timer.setSingleShot(True)
        self._mem_viz_timer.timeout.connect(self._flush_mem_viz)
        self.server.log_received.connect(self._on_log_for_mem_viz)
        self.server.server_stopped.connect(self._on_server_stopped)
        self.server.bench_finished.connect(self._on_bench_finished)

    def _fmt_counter(self, value) -> str:
        # Без разделителей тысяч: пользователь счёл запятые сбивающими.
        return f"{max(int(value or 0), 0)}"

    def _server_metrics_url(self) -> str:
        host = str(self.ui.host.text() or "127.0.0.1").strip() or "127.0.0.1"
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{self.ui.port.value()}"

    def _tooltip_summary(self, text: str, max_lines: int = 4) -> str:
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[:max_lines]) + "\n\nClick ? for the full table."

    def _set_ctx_help_text(self, text: str) -> None:
        self._ctx_help_text = text
        self.ui.ctx_size.setToolTip(self._tooltip_summary(text))
        self.ui.ctx_help_btn.setEnabled(bool(text))

    def _set_ncmoe_help_text(self, text: str) -> None:
        self._ncmoe_help_text = text
        self.ui.cpu_moe_layers.setToolTip(self._tooltip_summary(text))
        self.ui.ncmoe_help_btn.setEnabled(bool(text))

    def _show_parameter_help(self, title: str, text: str) -> None:
        if not text:
            QMessageBox.information(self.ui, title, "No model-specific help available")
            return
        dialog = QDialog(self.ui)
        dialog.setWindowTitle(title)
        dialog.resize(760, 620)
        layout = QVBoxLayout(dialog)
        viewer = QTextEdit(readOnly=True)
        viewer.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        viewer.setPlainText(text)
        viewer.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(viewer)
        layout.addWidget(buttons)
        dialog.exec()

    def _set_table_item(
        self,
        table,
        row: int,
        col: int,
        text: object,
        data: object | None = None,
        tooltip: str = "",
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if data is not None:
            item.setData(Qt.ItemDataRole.UserRole, data)
        if tooltip:
            item.setToolTip(tooltip)
        table.setItem(row, col, item)
        return item

    def _selected_table_rows(self, table) -> list[int]:
        model = table.selectionModel()
        if not model:
            return []
        rows = sorted({index.row() for index in model.selectedRows()})
        if rows:
            return rows
        current = table.currentRow()
        return [current] if current >= 0 else []

    def _table_row_data(self, table, row: int):
        item = table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _selected_table_data(self, table) -> list:
        return [
            data
            for row in self._selected_table_rows(table)
            if (data := self._table_row_data(table, row))
        ]

    def _launch_preview(self) -> dict:
        model_path = self._current_model_path()
        info = self.ui.models_by_path.get(model_path) if model_path else {}
        if model_path and not info:
            try:
                info = extract_model_info(model_path)
                self.ui.models_by_path[model_path] = info
            except Exception:
                info = {}

        block_count = int((info or {}).get("block_count") or 0)
        ctx_value = int(self.ui.ctx_size.value() or 0)
        if ctx_value <= 0:
            ctx_value = int(
                (info or {}).get("recommended_ctx")
                or (info or {}).get("context_length")
                or 4096
            )
            ctx_text = f"auto -> {ctx_value:,}"
        else:
            ctx_text = f"{ctx_value:,}"

        if self.ui.gpu_layers_all.isChecked():
            gpu_layers = block_count or 999
            gpu_text = "all"
        elif self.ui.gpu_auto.isChecked():
            gpu_layers = block_count or 999
            resolved = str(block_count) if block_count else "all"
            gpu_text = f"auto -> {resolved}"
        else:
            gpu_layers = int(self.ui.gpu_layers.value() or 0)
            gpu_text = str(gpu_layers)

        slots_value = int(self.ui.parallel_slots.value() or 0)
        slots = max(slots_value, 1)
        slots_text = f"auto -> {slots}" if slots_value <= 0 else str(slots)
        ncmoe = max(int(self.ui.cpu_moe_layers.value() or 0), 0)
        mtp_on = bool(self.ui.speculative_mtp.isChecked())
        mtp_text = (
            f"ON · n={self.ui.spec_draft_n_max.value()} · p={self.ui.spec_draft_p_min.value():.2f}"
            if mtp_on
            else "off"
        )
        kv_text = f"{self.ui.cache_type_k.currentText()} / {self.ui.cache_type_v.currentText()}"
        estimate = None
        if model_path and info:
            try:
                estimate = full_vram_estimate(
                    info,
                    ctx_size=ctx_value,
                    gpu_layers=gpu_layers,
                    cache_type_k=self.ui.cache_type_k.currentText(),
                    cache_type_v=self.ui.cache_type_v.currentText(),
                    flash_attn=bool(self.ui.flash_attn.isChecked()),
                    parallel_slots=slots,
                    ncmoe=ncmoe,
                )
            except Exception:
                estimate = None
        return {
            "model_path": model_path,
            "model_name": self.ui.current_model_id() or "No model selected",
            "ctx_value": ctx_value,
            "ctx_text": ctx_text,
            "gpu_text": gpu_text,
            "kv_text": kv_text,
            "slots_text": slots_text,
            "mtp_text": mtp_text,
            "endpoint": f"{self._server_metrics_url()}/v1",
            "estimate": estimate,
            "missing_paths": self._missing_launch_paths(model_path),
        }

    def _missing_launch_paths(self, model_path: str | None) -> list[str]:
        """Referenced files that are configured but no longer on disk.

        Checked at every preview refresh (not just at launch) so a moved or
        deleted model/draft/template surfaces before the user clicks Start,
        instead of as an opaque llama-server startup failure.
        """
        missing: list[str] = []
        if not self._resolve_llamacpp_executable("server"):
            exe_text = self.ui.exe_path.text().strip()
            if exe_text:
                missing.append(f"llama-server executable: {exe_text}")
        if model_path and not os.path.isfile(model_path):
            missing.append(f"model file: {model_path}")
        if self.ui.speculative_mtp.isChecked():
            draft = self.ui.spec_draft_model_path.text().strip()
            if draft and not os.path.isfile(draft):
                missing.append(f"MTP draft GGUF: {draft}")
        if self.ui.use_chat_template.isChecked():
            tmpl = self.ui.chat_template_file.text().strip()
            if tmpl and not os.path.isfile(tmpl):
                missing.append(f"chat template: {tmpl}")
        return missing

    def _refresh_overview(self):
        overview_status = getattr(self.ui, "overview_status", None)
        if overview_status is None or type(overview_status).__module__.startswith(
            "unittest.mock"
        ):
            return

        preview = self._launch_preview()
        running = self.server.is_server_running()
        bench = self.server.is_bench_running()
        if self.launcher.is_pending:
            status_text = "◐ Restarting server..."
            status_color = STATUS_COLOR_PENDING
        elif running and getattr(self._mem_data, "server_ready", False):
            status_text = f"● READY · {preview['endpoint']}"
            status_color = STATUS_COLOR_READY
        elif running:
            status_text = f"◐ Loading · {preview['model_name']}"
            status_color = STATUS_COLOR_PENDING
        elif bench:
            status_text = "◐ Benchmark running"
            status_color = STATUS_COLOR_BENCH
        else:
            status_text = "○ Server stopped"
            status_color = STATUS_COLOR_MUTED_DARK
        self.ui.overview_status.setText(status_text)
        self.ui.overview_status.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {status_color};"
        )
        self.ui.overview_model.setText(preview["model_name"])

        snapshot = self.runtime_stats_snapshot()
        labels = snapshot.get("labels", {})
        speed = labels.get("speed", "Speed: -").replace("Speed:", "").strip() or "-"
        request = (
            labels.get("request", "Request: -").replace("Request:", "").strip() or "-"
        )
        active = (
            labels.get("active_time", "Work time: 0:00")
            .replace("Work time:", "")
            .strip()
            or "0:00"
        )
        tokens = snapshot.get("tokens", {})

        vram_total = self._mem_data.total("VRAM")
        vram_cap = self._mem_data.system_memory.get("VRAM")
        estimate = preview["estimate"]
        if vram_total > 0:
            if vram_cap:
                vram_text = f"{fmt_mem(vram_total, short=True)} / {fmt_mem(vram_cap, short=True)}"
                vram_detail = f"{self._mem_data.utilization('VRAM'):.1f}% VRAM"
            else:
                vram_text = fmt_mem(vram_total, short=True)
                vram_detail = "measured from llama.cpp/process counters"
        elif estimate:
            vram_text = f"{estimate.total_gib:.2f} GiB"
            vram_detail = "estimated before launch"
        else:
            vram_text = "-"
            vram_detail = "waiting for model or memory data"

        self.ui.overview_speed_value.setText(speed)
        self.ui.overview_speed_detail.setText("PP/TG from /slots or timings log")
        self.ui.overview_request_value.setText(request)
        self.ui.overview_request_detail.setText(
            f"task {tokens.get('task', 0)} · total {tokens.get('total', 0)}"
        )
        self.ui.overview_active_value.setText(active)
        self.ui.overview_active_detail.setText("PP + TG active time")
        self.ui.overview_vram_value.setText(vram_text)
        self.ui.overview_vram_detail.setText(vram_detail)
        self.ui.overview_context_value.setText(preview["ctx_text"])
        self.ui.overview_context_detail.setText(
            f"KV {preview['kv_text']} · slots {preview['slots_text']}"
        )
        self.ui.overview_endpoint_value.setText(preview["endpoint"])
        self.ui.overview_endpoint_detail.setText("OpenAI-compatible API")
        self.ui.overview_settings.setText(
            "Settings: "
            f"GPU {preview['gpu_text']} · MTP {preview['mtp_text']} · "
            f"KV {preview['kv_text']}"
        )

        # Постоянная полоса статуса поверх вкладок (всегда видна).
        if hasattr(self.ui, "status_indicator"):
            self.ui.status_indicator.setText("●" if running or bench else "○")
            self.ui.status_short.setText(status_text)
            self.ui.status_speed.setText(speed)
            self.ui.status_vram.setText(
                vram_text if (vram_total > 0 or estimate) else "—"
            )

        detailed = self._parsed_memory_without_process_fallback() > 0
        if vram_total <= 0 and not estimate:
            note = (
                "Detailed or fallback memory data will appear in Overview after launch."
            )
        elif detailed:
            note = "Overview is using detailed llama.cpp buffer breakdown."
        elif vram_total > 0:
            note = (
                "Detailed memory breakdown unavailable for this llama.cpp build; "
                "showing process/system fallback."
            )
        else:
            note = "Preflight uses heuristic VRAM estimation; measured data appears after launch."
        self.ui.overview_memory_note.setText(note)

        self.ui.preflight_model.setText(f"Model: {preview['model_name']}")
        self.ui.preflight_context.setText(f"Context: {preview['ctx_text']}")
        self.ui.preflight_kv.setText(f"KV: {preview['kv_text']}")
        self.ui.preflight_gpu.setText(f"GPU offload: {preview['gpu_text']}")
        self.ui.preflight_mtp.setText(f"MTP: {preview['mtp_text']}")
        self.ui.preflight_endpoint.setText(f"Endpoint: {preview['endpoint']}")
        if not preview["model_path"]:
            self.ui.preflight_status.setText(
                "Select a model to estimate launch readiness"
            )
            self.ui.preflight_status.setStyleSheet(
                "font-weight: bold; color: " + STATUS_COLOR_MUTED_DARK + ";"
            )
            self.ui.preflight_warning.setText("")
            return
        if preview["missing_paths"]:
            self.ui.preflight_status.setText("Missing files — launch will likely fail")
            self.ui.preflight_status.setStyleSheet(
                "font-weight: bold; color: " + STATUS_COLOR_ERROR + ";"
            )
            self.ui.preflight_warning.setText(
                "Missing: " + "; ".join(preview["missing_paths"])
            )
            return
        # VRAM capacity bar removed: llama.cpp reports VRAM only after the
        # model loads, so the pre-launch estimate was unreliable
        # ("GPU capacity not available"). Preflight shows launch readiness
        # without it.
        self.ui.preflight_status.setText("Ready to launch")
        self.ui.preflight_status.setStyleSheet(
            "font-weight: bold; color: " + STATUS_COLOR_READY + ";"
        )
        if estimate:
            self.ui.preflight_warning.setText(
                f"Estimated VRAM: {estimate.total_gib:.2f} GiB "
                f"(weights {estimate.model_vram_gib:.2f}, KV {estimate.kv_cache_gib:.2f})"
            )
        else:
            self.ui.preflight_warning.setText("Model metadata is incomplete.")

    def _on_log_speed_updated(self, text: str):
        # Скорость llama_print_timings из логов — приоритетный источник:
        # точный замер завершённого запроса. Дельты /slots занижают скорость
        # (теряют хвост генерации и включают время HTTP-опроса), поэтому
        # когда есть замер из логов — показываем именно его.
        self.ui.speed_label.setText(text)
        self._refresh_overview()

    def _on_log_timing_updated(self, pp_seconds: float, tg_seconds: float):
        self.stats.set_log_timing(pp_seconds, tg_seconds)
        self._refresh_overview()

    def _start_metrics_polling(self):
        self.metrics.set_url(self._server_metrics_url())
        self.stats.reset_server_scope()
        self.metrics.start()
        self.ui.speed_label.setText("Speed: waiting for /slots...")
        self.ui.request_tokens_label.setText("Request: -")
        self._refresh_overview()

    def _stop_metrics_polling(self):
        self.metrics.stop()
        self.ui.speed_label.setText("Speed: -")
        self.ui.request_tokens_label.setText("Request: -")
        self._refresh_overview()

    def _render_tokens_label(self, display: dict):
        self.ui.tokens_label.setText(
            "Tokens: "
            + stat_sep().join(
                [
                    stat_kv(
                        "total",
                        self._fmt_counter(display["total"]),
                        STAT_COLOR_TOTAL,
                    ),
                    stat_kv(
                        "task", self._fmt_counter(display["task"]), STAT_COLOR_TASK
                    ),
                    stat_kv(
                        "prompt",
                        self._fmt_counter(display["prompt"]),
                        STAT_COLOR_PROMPT,
                    ),
                    stat_kv(
                        "generated",
                        self._fmt_counter(display["generated"]),
                        STAT_COLOR_GENERATED,
                    ),
                ]
            )
        )

    def _plain_label_text(self, label) -> str:
        text = str(label.text() if label is not None else "")
        text = re.sub(r"<[^>]+>", "", text)
        return (
            text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
        )

    def runtime_stats_snapshot(self) -> dict:
        stats = self.stats.stats_snapshot()
        model_path = self._current_model_path()
        return {
            "schema_version": 1,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "server": {
                "running": self.server.is_server_running(),
                "base_url": self._server_metrics_url(),
            },
            "model": {
                "path": model_path,
                "id": self.ui.current_model_id(),
            },
            "tokens": stats["tokens"],
            "time_seconds": stats["time_seconds"],
            "labels": {
                "speed": self._plain_label_text(self.ui.speed_label),
                "tokens": self._plain_label_text(self.ui.tokens_label),
                "request": self._plain_label_text(self.ui.request_tokens_label),
                "saved": self._plain_label_text(self.ui.tokens_saved_label),
                "active_time": self._plain_label_text(self.ui.active_time_label),
                "current_time": self._plain_label_text(self.ui.current_time_label),
            },
        }

    def runtime_stats_markdown(self) -> str:
        return format_runtime_stats_markdown(self.runtime_stats_snapshot())

    def _time_row_html(self, caption: str, pp_s: float, tg_s: float) -> str:
        """HTML-строка времени: `Caption: total (Prompt pp | Gen tg)`."""
        inner = (
            f'<span style="color:{STAT_COLOR_CAPTION};">(</span>'
            + stat_sep().join(
                [
                    stat_kv("Prompt", format_duration(pp_s), STAT_COLOR_PROMPT),
                    stat_kv("Gen", format_duration(tg_s), STAT_COLOR_GENERATED),
                ]
            )
            + f'<span style="color:{STAT_COLOR_CAPTION};">)</span>'
        )
        total = pp_s + tg_s
        return stat_kv(caption, format_duration(total), STAT_COLOR_TIME) + " " + inner

    def _render_active_time_label(self, pp_s: float, tg_s: float):
        """Отрисовка total активного времени (за текущую сессию)."""
        self.ui.active_time_label.setText(self._time_row_html("Work time:", pp_s, tg_s))
        self._refresh_overview()

    def _render_current_time_label(self, pp_s: float, tg_s: float):
        """Отрисовка времени текущего/последнего запроса."""
        self.ui.current_time_label.setText(
            self._time_row_html("Last request:", pp_s, tg_s)
        )
        self._refresh_overview()

    def _on_slot_metrics_updated(self, slots):
        info = self.stats.update_slot_metrics(slots)
        if info is None:
            return
        if not info["visible"]:
            if not self.log_mgr.has_speed:
                self.ui.speed_label.setText("Speed: -")
            self.ui.request_tokens_label.setText("Request: -")
            self._refresh_overview()
            return

        parts = []
        if info["prompt_speed"] > 0:
            parts.append(
                stat_kv(
                    "Prompt",
                    f"{format_speed(info['prompt_speed'])} tok/s",
                    STAT_COLOR_PROMPT,
                )
            )
        if info["predicted_speed"] > 0:
            parts.append(
                stat_kv(
                    "Gen",
                    f"{format_speed(info['predicted_speed'])} tok/s",
                    STAT_COLOR_GENERATED,
                )
            )
        if not self.log_mgr.has_speed:
            self.ui.speed_label.setText(
                "Speed: " + (stat_sep().join(parts) if parts else "-")
            )
        if info["prompt_tokens"] or info["predicted_tokens"]:
            self.ui.request_tokens_label.setText(
                "Request: "
                + stat_sep().join(
                    [
                        stat_kv(
                            "prompt",
                            self._fmt_counter(info["prompt_tokens"]),
                            STAT_COLOR_PROMPT,
                        ),
                        stat_kv(
                            "generated",
                            self._fmt_counter(info["predicted_tokens"]),
                            STAT_COLOR_GENERATED,
                        ),
                    ]
                )
            )
        else:
            self.ui.request_tokens_label.setText("Request: -")

    def _on_server_metrics_updated(self, metrics):
        self.stats.update_server_metrics(metrics)

    def reset_task_tokens(self):
        """Сохранить текущую задачу в Saved и начать отсчёт новой с нуля.

        Обнуляет task-счётчик, Current time и Request. Total-токены и Active
        время (server-scope) не трогает — для них есть Reset session.
        """
        task_total = self.stats.reset_task()
        self.log_mgr.reset_runtime_extractors(reset_speed=False, reset_timing=True)
        self.ui.request_tokens_label.setText("Request: -")
        self.log_mgr.append(
            f"Token counter reset: saved {task_total} tokens, "
            "Current time and Request reset"
        )

    def reset_session(self):
        """Обнулить все живые счётчики сессии (total/task, время, Request).

        Saved-история сохраняется. Реализовано через baseline-смещения,
        поэтому следующий опрос /metrics не вернёт старые значения на экран.
        """
        self.stats.reset_session()
        self.log_mgr.reset_runtime_extractors(reset_speed=True, reset_timing=True)
        self.ui.request_tokens_label.setText("Request: -")
        self.log_mgr.append("Session reset: tokens and time zeroed")

    def reset_saved_total(self):
        """Обнулить накопленную Saved-историю (last и total)."""
        self.stats.reset_saved()
        self.log_mgr.append("Saved history reset")

    def _render_saved_label(self, last: int, total: int):
        self.ui.tokens_saved_label.setText(
            "Saved: "
            + stat_sep().join(
                [
                    stat_kv("last", self._fmt_counter(last), STAT_COLOR_SAVED),
                    stat_kv("total", self._fmt_counter(total), STAT_COLOR_SAVED),
                ]
            )
        )
        self._refresh_overview()

    def export_runtime_stats(self):
        default_name = f"runtime-stats-{time.strftime('%Y%m%d-%H%M%S')}.json"
        file_name, _ = QFileDialog.getSaveFileName(
            self.ui,
            "Export runtime stats",
            str(Path.cwd() / default_name),
            "JSON (*.json);;All files (*.*)",
        )
        if not file_name:
            return

        path = Path(file_name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    self.runtime_stats_snapshot(), f, ensure_ascii=False, indent=2
                )
                f.write("\n")
        except OSError as exc:
            QMessageBox.critical(self.ui, "Export failed", str(exc))
            return

        self.log_mgr.append(f"Runtime stats exported: {path}")

    def copy_runtime_stats_markdown(self):
        QApplication.clipboard().setText(self.runtime_stats_markdown())
        self.log_mgr.append("Runtime stats copied to clipboard as Markdown")

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
            lambda r: (
                self.ui.hide()
                if self.ui.isVisible() and r == QSystemTrayIcon.DoubleClick
                else self.ui.showNormal()
            )
        )
        self.tray.show()

    def _shutdown_warn(self, message: str):
        try:
            self.log_mgr.append(message, "warn")
        except Exception:
            pass

    def _stop_qthread(self, thread, name: str, timeout_ms: int, stop=None) -> bool:
        if thread is None:
            return True
        try:
            running = thread.isRunning()
        except RuntimeError:
            return True
        if not running:
            return True
        try:
            if stop:
                stop(thread)
            else:
                thread.requestInterruption()
        except RuntimeError:
            return True
        except Exception as exc:
            self._shutdown_warn(f"{name}: stop request failed: {exc}")
        try:
            stopped = thread.wait(timeout_ms)
        except RuntimeError:
            return True
        if not stopped:
            self._shutdown_warn(
                f"{name}: still running after {timeout_ms} ms; waiting before exit"
            )
            try:
                thread.wait()
                stopped = True
            except RuntimeError:
                stopped = True
        return bool(stopped)

    def _dispose_qthread_attr(
        self, attr_name: str, name: str, timeout_ms: int = 5000, stop=None
    ) -> bool:
        thread = getattr(self, attr_name, None)
        if thread is None:
            return True
        stopped = self._stop_qthread(thread, name, timeout_ms, stop=stop)
        if stopped:
            try:
                thread.deleteLater()
            except RuntimeError:
                pass
            if getattr(self, attr_name, None) is thread:
                setattr(self, attr_name, None)
        return stopped

    def _stop_hf_downloaders_for_shutdown(self):
        workers = []
        for key, task in list(self.hf.tasks().items()):
            worker = task.get("worker")
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    task["status"] = "pausing"
                    worker.pause()
            except RuntimeError:
                task["worker"] = None
                continue
            workers.append((key, task, worker))

        for key, task, worker in workers:
            stopped = self._stop_qthread(
                worker,
                f"Hugging Face download {key}",
                65000,
                stop=lambda thread: thread.pause(),
            )
            if stopped:
                try:
                    worker.deleteLater()
                except RuntimeError:
                    pass
                task["worker"] = None

    def shutdown_background_work(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        self.launcher.cancel_pending()
        try:
            self.save_settings()
            self.ui.save_ui_state()
        except Exception as exc:
            self._shutdown_warn(f"Shutdown settings save failed: {exc}")
        try:
            self.metrics.stop()
        except Exception as exc:
            self._shutdown_warn(f"Metrics shutdown failed: {exc}")
        self._dispose_qthread_attr(
            "autotune",
            "AutoTune",
            10000,
            stop=lambda thread: thread.cancel(),
        )
        self._dispose_qthread_attr("scanner", "model scanner", 10000)
        self._dispose_qthread_attr("hf_scanner", "Hugging Face scanner", 35000)
        self._dispose_qthread_attr("updater", "llama.cpp updater", 65000)
        self._stop_hf_downloaders_for_shutdown()
        try:
            self.server.terminate_all()
        except Exception as exc:
            self._shutdown_warn(f"Server shutdown failed: {exc}")
        try:
            self.log_mgr.stop()
        except Exception:
            pass
        if hasattr(self, "tray"):
            self.tray.hide()

    def save_settings(self):
        self.auto_detect_bench()
        self.config.read_from_ui(self.ui)
        self.config.settings.model_cache = self.ui.models
        self.config.save()

    def _selected_cuda_version(self) -> str:
        return str(self.ui.cuda_version_combo.currentData() or "12")

    def _update_cuda_status(self):
        label = getattr(self.ui, "cuda_status_label", None)
        if label is None:
            return

        cuda_ver = self._selected_cuda_version()
        exe = self._resolve_llamacpp_executable("server")
        raw_path = self.ui.exe_path.text().strip().strip('"')
        model_path = self._current_model_path()

        if exe:
            build_name = Path(exe).parent.name
            parts = [f"CUDA {cuda_ver}: {build_name}"]
            color = STATUS_COLOR_RUNNING
        elif raw_path:
            parts = [f"CUDA {cuda_ver}: build not found"]
            color = STATUS_COLOR_WARNING
        else:
            parts = [f"CUDA {cuda_ver}: select llama.cpp folder"]
            color = STATUS_COLOR_MUTED

        parts.append("model selected" if model_path else "select model")

        update_info = self._llamacpp_update_info.get(cuda_ver)
        if (
            update_info
            and update_info.get("current") is not None
            and update_info["current"] < update_info["latest"]
        ):
            parts.append(f"update available: b{update_info['current']} → b{update_info['latest']}")
            color = STATUS_COLOR_WARNING

        if cuda_ver == "13":
            note = "CUDA 13 requires NVIDIA driver 580+; best for RTX 50/Blackwell."
        else:
            note = "CUDA 12 is the safer default for RTX 30/40 and older stable builds."

        label.setText(" | ".join(parts))
        label.setToolTip(note)
        label.setStyleSheet(f"color: {color};")
        self._refresh_feature_detection(exe)

    _FEATURE_DETECT_SEP = "\n\n[!] "

    def _refresh_feature_detection(self, exe: str) -> None:
        """Re-probe --help only when the resolved binary actually changed.

        update_cli_preview -> _update_cuda_status runs on nearly every param
        edit; caching by resolved exe path keeps this to one subprocess call
        per binary switch instead of one per keystroke.
        """
        if exe == getattr(self, "_last_probed_exe", None):
            return
        self._last_probed_exe = exe
        self._supported_flags = probe_supported_flags(exe) if exe else None
        self._apply_feature_detection()

    def _apply_feature_detection(self) -> None:
        supported = getattr(self, "_supported_flags", None)
        exe_name = Path(self._last_probed_exe).name if self._last_probed_exe else "this build"
        sep = self._FEATURE_DETECT_SEP
        for spec in PARAM_REGISTRY:
            if not spec.widget_attr or not spec.cli_flags:
                continue
            widget = getattr(self.ui, spec.widget_attr, None)
            if widget is None:
                continue
            base_tip = widget.toolTip().split(sep, 1)[0]
            if is_spec_supported(spec, supported):
                if widget.toolTip() != base_tip:
                    widget.setToolTip(base_tip)
                widget.setStyleSheet("")
            else:
                flags = "/".join(spec.cli_flags)
                widget.setToolTip(
                    f"{base_tip}{sep}Not found in `{exe_name} --help` — "
                    f"this llama-server build may not support {flags}."
                )
                widget.setStyleSheet("background-color: #4a3a1a;")

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

    def _current_perf_preset_name(self) -> str:
        combo = getattr(self.ui, "preset_name_combo", None)
        if combo is None:
            return "default"
        return combo.currentText().strip() or "default"

    def _update_preset_buttons(self):
        delete_btn = getattr(self.ui, "delete_preset_btn", None)
        if delete_btn is None:
            return
        delete_btn.setEnabled(self._current_perf_preset_name() != "default")

    def _refresh_perf_preset_names(self, selected_name: str = ""):
        combo = getattr(self.ui, "preset_name_combo", None)
        if combo is None:
            return

        selected_name = (selected_name or combo.currentText()).strip() or "default"
        model_path = self._current_model_path()
        names = (
            self.config.list_perf_preset_names(model_path)
            if model_path
            else ["default"]
        )
        if selected_name not in names:
            names.append(selected_name)

        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(names)
            combo.setCurrentText(selected_name)
        finally:
            combo.blockSignals(False)
        self._update_preset_buttons()

    def _on_preset_selected(self):
        if getattr(self, "_loading_preset", False) or self.ui.loading_profile:
            return

        model_path = self._current_model_path()
        if not model_path:
            return

        preset_name = self._current_perf_preset_name()
        if self._try_load_perf_preset(
            model_path,
            self.ui.ctx_size.value(),
            preset_name=preset_name,
        ):
            self._mark_restart_needed()
            return

        if hasattr(self.ui, "preset_status"):
            self.ui.preset_status.setText(f"Preset: none ({preset_name})")
            self.ui.preset_status.setStyleSheet("color: " + STATUS_COLOR_MUTED + ";")
        self._update_preset_buttons()

    def add_preset(self):
        model_path = self._current_model_path()
        if not model_path:
            QMessageBox.warning(self.ui, "Error", "Select a model first")
            return

        name, ok = QInputDialog.getText(
            self.ui,
            "Add preset",
            "Preset name:",
        )
        if not ok:
            return

        preset_name = str(name or "").strip()
        if not preset_name:
            QMessageBox.warning(self.ui, "Error", "Preset name cannot be empty")
            return
        if preset_name.casefold() == "default":
            QMessageBox.warning(self.ui, "Error", "Default preset already exists")
            return

        if self.ui.preset_name_combo.findText(preset_name) < 0:
            self.ui.preset_name_combo.addItem(preset_name)
        self.ui.preset_name_combo.setCurrentText(preset_name)
        self._update_preset_buttons()
        if hasattr(self.ui, "preset_status"):
            self.ui.preset_status.setText(f"Preset: new {preset_name}, not saved")
            self.ui.preset_status.setStyleSheet("color: " + STATUS_COLOR_WARNING + ";")

    def delete_preset(self):
        model_path = self._current_model_path()
        if not model_path:
            QMessageBox.warning(self.ui, "Error", "Select a model first")
            return

        preset_name = self._current_perf_preset_name()
        if preset_name == "default":
            QMessageBox.warning(self.ui, "Error", "Default preset cannot be deleted")
            return

        if not confirm_destructive_action(
            self.ui, "Delete preset", f"Delete preset '{preset_name}'?"
        ):
            return

        if not self.config.delete_perf_preset(model_path, preset_name):
            QMessageBox.warning(self.ui, "Error", "Preset was not found")
            return

        self.log_mgr.append(
            f"Preset deleted: {preset_name} | {os.path.basename(model_path)}"
        )
        self._refresh_perf_preset_names("default")
        if hasattr(self.ui, "preset_status"):
            self.ui.preset_status.setText(f"Preset: deleted {preset_name}")
            self.ui.preset_status.setStyleSheet("color: " + STATUS_COLOR_WARNING + ";")

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
            preset_name = self._current_perf_preset_name()
            self.config.save_perf_preset(
                model_path,
                ctx,
                self.ui,
                preset_name=preset_name,
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
            f"Preset saved: {preset_name} | {os.path.basename(model_path)} | ctx={ctx:,}"
        )
        self._refresh_perf_preset_names(preset_name)
        if hasattr(self.ui, "preset_status"):
            self.ui.preset_status.setText(f"Preset: saved {preset_name} | ctx={ctx:,}")
            self.ui.preset_status.setStyleSheet("color: " + STATUS_COLOR_RUNNING + ";")
        QMessageBox.information(
            self.ui,
            "Saved",
            f"Parameters for preset '{preset_name}' saved.",
        )

    def _try_load_perf_preset(
        self,
        model_path: str,
        ctx_size: int,
        preset_name: str = "",
    ) -> bool:
        if not model_path:
            return False

        if getattr(self, "_loading_preset", False):
            return False

        preset_name = (
            preset_name or self._current_perf_preset_name()
        ).strip() or "default"
        if preset_name == "default" and ctx_size <= 0:
            return False

        self._loading_preset = True
        try:
            loaded = self.config.load_perf_preset(
                model_path,
                ctx_size,
                self.ui,
                preset_name=preset_name,
            )
        finally:
            self._loading_preset = False

        if not loaded:
            return False

        loaded_ctx = self.ui.ctx_size.value()
        self.log_mgr.append(
            f"Loaded preset: {preset_name} | {os.path.basename(model_path)} | ctx={loaded_ctx:,}"
        )

        if hasattr(self.ui, "preset_status"):
            self.ui.preset_status.setText(
                f"Preset: loaded {preset_name} | ctx={loaded_ctx:,}"
            )
            self.ui.preset_status.setStyleSheet("color: " + STATUS_COLOR_RUNNING + ";")

        info = self.ui.models_by_path.get(model_path)
        if info:
            self._refresh_tooltips(info)

        self.update_cli_preview()
        return True

    def _mark_preset_modified(self):
        label = getattr(self.ui, "preset_status", None)
        if label is None:
            return
        text = label.text()
        if text.startswith("Preset: loaded") or text.startswith("Preset: saved"):
            label.setText("Preset: modified")
            label.setStyleSheet("color: " + STATUS_COLOR_WARNING + ";")

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
            self._set_mtp_manual_draft_path(f)
            self.ui.spec_draft_model_path.setText(f)
            self.ui.speculative_mtp.setChecked(True)
            self.save_settings()

    def browse_chat_template(self):
        start_dir = self.ui.model_dir.text().strip() or ""
        current = self.ui.chat_template_file.text().strip()
        if current:
            start_dir = str(Path(current).parent)
        f, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Select chat template (.jinja)",
            start_dir,
            "Jinja (*.jinja);;Text (*.txt);;All files (*.*)",
        )
        if f:
            self.ui.chat_template_file.setText(f)
            self.ui.use_chat_template.setChecked(True)
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
            lambda: (
                self.ui.scan_btn.setText("Scan")
                or self.ui.scan_progress.setVisible(False)
            )
        )
        self.scanner.start()

    def _update_local_model_delete_button(self):
        self.ui.local_models_delete_btn.setEnabled(
            bool(self._selected_table_rows(self.ui.local_models_list))
        )

    def refresh_local_model_manager(self, silent=False):
        self.ui.local_models_list.setRowCount(0)
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
            row = self.ui.local_models_list.rowCount()
            self.ui.local_models_list.insertRow(row)
            tooltip = str(entry.get("path") or "")
            self._set_table_item(
                self.ui.local_models_list,
                row,
                0,
                entry.get("relative") or "",
                data=entry,
                tooltip=tooltip,
            )
            self._set_table_item(
                self.ui.local_models_list, row, 1, entry.get("type") or ""
            )
            self._set_table_item(
                self.ui.local_models_list, row, 2, entry.get("gguf_count") or 0
            )
            self._set_table_item(
                self.ui.local_models_list, row, 3, entry.get("size_text") or ""
            )
            self._set_table_item(self.ui.local_models_list, row, 4, examples)

        self.ui.local_models_delete_btn.setEnabled(False)
        if entries:
            self.ui.local_models_status.setText(
                f"Local models: {len(entries)}, total {info.get('total_size_text')} | root: {info.get('root')}"
            )
        elif not silent:
            self.ui.local_models_status.setText(
                f"No local GGUF models found in {model_dir}"
            )

    def _selected_local_model_entry(self):
        rows = self._selected_table_rows(self.ui.local_models_list)
        if not rows:
            return None
        return self._table_row_data(self.ui.local_models_list, rows[0])

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
        if self._active_hf_downloads():
            QMessageBox.warning(
                self.ui,
                "Delete local model",
                "Pause or cancel active Hugging Face downloads before deleting local models.",
            )
            return

        model_dir = self.ui.model_dir.text().strip()
        target = Path(str(entry.get("path") or ""))
        base = Path(model_dir) if model_dir else Path()
        if (
            not target.exists()
            or not model_dir
            or not self._path_is_inside(target, base)
        ):
            QMessageBox.warning(
                self.ui,
                "Delete local model",
                "Selected path is invalid or outside Models folder.",
            )
            self.refresh_local_model_manager(silent=True)
            return
        if target.resolve() == base.resolve():
            QMessageBox.warning(
                self.ui,
                "Delete local model",
                "Refusing to delete the Models root folder.",
            )
            return
        if self.server.is_server_running() and self._current_model_uses_path(target):
            QMessageBox.warning(
                self.ui,
                "Delete local model",
                "This model is currently loaded. Stop the server first to release RAM/VRAM, then delete it.",
            )
            return

        delete_kind = "folder" if target.is_dir() else "file"
        if not confirm_destructive_action(
            self.ui,
            "Delete local model",
            f"Delete selected local model {delete_kind}?\n\n"
            f"{target}\n\n"
            "This cannot be undone. For folders, all GGUF/mmproj/.part files inside are removed.",
        ):
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
            self.ui.hf_status.setText("Cancelling Hugging Face request...")
            return

        repo = self.ui.hf_repo.text().strip()
        if not repo:
            QMessageBox.warning(self.ui, "Hugging Face", "Paste a repo id or model URL")
            return

        self.save_settings()
        self.ui.hf_files.setRowCount(0)
        self.ui.hf_progress.setValue(0)
        self.ui.hf_download_btn.setEnabled(False)
        self.ui.hf_progress.setVisible(True)
        self.ui.hf_progress.setRange(0, 0)
        self.ui.hf_status.setText("Scanning Hugging Face...")
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
        self.ui.hf_files.setRowCount(0)
        files = result.get("files") or []
        projectors = result.get("projectors") or []
        partial_count = 0
        for file_info in files:
            partial = self._hf_partial_info(file_info)
            if partial:
                partial_count += 1
                self._upsert_hf_partial_task(
                    result.get("repo_id") or "",
                    file_info,
                    partial,
                    model_dir=self.ui.model_dir.text().strip(),
                )
            self._add_hf_file_row(file_info, partial)

        if files:
            self.ui.hf_files.setCurrentCell(0, 0)

        target_text = ""
        model_dir = self.ui.model_dir.text().strip()
        if model_dir:
            target_text = (
                f" → {lmstudio_repo_dir(Path(model_dir), result.get('repo_id', ''))}"
            )
        projector = self._select_hf_projector()
        projector_text = f", vision: {projector.get('name')}" if projector else ""
        total_size = sum(int(f.get("size") or 0) for f in files)
        total_text = f", shown size: {format_bytes(total_size)}" if total_size else ""
        partial_text = f", partial/resumable: {partial_count}" if partial_count else ""
        self.ui.hf_status.setText(
            f"Found GGUF: {len(files)} of {len(result.get('all_files') or [])}"
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

    def _add_hf_file_row(self, file_info, partial=None):
        row = self.ui.hf_files.rowCount()
        self.ui.hf_files.insertRow(row)
        name = file_info.get("name") or file_info.get("rfilename") or ""
        quant = str(file_info.get("quant") or "").strip()
        size = int(file_info.get("size") or 0)
        tooltip = f"Partial file: {partial.get('partial_path')}" if partial else ""
        progress = ""
        if partial:
            progress = (
                f"{partial.get('partial_size_text')} / "
                f"{partial.get('expected_size_text')}"
            )
        self._set_table_item(
            self.ui.hf_files, row, 0, name, data=file_info, tooltip=tooltip
        )
        self._set_table_item(self.ui.hf_files, row, 1, quant)
        self._set_table_item(
            self.ui.hf_files, row, 2, format_bytes(size) if size else "size unknown"
        )
        self._set_table_item(self.ui.hf_files, row, 3, progress)

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
        if not model_dir:
            return
        try:
            partials = list_all_partial_downloads(Path(model_dir))
        except Exception:
            return
        if not partials:
            return
        for partial in partials:
            file_info = {
                "name": partial.get("name") or "",
                "rfilename": partial.get("rfilename") or "",
                "size": 0,
            }
            self._upsert_hf_partial_task(
                partial.get("repo_id") or "", file_info, partial, model_dir=model_dir
            )
        total = sum(int(p.get("partial_size") or 0) for p in partials)
        self.ui.hf_status.setText(
            f"Unfinished downloads: {len(partials)}, saved {format_bytes(total)}. "
            "They are already listed; Scan HF will add the full size and exact percent."
        )
        self._update_hf_download_button()

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
        self.ui.hf_local_files.setRowCount(0)
        model_dir = self.ui.model_dir.text().strip()
        repo_id = self._current_hf_repo_id()
        if not model_dir or not repo_id:
            self.ui.hf_delete_local_folder_btn.setEnabled(False)
            if not silent:
                self.ui.hf_status.setText(
                    "Local files: specify Models folder and HF repo"
                )
            return

        info = list_local_repo_files(Path(model_dir), repo_id)
        files = info.get("files") or []
        for file_info in files:
            marker = "partial" if file_info.get("is_partial") else "local"
            row = self.ui.hf_local_files.rowCount()
            self.ui.hf_local_files.insertRow(row)
            tooltip = str(file_info.get("path") or "")
            self._set_table_item(
                self.ui.hf_local_files,
                row,
                0,
                file_info.get("relative") or "",
                tooltip=tooltip,
            )
            self._set_table_item(self.ui.hf_local_files, row, 1, marker)
            self._set_table_item(
                self.ui.hf_local_files, row, 2, file_info.get("size_text") or ""
            )
        self.ui.hf_delete_local_folder_btn.setEnabled(bool(info.get("exists")))
        if files:
            self.ui.hf_status.setText(
                f"Local folder: {info.get('root')} | files: {len(files)}, total {info.get('total_size_text')}"
            )
        elif info.get("exists"):
            self.ui.hf_status.setText(
                f"Local folder exists but is empty: {info.get('root')}"
            )
        elif not silent:
            self.ui.hf_status.setText(f"Local folder not found: {info.get('root')}")

    def delete_hf_local_folder(self):
        repo_id = self._current_hf_repo_id()
        if self._active_hf_downloads(repo_id):
            QMessageBox.warning(
                self.ui,
                "Hugging Face",
                "Stop all downloads for this repository before deleting local files",
            )
            return
        model_dir = self.ui.model_dir.text().strip()
        if not model_dir or not repo_id:
            return
        target_root = lmstudio_repo_dir(Path(model_dir), repo_id)
        if not target_root.exists():
            self.refresh_hf_local_files(silent=True)
            return
        if not confirm_destructive_action(
            self.ui,
            "Delete local model folder",
            "Delete the whole local model folder, including main GGUF, vision/mmproj and .part?\n\n"
            f"{target_root}",
        ):
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
        selected = self._selected_table_data(self.ui.hf_files)
        if not selected:
            return None
        return selected[0]

    def _selected_hf_partial_info(self):
        file_info = self._selected_hf_file_info()
        return self._hf_partial_info(file_info) if file_info else {}

    def _update_hf_download_button(self):
        selected_files = self._selected_table_data(self.ui.hf_files)
        has_selection = bool(selected_files)
        has_partial = bool(self._selected_hf_partial_info())
        count = len(selected_files)
        self.ui.hf_download_btn.setText(
            f"{'Resume' if has_partial else 'Download'} selected models"
            + (f" ({count})" if count > 1 else "")
        )
        self.ui.hf_download_btn.setEnabled(has_selection)

        selected_task_keys = self._selected_hf_task_keys()
        selected_running = any(self.hf.is_running(key) for key in selected_task_keys)
        self.ui.hf_pause_btn.setEnabled(selected_running)
        self.ui.hf_cancel_btn.setEnabled(
            bool(selected_task_keys) or (has_partial and not selected_running)
        )

    @staticmethod
    def _hf_task_key(repo_id, file_info):
        return HfDownloadCoordinator.task_key(repo_id, file_info)

    def _selected_hf_task_keys(self):
        return [str(data) for data in self._selected_table_data(self.ui.hf_downloads)]

    def _active_hf_downloads(self, repo_id=None):
        return self.hf.active(repo_id)

    def _set_hf_task_display(self, task_key):
        task = self.hf.task(task_key)
        if not task:
            return
        row = task.get("row")
        if row is None or row < 0:
            return
        name = task.get("name") or task_key
        percent_value = task.get("percent")
        status = task.get("status") or "queued"
        message = str(task.get("message") or "").strip()
        parts = self._parse_hf_task_message(message)
        size_text = parts.get("size") or ""
        speed_text = parts.get("speed") or ""
        eta_text = parts.get("eta") or ""
        if not size_text and message:
            size_text = message.splitlines()[0]
        self._set_table_item(
            self.ui.hf_downloads, row, 0, name, data=task_key, tooltip=message
        )
        self._set_table_item(self.ui.hf_downloads, row, 1, status, tooltip=message)
        self._set_table_item(self.ui.hf_downloads, row, 3, size_text, tooltip=message)
        self._set_table_item(self.ui.hf_downloads, row, 4, speed_text, tooltip=message)
        self._set_table_item(self.ui.hf_downloads, row, 5, eta_text, tooltip=message)
        progress = self.ui.hf_downloads.cellWidget(row, 2)
        if not isinstance(progress, QProgressBar):
            progress = QProgressBar()
            progress.setTextVisible(True)
            self.ui.hf_downloads.setCellWidget(row, 2, progress)
        if percent_value is None:
            progress.setRange(0, 0)
            progress.setFormat("unknown")
        else:
            progress.setRange(0, 100)
            progress.setValue(max(0, min(int(percent_value), 100)))
            progress.setFormat(f"{int(percent_value)}%")

    def _parse_hf_task_message(self, message: str) -> dict[str, str]:
        text = str(message or "")
        parts: dict[str, str] = {}
        if m := re.search(r":\s*([^;]+?/\s*[^;,]+)", text):
            parts["size"] = m.group(1).strip()
        elif m := re.search(r"(?:Сохранено|Saved):\s*([^\n]+)", text):
            parts["size"] = m.group(1).strip()
        if m := re.search(r"(?:скорость|speed)\s+([^;\n,]+)", text, re.IGNORECASE):
            parts["speed"] = m.group(1).strip()
        if m := re.search(r"ETA\s+([^;\n,]+)", text, re.IGNORECASE):
            parts["eta"] = m.group(1).strip()
        return parts

    def _upsert_hf_partial_task(self, repo_id, file_info, partial, model_dir=None):
        """Show a saved .part immediately, even before its repository is scanned."""

        def ensure_row() -> int:
            row = self.ui.hf_downloads.rowCount()
            self.ui.hf_downloads.insertRow(row)
            return row

        task_key = self.hf.upsert_partial(
            repo_id,
            file_info,
            partial,
            model_dir or self.ui.model_dir.text().strip(),
            ensure_row=ensure_row,
        )
        if task_key:
            self._set_hf_task_display(task_key)

    def _refresh_hf_download_summary(self):
        active = self._active_hf_downloads()
        if active:
            average = int(
                sum(int(task.get("percent") or 0) for task in active) / len(active)
            )
            self.ui.hf_progress.setRange(0, 100)
            self.ui.hf_progress.setValue(average)
            self.ui.hf_progress.setVisible(True)
            self.ui.hf_status.setText(
                f"Parallel downloads: {len(active)} | total progress ~{average}%"
            )
        elif self.hf.tasks():
            self.ui.hf_progress.setVisible(False)
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
        projectors.sort(
            key=lambda item: (
                int(item.get("size") or 0),
                str(item.get("name") or "").lower(),
            )
        )
        return projectors[0]

    def download_hf_selection(self):
        if not self.hf_scan_result:
            QMessageBox.warning(self.ui, "Hugging Face", "Scan the repository first")
            return
        selected = self._selected_table_data(self.ui.hf_files)
        if not selected:
            QMessageBox.warning(
                self.ui, "Hugging Face", "Select a GGUF file to download"
            )
            return
        model_dir = self.ui.model_dir.text().strip()
        if not model_dir:
            QMessageBox.warning(self.ui, "Hugging Face", "Set the Models base folder")
            return

        files = list(selected)
        if self.ui.hf_include_mmproj.isChecked():
            projector = self._select_hf_projector()
            selected_names = {
                str(file_info.get("rfilename") or file_info.get("name") or "")
                for file_info in files
            }
            projector_name = str(
                (projector or {}).get("rfilename")
                or (projector or {}).get("name")
                or ""
            )
            if projector and projector_name not in selected_names:
                files.append(projector)

        repo_id = self.hf_scan_result.get("repo_id") or ""
        files = [
            file_info
            for file_info in files
            if file_info
            and not self.hf.is_running(self._hf_task_key(repo_id, file_info))
        ]
        if not files:
            self.ui.hf_status.setText("All selected files are already downloading")
            return
        target_root = lmstudio_repo_dir(Path(model_dir), repo_id)
        total_size = sum(int(f.get("size") or 0) for f in files)
        names = "\n".join(f"• {self._hf_file_display(f)}" for f in files)
        size_line = f"\nTotal: {format_bytes(total_size)}" if total_size else ""
        reply = QMessageBox.question(
            self.ui,
            "Download GGUF models",
            f"Start {len(files)} parallel downloads into the LM Studio-compatible folder:\n"
            f"{target_root}\n\n{names}{size_line}",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.ui.hf_downloads.clearSelection()
        for file_info in files:
            self._start_hf_download_task(repo_id, file_info, model_dir)
        self._refresh_hf_download_summary()

    def _start_hf_download_task(self, repo_id, file_info, model_dir):
        def ensure_row() -> int:
            row = self.ui.hf_downloads.rowCount()
            self.ui.hf_downloads.insertRow(row)
            return row

        task_key = self.hf.start(repo_id, file_info, model_dir, ensure_row=ensure_row)
        self._set_hf_task_display(task_key)
        row = self.hf.task(task_key).get("row")
        selection = self.ui.hf_downloads.selectionModel()
        if selection:
            selection.select(
                self.ui.hf_downloads.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows,
            )

    def pause_hf_download(self):
        paused = self.hf.pause(self._selected_hf_task_keys())
        if paused:
            self.ui.hf_status.setText(
                f"Pausing {paused} download(s): saving .part for resume..."
            )

    def cancel_hf_download(self):
        selected_keys = self._selected_hf_task_keys()
        running_keys = self.hf.running_keys(selected_keys)
        if running_keys:
            if not confirm_destructive_action(
                self.ui,
                "Cancel downloads",
                f"Abort selected downloads ({len(running_keys)}) and delete their .part files?\n\n"
                "Use Pause instead of Cancel to resume later.",
            ):
                return
            for key in running_keys:
                self.hf.cancel_and_delete([key])
            self.ui.hf_status.setText(
                f"Cancelling {len(running_keys)} download(s): deleting partial .part files..."
            )
            return

        saved_partials = []
        for key in selected_keys:
            task = self.hf.task(key) or {}
            file_info = task.get("file_info") or {}
            filename = str(file_info.get("rfilename") or file_info.get("name") or "")
            if not filename:
                continue
            partial_info = partial_download_info(
                Path(task.get("model_dir") or self.ui.model_dir.text().strip()),
                str(task.get("repo_id") or ""),
                filename,
                int(file_info.get("size") or 0),
            )
            if partial_info:
                saved_partials.append((key, partial_info))
        if saved_partials:
            if not confirm_destructive_action(
                self.ui,
                "Delete partial downloads",
                f"Delete saved .part files of the selected tasks ({len(saved_partials)})?",
            ):
                return
            for key, partial_info in saved_partials:
                delete_file_safely(Path(partial_info.get("partial_path") or ""))
                self.hf.mark_partial_deleted(key)
            self.ui.hf_status.setText(
                f"Deleted partial downloads: {len(saved_partials)}"
            )
            self.refresh_hf_local_files(silent=True)
            self._update_hf_download_button()
            return

        partial = self._selected_hf_partial_info()
        if not partial:
            return
        if not confirm_destructive_action(
            self.ui,
            "Cancel partial download",
            "Delete the saved .part and restart this file on the next download?\n\n"
            f"{partial.get('partial_path')}\n"
            f"Saved: {partial.get('partial_size_text')}",
        ):
            return
        delete_file_safely(Path(partial.get("partial_path") or ""))
        item = self.ui.hf_files.currentItem()
        file_info = self._selected_hf_file_info()
        if file_info:
            row = self.ui.hf_files.currentRow()
            if row >= 0:
                self._set_table_item(
                    self.ui.hf_files,
                    row,
                    0,
                    file_info.get("name") or file_info.get("rfilename") or "",
                    data=file_info,
                )
                self._set_table_item(self.ui.hf_files, row, 3, "")
        self.ui.hf_status.setText(
            "Partial .part deleted. The next download will start from scratch."
        )
        self.refresh_hf_local_files(silent=True)
        self._update_hf_download_button()

    def _on_hf_task_finished(self, task_key):
        self.refresh_hf_local_files(silent=True)
        self.refresh_local_model_manager(silent=True)
        self._refresh_hf_download_summary()

    def _on_hf_task_completed(self, task_key, ok, message):
        self.log_mgr.append(message, "info" if ok else "error")
        self.refresh_hf_local_files(silent=True)
        self.refresh_local_model_manager(silent=True)
        if ok:
            self.scan_models(silent=True)
        self._refresh_hf_download_summary()

    def on_models_found(self, models):
        self._syncing_model_combo = True
        try:
            self.ui.set_model_list(models)
        finally:
            self._syncing_model_combo = False
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

    def _sync_autotune_model_from_main(self, *_):
        if self._syncing_model_combo:
            return
        path = self.ui.model_combo.currentData()
        idx = self.ui.autotune.model_combo.findData(path)
        if idx < 0:
            return
        self._syncing_model_combo = True
        try:
            self.ui.autotune.model_combo.setCurrentIndex(idx)
        finally:
            self._syncing_model_combo = False

    def _sync_main_model_from_autotune(self, *_):
        if self._syncing_model_combo:
            return
        path = self.ui.autotune.model_combo.currentData()
        idx = self.ui.model_combo.findData(path)
        if idx < 0:
            return
        self._syncing_model_combo = True
        try:
            self.ui.model_combo.setCurrentIndex(idx)
        finally:
            self._syncing_model_combo = False

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
                self._refresh_overview()
                return
        info = self.ui.models_by_path.get(path) or extract_model_info(path)
        info.setdefault("path", path)
        info.setdefault("_model_path", path)
        cached_draft = str(info.get("mtp_draft_path") or "").strip()
        if cached_draft and (
            not os.path.isfile(cached_draft) or not is_mtp_draft_file(cached_draft)
        ):
            # A model scan may have cached a draft that the user deleted later.
            info["mtp_draft_path"] = ""
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
        self._refresh_perf_preset_names()

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
        manual_draft = self._mtp_manual_draft_path(info)
        detected_draft = str(info.get("mtp_draft_path") or "").strip()
        auto_draft = ""
        if (
            detected_draft
            and os.path.isfile(detected_draft)
            and not self._is_mtp_draft_auto_disabled(info)
        ):
            auto_draft = detected_draft
        for widget in (
            self.ui.spec_draft_n_max,
            self.ui.spec_draft_p_min,
            self.ui.spec_draft_gpu_layers,
            self.ui.spec_draft_model_path,
            self.ui.spec_draft_model_btn,
            self.ui.spec_draft_device,
        ):
            widget.setEnabled(True)
        self.ui.speculative_mtp.setEnabled(True)

        if manual_draft and os.path.isfile(manual_draft):
            self.ui.spec_draft_model_path.setText(manual_draft)
            if not self._is_mtp_draft_auto_disabled(info):
                self.ui.speculative_mtp.setChecked(True)
            self.ui.spec_draft_model_path.setPlaceholderText(
                "Manually selected draft GGUF"
            )
            return

        if self._uses_embedded_mtp_mode(info):
            self.ui.spec_draft_model_path.clear()
            self.ui.spec_draft_model_path.setPlaceholderText(
                "Auto: embedded/package MTP mode, no --model-draft"
            )
            self.ui.speculative_mtp.setToolTip(
                "Enable MTP speculative decoding. The selected GGUF looks embedded/package-capable, but manual mode is still allowed."
            )
            return

        if auto_draft:
            self.ui.spec_draft_model_path.setText(auto_draft)
            self.ui.speculative_mtp.setChecked(True)
            self.ui.spec_draft_model_path.setPlaceholderText(
                "Auto-detected nearby MTP draft GGUF"
            )
            self.ui.speculative_mtp.setToolTip(
                "Enable MTP speculative decoding. A nearby draft is auto-detected; clearing the field disables auto-selection for this model."
            )
            return

        self.ui.spec_draft_model_path.clear()
        self.ui.spec_draft_model_path.setPlaceholderText(
            "Optional draft GGUF; leave empty for embedded/manual MTP parameters"
        )
        self.ui.speculative_mtp.setToolTip(
            "Enable MTP speculative decoding manually. The model name/metadata does not need to contain MTP."
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
            self._set_ncmoe_help_text(tooltip)
        else:
            self._set_ncmoe_help_text("Model is not MoE")

        tooltip_ctx = build_ctx_tooltip(
            info=info,
            current_ctx=self.ui.ctx_size.value(),
            gpu_layers=self.ui.gpu_layers.value(),
            cache_type_k=self.ui.cache_type_k.currentText(),
            cache_type_v=self.ui.cache_type_v.currentText(),
            flash_attn=self.ui.flash_attn.isChecked(),
            parallel_slots=self.ui.parallel_slots.value(),
        )
        self._set_ctx_help_text(tooltip_ctx)

    def _is_mtp_model_info(self, info):
        if info.get("mtp_capable") or info.get("mtp_draft_path"):
            return True
        text = " ".join(
            str(info.get(k) or "")
            for k in ("path", "name", "display", "architecture", "_model_path")
        ).lower()
        return "mtp" in text

    @staticmethod
    def _mtp_model_key(model_path):
        return MtpModelRules.model_key(model_path)

    def _mtp_model_key(model_path):
        text = str(model_path or "").strip()
        if not text:
            return ""
        return os.path.normcase(os.path.abspath(text))

    def _mtp_info_model_key(self, info=None):
        return MtpModelRules.info_model_key(info, self._current_model_path())

    def _is_mtp_draft_auto_disabled(self, info=None):
        return MtpModelRules.is_draft_auto_disabled(
            self.config.settings, info, self._current_model_path()
        )

    def _set_mtp_draft_auto_disabled(self, disabled, info=None):
        MtpModelRules.set_draft_auto_disabled(
            self.config.settings, disabled, info, self._current_model_path()
        )

    def _mtp_manual_draft_path(self, info=None):
        return MtpModelRules.manual_draft_path(
            self.config.settings, info, self._current_model_path()
        )

    def _set_mtp_manual_draft_path(self, draft_path, info=None):
        MtpModelRules.set_manual_draft_path(
            self.config.settings, draft_path, info, self._current_model_path()
        )

    def _on_mtp_draft_path_edited(self, text):
        # textEdited is emitted only for a user edit, not for automatic setText().
        # Therefore an empty value is an explicit request not to auto-add draft.
        self._set_mtp_manual_draft_path(text)
        if str(text or "").strip() and os.path.isfile(str(text).strip()):
            self.ui.speculative_mtp.setChecked(True)
        self.config.read_from_ui(self.ui)
        self.config.save()

    def _on_mtp_checkbox_toggled(self, checked):
        # Persists per-model whether MTP should stay off: without this,
        # _sync_mtp_controls_for_model would force the checkbox back on the
        # next time this model is (re)selected, because a manual/auto draft
        # path is still on file. Idempotent: sync only ever checks the box
        # when this flag is already False, so recording False again there is
        # a no-op.
        self._set_mtp_draft_auto_disabled(not checked)

    def _uses_embedded_mtp_mode(self, info):
        """True when llama.cpp should use --spec-type draft-mtp without --model-draft."""
        return MtpModelRules.uses_embedded_mtp_mode(info)

    def _auto_mtp_supported(self, info):
        """Auto-enable known embedded MTP or an available non-disabled nearby draft."""
        return MtpModelRules.auto_mtp_supported(
            self.config.settings, info, self._current_model_path()
        )

    def _auto_mtp_draft_path(self, info):
        return MtpModelRules.auto_mtp_draft_path(
            self.config.settings, info, self._current_model_path()
        )

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
        self.ui.spec_draft_n_max.setValue(8)
        self.ui.spec_draft_p_min.setValue(0.8)
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
        if str(info.get("architecture") or "").lower().startswith("gemma4") or info.get(
            "is_qat"
        ):
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
        if (
            getattr(self, "_loading_preset", False)
            or getattr(self, "_applying_cli", False)
            or self.ui.loading_profile
        ):
            return

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
        self._set_ncmoe_help_text(tooltip)

        tooltip_ctx = build_ctx_tooltip(
            info=info,
            current_ctx=ctx_size,
            gpu_layers=self.ui.gpu_layers.value(),
            cache_type_k=self.ui.cache_type_k.currentText(),
            cache_type_v=self.ui.cache_type_v.currentText(),
            flash_attn=self.ui.flash_attn.isChecked(),
            parallel_slots=self.ui.parallel_slots.value(),
        )
        self._set_ctx_help_text(tooltip_ctx)

        preset_loaded = False
        model_path = self._current_model_path()
        if model_path and self._current_perf_preset_name() == "default":
            preset_loaded = self._try_load_perf_preset(model_path, ctx_size)

        if not preset_loaded:
            self._mark_preset_modified()
            self.update_cli_preview()

        self._mark_restart_needed()

    def _on_gpu_layers_changed(self, value):
        if (
            getattr(self, "_loading_preset", False)
            or getattr(self, "_applying_cli", False)
            or self.ui.loading_profile
        ):
            return

        info = self.ui.models_by_path.get(self.ui.model_combo.currentData())
        if info:
            self._refresh_tooltips(info)

        self._mark_preset_modified()
        self.update_cli_preview()
        self._mark_restart_needed()

    def _on_param_changed(self, _value=None):
        if (
            getattr(self, "_loading_preset", False)
            or getattr(self, "_applying_cli", False)
            or self.ui.loading_profile
        ):
            return

        info = self.ui.models_by_path.get(self.ui.model_combo.currentData())
        if info:
            self._refresh_tooltips(info)

        self._mark_preset_modified()
        self.update_cli_preview()
        self._mark_restart_needed()

    def _cli_manual_enabled(self) -> bool:
        return bool(
            getattr(self.ui, "cli_manual_mode", None)
            and self.ui.cli_manual_mode.isChecked()
        )

    def _on_cli_manual_mode_toggled(self, checked: bool):
        self.ui.cli_preview.setReadOnly(not checked)
        self.ui.cli_apply_btn.setEnabled(checked)
        if checked:
            self.ui.cli_status.setText("Manual CLI editing")
            self.ui.cli_status.setStyleSheet("color: " + STATUS_COLOR_WARNING + ";")
            self.ui.cli_preview.setStyleSheet(
                "background-color: #1f2933; color: #ffffff; font-family: Consolas; padding: 4px;"
            )
        else:
            self.ui.cli_status.setText("Generated from UI")
            self.ui.cli_status.setStyleSheet("color: " + STATUS_COLOR_MUTED + ";")
            self.ui.cli_preview.setStyleSheet(
                "background-color: #2a2a2a; color: #b5cea8; font-family: Consolas; padding: 4px;"
            )
            self.update_cli_preview(force=True)

    def _select_or_add_model_path(self, model_path: str):
        path = str(model_path or "").strip()
        if not path:
            return
        idx = self.ui.model_combo.findData(path)
        if idx < 0:
            name = Path(path).name or path
            self.ui.model_combo.addItem(name, path)
            self.ui.models_by_path.setdefault(path, {"path": path, "_model_path": path})
            idx = self.ui.model_combo.findData(path)
        if idx >= 0:
            self.ui.model_combo.setCurrentIndex(idx)

    def _cli_export_base_dirs(self) -> list[Path]:
        bases: list[Path] = []
        for raw in (
            self.ui.model_dir.text().strip(),
            self.ui.exe_path.text().strip(),
            str(Path(self._current_model_path()).parent)
            if self._current_model_path()
            else "",
            str(Path.cwd()),
        ):
            if not raw:
                continue
            path = Path(raw)
            if path.is_file():
                path = path.parent
            if path.exists() and path.is_dir():
                bases.append(path.resolve())
        return bases

    def _relative_cli_path(self, value: str, base_dirs: list[Path]) -> str:
        text = str(value or "").strip()
        if not text:
            return text
        path = Path(text)
        if not path.is_absolute():
            return text

        resolved = path.resolve()
        for base in base_dirs:
            try:
                rel = os.path.relpath(str(resolved), str(base))
            except ValueError:
                continue
            if rel == "." or rel.startswith(".." + os.sep) or rel == "..":
                continue
            return rel

        try:
            return os.path.relpath(str(resolved), str(Path.cwd()))
        except ValueError:
            return path.name

    def _portable_cli_tokens(self) -> list[str]:
        self.config.read_from_ui(self.ui)
        args = build_args(self.config.settings, self.ui.model_combo.currentData())
        if not args:
            return []

        exe = self._resolve_llamacpp_executable("server") or "llama-server.exe"
        tokens = [exe, *args]
        base_dirs = self._cli_export_base_dirs()
        path_flags = {
            "-m",
            "--model",
            "-md",
            "--model-draft",
            "-mm",
            "--mmproj",
            "--chat-template-file",
            "--grammar-file",
            "--api-key-file",
            "--lora",
            "--lora-scaled",
        }

        portable: list[str] = []
        expect_path_value = False
        for index, token in enumerate(tokens):
            if index == 0:
                portable.append(self._relative_cli_path(token, base_dirs))
                continue

            if expect_path_value:
                portable.append(self._relative_cli_path(token, base_dirs))
                expect_path_value = False
                continue

            flag, separator, inline_value = str(token).partition("=")
            if separator and flag in path_flags:
                portable.append(
                    f"{flag}={self._relative_cli_path(inline_value, base_dirs)}"
                )
                continue

            portable.append(token)
            expect_path_value = token in path_flags

        return portable

    def portable_cli_text(self) -> str:
        tokens = self._portable_cli_tokens()
        return subprocess.list2cmdline(tokens) if tokens else ""

    def apply_cli_preview(self):
        parsed = parse_llama_server_command(self.ui.cli_preview.toPlainText())
        if parsed.warnings and not parsed.settings:
            self.ui.cli_status.setText("; ".join(parsed.warnings))
            self.ui.cli_status.setStyleSheet("color: " + STATUS_COLOR_ERROR + ";")
            return

        self._applying_cli = True
        try:
            settings = dict(parsed.settings)
            if parsed.executable:
                settings.pop("cuda_version", None)
            if (
                settings.get("speculative_mtp") is True
                and "spec_draft_model_path" not in settings
            ):
                settings["spec_draft_model_path"] = ""
            # Мерж extra-флагов: существующие сохраняются, импорт побеждает
            # при конфликте, новые добавляются в конец.
            settings["extra_args"] = merge_extra_args(
                self.ui.extra_args.text(), parsed.extra_args
            )
            self.config.apply_values_to_ui(self.ui, settings)
            if "spec_draft_model_path" in settings:
                self._set_mtp_manual_draft_path(settings["spec_draft_model_path"])
        finally:
            self._applying_cli = False

        self._mark_preset_modified()
        self.update_cli_preview(force=False)
        self.save_settings()
        self._mark_restart_needed()

        if parsed.warnings:
            status = "Applied with warnings: " + "; ".join(parsed.warnings)
            color = STATUS_COLOR_WARNING
        else:
            status = "CLI applied"
            color = STATUS_COLOR_RUNNING
        ignored = []
        if parsed.executable:
            ignored.append("program path")
        if parsed.model_path:
            ignored.append("model path")
        if ignored and not parsed.warnings:
            status += " (ignored " + ", ".join(ignored) + ")"
        self.ui.cli_status.setText(status)
        self.ui.cli_status.setStyleSheet(f"color: {color};")
        if parsed.extra_args:
            self.log_mgr.append(f"CLI applied; extra params: {parsed.extra_args}")
        else:
            self.log_mgr.append("CLI applied")

    def copy_cli_preview(self):
        self.update_cli_preview(force=True)
        text = self.portable_cli_text().strip()
        if not text:
            self.ui.cli_status.setText("CLI is empty")
            self.ui.cli_status.setStyleSheet("color: " + STATUS_COLOR_ERROR + ";")
            return

        QApplication.clipboard().setText(text)
        self.ui.cli_status.setText("CLI copied with relative paths")
        self.ui.cli_status.setStyleSheet("color: " + STATUS_COLOR_RUNNING + ";")
        self.log_mgr.append("CLI copied to clipboard with relative paths")

    def import_cli_from_clipboard(self):
        text = QApplication.clipboard().text().strip()
        if not text:
            self.ui.cli_status.setText("Clipboard is empty")
            self.ui.cli_status.setStyleSheet("color: " + STATUS_COLOR_ERROR + ";")
            return

        self.ui.cli_preview.setText(text)
        self.apply_cli_preview()

    def update_cli_preview(self, force: bool = False):
        if self._cli_manual_enabled() and not force:
            self._update_cuda_status()
            return
        try:
            self.config.read_from_ui(self.ui)
            args = build_args(self.config.settings, self.ui.model_combo.currentData())
            exe = self._resolve_llamacpp_executable("server") or "llama-server.exe"
            self.ui.cli_preview.setText(
                subprocess.list2cmdline([exe, *args]) if args else ""
            )
        except Exception:
            self.ui.cli_preview.setText("")
        finally:
            self._update_cuda_status()
            self._refresh_overview()

    def _mark_restart_needed(self):
        """Подсвечивает, что запущенному серверу нужен рестарт для новых параметров."""
        if not self.server.is_server_running():
            return
        self.launcher.mark_restart_needed()
        self.ui.start_btn.setVisible(False)
        self.ui.reload_btn.setVisible(True)
        self.ui.reload_btn.setText("Restart to apply")
        self.ui.reload_btn.setStyleSheet(
            "background-color: "
            + STATUS_COLOR_WARNING
            + "; color: white; font-weight: bold; padding: 8px;"
        )
        self.ui.reload_btn.setEnabled(True)
        self._refresh_overview()

    def _reset_restart_indicator(self):
        self.launcher.clear_restart_needed()
        self.ui.reload_btn.setText("Restart")
        self.ui.reload_btn.setStyleSheet(
            "background-color: "
            + STATUS_COLOR_WARNING
            + "; color: white; font-weight: bold; padding: 8px;"
        )
        self._refresh_overview()

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
        except Exception as exc:
            logger.debug("nvidia-smi query failed: %s", exc)
            return []
        if result.returncode != 0:
            logger.debug("nvidia-smi returned code %s", result.returncode)
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
        except Exception as exc:
            logger.debug("tasklist query failed for PID %s: %s", pid, exc)
            return None
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if not line or "INFO:" in line:
            logger.debug("tasklist returned no row for PID %s", pid)
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
            gpu_layers = int(
                getattr(self.config.settings, "gpu_layers", gpu_layers) or gpu_layers
            )
        ctx_size = int(getattr(self.config.settings, "ctx_size", 0) or 0)
        if ctx_size <= 0:
            ctx_size = int(
                info.get("recommended_ctx") or info.get("context_length") or 4096
            )
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
        gpu_text = (
            "all"
            if getattr(self.config.settings, "gpu_layers_all", False)
            else str(gpu_layers)
        )
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
                lines.append(
                    f"    System RAM: {fmt_mem(used)} / {fmt_mem(total)} ({pct:.1f}%)"
                )
            if gpu_snapshots:
                lines.append(
                    "    Note: GPU total includes desktop and other processes."
                )
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
            for comp, mib in sorted(
                comps.items(), key=lambda item: item[1], reverse=True
            ):
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
        if self.mtp.mark_failed(reason, fatal=fatal):
            QTimer.singleShot(0, self._abort_bad_mtp_launch)

    def _schedule_mem_viz_flush(self):
        self._mem_viz_dirty = True
        timer = getattr(self, "_mem_viz_timer", None)
        if timer is None:
            self._flush_mem_viz()
        elif not timer.isActive():
            timer.start()

    def _flush_mem_viz(self):
        if not getattr(self, "_mem_viz_dirty", False):
            return
        self._mem_viz_dirty = False
        self._refresh_overview()

    def _on_log_for_mem_viz(self, text: str, level: str):
        """Обработка логов для визуализации памяти."""
        lower_text = text.lower()
        if "model doesn't contain mtp layers" in lower_text:
            self._mark_mtp_launch_failed(
                "main GGUF does not contain MTP layers", fatal=True
            )
        elif "failed to create mtp context" in lower_text:
            self._mark_mtp_launch_failed("failed to create MTP context", fatal=True)
        elif (
            "failed to load draft model" in lower_text
            or "invalid vector subscript" in lower_text
            or (
                "common_speculative_init_result" in lower_text
                and any(
                    marker in lower_text
                    for marker in ("failed", "error", "invalid", "exception")
                )
            )
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
            self._update_load_progress(line)
        if self._mem_data.server_ready:
            self._update_process_memory_fallbacks()
        self._maybe_log_memory_summary()
        self._schedule_mem_viz_flush()

    def _update_load_progress(self, line: str) -> None:
        bar = getattr(self.ui, "overview_load_progress", None)
        if bar is None:
            return
        progress = progress_from_load_line(line)
        if progress is None:
            return
        pct, phase = progress
        # Monotonic: log lines are mostly ordered, but a stray match for an
        # earlier phase (e.g. from a retry) must not walk the bar backwards.
        if pct < getattr(self, "_load_progress_pct", -1):
            return
        self._load_progress_pct = pct
        bar.setValue(pct)
        bar.setFormat(f"{phase} — %p%")
        if pct >= 100:
            bar.setVisible(False)

    def _reset_mem_viz(self, status: str | None = None):
        """Сброс данных памяти (визуализация удалена)."""
        self._mem_viz_dirty = False
        timer = getattr(self, "_mem_viz_timer", None)
        if timer is not None:
            timer.stop()
        self._mem_data = MemoryData()
        self._memory_summary_logged = False
        self._load_progress_pct = -1
        bar = getattr(self.ui, "overview_load_progress", None)
        if bar is not None:
            bar.setVisible(False)
        self._refresh_overview()

    def _finalize_mem_viz_after_stop(self, exit_code: int | None, status: str):
        """Финализирует данные памяти после остановки процесса."""
        if self._mem_data.fatal_error:
            # При ошибке оставляем диагностические данные, чтобы было видно,
            # какой компонент и сколько памяти пытались выделить.
            self._mem_data.process_exit_code = exit_code
            self._refresh_overview()
        else:
            # Нормальная выгрузка освобождает RAM/VRAM — старые allocations
            # больше неактуальны, поэтому полностью очищаем данные.
            self._reset_mem_viz(status)

    @staticmethod
    def _strip_mtp_args(args: list[str]) -> list[str]:
        return strip_mtp_args(args)

    def _retry_without_mtp_if_needed(self, exit_code: int) -> bool:
        should_retry, launch, reason = self.mtp.retry_plan(exit_code)
        if not should_retry:
            return False

        exe, fallback_args, env = launch
        self.ui.speculative_mtp.setChecked(False)
        self._set_mtp_draft_auto_disabled(True)
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

    def _record_server_diagnostic(self, exit_code: int):
        output = self.server.recent_server_output()
        result = analyze_server_failure(
            exit_code,
            output,
            crash_exit=self.server.server_last_crash_exit,
            stop_requested=self.server.server_last_stop_requested,
            process_error=self.server.server_last_process_error,
        )
        if not result:
            return None

        exe = ""
        args = []
        env = {}
        if self._last_server_launch:
            exe, args, env = self._last_server_launch
        report_path = ""
        try:
            report_path = str(
                write_server_report(
                    result,
                    executable=exe,
                    args=args,
                    env=env,
                    output=output,
                    process_error=self.server.server_last_process_error,
                    runtime_seconds=self.server.server_last_runtime_seconds,
                )
            )
        except OSError as exc:
            result["action"] += f" Failed to write the report: {exc}"

        summary = format_diagnostic_summary(result, report_path)
        self.last_diagnostic_path = report_path
        self.last_diagnostic_summary = summary
        self.ui.copy_last_error_btn.setEnabled(True)
        self.log_mgr.append(summary, "error")
        return result

    def _record_preflight_diagnostic(self, cause: str, action: str):
        result = {
            "cause": cause,
            "action": action,
            "exit_code": "process not started",
        }
        summary = format_diagnostic_summary(result)
        self.last_diagnostic_path = ""
        self.last_diagnostic_summary = summary
        self.ui.copy_last_error_btn.setEnabled(True)
        self.log_mgr.append(summary, "error")

    def copy_last_error(self):
        if not self.last_diagnostic_summary:
            return
        text = self.last_diagnostic_summary
        if self.last_diagnostic_path:
            try:
                report = Path(self.last_diagnostic_path).read_text(
                    encoding="utf-8", errors="replace"
                )
                text += f"\n\n{report}"
            except OSError:
                pass
        QApplication.clipboard().setText(text)
        self.ui.copy_last_error_btn.setText("Copied")
        QTimer.singleShot(
            1500, lambda: self.ui.copy_last_error_btn.setText("Copy last error")
        )

    def open_diagnostics_folder(self):
        folder = diagnostics_dir()
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as exc:
            QMessageBox.warning(self.ui, "Diagnostics", str(exc))

    def show_previous_native_crash(self, report_path: Path):
        self.last_diagnostic_path = str(report_path)
        self.last_diagnostic_summary = (
            "❌ DIAGNOSTICS\n"
            "Cause: the application itself crashed in the previous session.\n"
            "What to do: open the report; it contains the Python threads at crash time.\n"
            f"Full report: {report_path}"
        )
        self.ui.copy_last_error_btn.setEnabled(True)
        self.log_mgr.append(self.last_diagnostic_summary, "error")

    def handle_unhandled_exception(self, exc_type, exc_value, exc_traceback):
        exception_text = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        try:
            report_path = write_app_exception_report(exception_text)
            self.last_diagnostic_path = str(report_path)
        except OSError:
            self.last_diagnostic_path = ""
        self.last_diagnostic_summary = (
            "❌ DIAGNOSTICS\n"
            f"Cause: unhandled UI error: {exc_value}\n"
            "What to do: copy the report with the Copy last error button.\n"
            f"Full report: {self.last_diagnostic_path or 'failed to write'}"
        )
        self.ui.copy_last_error_btn.setEnabled(True)
        self.log_mgr.append(self.last_diagnostic_summary, "error")
        self.log_mgr.append(exception_text, "error")

    def handle_qt_message(self, message_type, context, message):
        location = ""
        if getattr(context, "file", None):
            location = f" ({context.file}:{getattr(context, 'line', 0)})"
        text = f"Qt: {message}{location}"
        serious = (
            message_type
            in {
                QtMsgType.QtCriticalMsg,
                QtMsgType.QtFatalMsg,
            }
            or "destroyed while thread is still running" in message.lower()
        )
        self.log_mgr.append(text, "error" if serious else "warn")
        if not serious:
            return
        stack = "".join(traceback.format_stack())
        try:
            report_path = write_app_exception_report(f"{text}\n\n{stack}")
            self.last_diagnostic_path = str(report_path)
        except OSError:
            self.last_diagnostic_path = ""
        self.last_diagnostic_summary = (
            "❌ DIAGNOSTICS\n"
            f"Cause: critical Qt error: {message}\n"
            "What to do: open the report; it contains the application stack.\n"
            f"Full report: {self.last_diagnostic_path or 'failed to write'}"
        )
        self.ui.copy_last_error_btn.setEnabled(True)

    def _on_server_stopped(self):
        """Обработчик остановки сервера."""
        self._stop_metrics_polling()
        exit_code = self.server.server_last_exit_code
        self._record_server_diagnostic(exit_code)
        if self.launcher.is_pending:
            self._reset_mem_viz("Server stopped, restarting with new parameters...")
            QTimer.singleShot(150, self._start_pending_restart)
            return
        if self._retry_without_mtp_if_needed(exit_code):
            return
        self._finalize_mem_viz_after_stop(
            exit_code,
            "Server stopped",
        )

    def _on_bench_finished(self, exit_code: int):
        self._finalize_mem_viz_after_stop(exit_code, "Benchmark finished")

    def _prepare_server_launch(self):
        if self.server.is_bench_running():
            self._record_preflight_diagnostic(
                "model not started: a benchmark is running",
                "Stop the benchmark and start again.",
            )
            QMessageBox.warning(
                self.ui,
                "Benchmark running",
                "Stop benchmark before starting server",
            )
            return None
        exe = self._resolve_llamacpp_executable("server")
        if not exe or not os.path.exists(exe):
            self._record_preflight_diagnostic(
                "llama-server.exe not found in the selected CUDA build",
                "Check the llama.cpp folder and the selected CUDA version.",
            )
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
            self.config.settings.spec_draft_n_max = 8
            self.config.settings.spec_draft_p_min = 0.8
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
            self._record_preflight_diagnostic(
                f"launch parameter error: {e}",
                "Fix the reported parameter in Launch settings and start again.",
            )
            QMessageBox.warning(self.ui, "Error", str(e))
            return None
        if not args:
            self._record_preflight_diagnostic(
                "failed to build the llama-server command",
                "Check the selected model and Launch settings.",
            )
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
        return ServerLaunchController.env_from_settings(self.config.settings)

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
        self.mtp.remember_launch(exe, args, env, is_retry=is_retry)
        self._reset_mem_viz()
        bar = getattr(self.ui, "overview_load_progress", None)
        if bar is not None:
            bar.setValue(0)
            bar.setFormat("starting...")
            bar.setVisible(True)
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
        had_pending, launch = self.launcher.poll_pending(
            self.server.is_server_running()
        )
        if not had_pending:
            return
        if launch is None:
            # Сервер ещё не остановился — опросим снова.
            QTimer.singleShot(150, self._start_pending_restart)
            return
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
        self.launcher.request_restart(launch)
        self.log_mgr.append("Restart requested: stopping current server...")
        self._reset_mem_viz("Stopping the server for restart...")
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

    def _auto_detect_mmproj(self, model_path: str) -> tuple[str, list[Path]]:
        """Ищет companion mmproj GGUF рядом с моделью, если поле mmproj пустое."""
        try:
            folder = Path(model_path).expanduser().resolve().parent
            candidates = [
                p for p in folder.glob("*.gguf")
                if p.name.lower().startswith("mmproj") or "-mmproj" in p.name.lower()
            ]
        except OSError:
            return "", []
        best = preferred_mmproj(candidates)
        return (str(best) if best else ""), candidates

    def _current_model_info(self):
        model_path = self._current_model_path()
        if not model_path:
            return {}
        info = self.ui.models_by_path.get(model_path)
        if not info:
            info = extract_model_info(model_path)
            self.ui.models_by_path[model_path] = info
        return info

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
        reply = QMessageBox.question(
            self.ui,
            "AutoTune blocked",
            "Found already running llama.cpp process. It can occupy VRAM/GPU and make AutoTune results invalid.\n\n"
            "Stop it first, then start AutoTune again — or continue anyway if you know it's safe "
            "(e.g. a CPU-only process that won't compete for GPU/VRAM).\n\n"
            f"{details}",
            QMessageBox.StandardButton.Ignore | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Ignore:
            self.log_mgr.append(
                "AutoTune: external llama.cpp process detected, user chose to continue anyway.",
                "warn",
            )
            return False
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
        model_path = self._current_model_path()
        if not model_path:
            QMessageBox.warning(self.ui, "AutoTune", "Select a GGUF model first")
            return
        server_exe = self._resolve_llamacpp_executable("server")
        if not server_exe or not os.path.exists(server_exe):
            QMessageBox.critical(
                self.ui,
                "AutoTune",
                "Select llama.cpp base folder with the requested CUDA build.",
            )
            return

        options = self.ui.autotune.options()
        mmproj = options["mmproj"]
        if not mmproj:
            mmproj, mmproj_candidates = self._auto_detect_mmproj(model_path)
            if mmproj:
                self.ui.autotune.mmproj_edit.setText(mmproj)
                self.log_mgr.append(f"AutoTune: auto-detected mmproj {mmproj}")
            elif mmproj_candidates:
                self.log_mgr.append(
                    "AutoTune: found multiple candidate mmproj files next to the model, "
                    "pick one manually in the mmproj field: "
                    + ", ".join(p.name for p in mmproj_candidates),
                    "warn",
                )
        config = SessionConfig(
            server_exe=server_exe,
            model_path=model_path,
            ctx=int(options["ctx"]),
            vision=options["vision"],
            mmproj=mmproj,
            mode=options["mode"],
            priority=options["priority"],
            kv_k=options["kv_k"],
            kv_v=options["kv_v"],
            degradation_policy=options["degradation_policy"],
            allow_kv_degradation=bool(options["allow_kv_degradation"]),
            allow_context_reduction=bool(options["allow_context_reduction"]),
            min_tg_tps=options["min_tg_tps"],
            min_pp_tps=options["min_pp_tps"],
            mtp_mode=options["mtp_mode"],
            vram_margin_mb=int(options["vram_margin_mb"]),
            require_vram_margin=bool(options["require_vram_margin"]),
            absolute_vram_floor_mb=int(options["absolute_vram_floor_mb"]),
            max_minutes=options["max_minutes"],
            max_runs=options["max_runs"],
            runtime_args=list(options["runtime_args"]),
        )
        self.autotune_session_result = None
        self.ui.autotune.mark_started(config.max_runs or 0)
        self.autotune = AutoTuneManager(config)
        self.autotune.log.connect(lambda text, level: self.log_mgr.append(text, level))
        self.autotune.result_ready.connect(self.ui.autotune.add_result)
        self.autotune.progress.connect(self.ui.autotune.set_progress)
        self.autotune.session_finished.connect(self._on_autotune_finished)
        self.autotune.session_failed.connect(self._on_autotune_failed)
        self.autotune.finished.connect(self.update_action_buttons)
        self._autotune_running = True
        self.ui.autotune.set_running(True)
        self.autotune.start()
        self.update_action_buttons()

    def cancel_autotune(self):
        if self.autotune and self.autotune.isRunning():
            self.log_mgr.append("AutoTune cancel requested", "warn")
            self.autotune.cancel()

    def _on_autotune_finished(self, session_result):
        self.autotune_session_result = session_result
        self._autotune_running = False
        self.ui.autotune.set_running(False)
        self.ui.autotune.show_session_result(
            session_result.status,
            session_result.stop_reason,
            session_result.profiles,
            session_result.elapsed_seconds,
            str(session_result.output_dir),
        )
        if session_result.profiles:
            self.log_mgr.append(
                f"AutoTune finished: {session_result.status}, "
                f"{len(session_result.profiles)} profile(s), results={session_result.output_dir}"
            )
            self.log_mgr.append(
                "Review the profiles, then click Apply on the one you want.", "info"
            )
        else:
            self.log_mgr.append(
                f"AutoTune finished: {session_result.status}, no stable profile, "
                f"results={session_result.output_dir}",
                "warn",
            )
        self.update_action_buttons()

    def _on_autotune_failed(self, message: str):
        self._autotune_running = False
        self.ui.autotune.set_running(False)
        self.ui.autotune.show_error(message)
        self.log_mgr.append(f"AutoTune failed: {message}", "error")
        self.update_action_buttons()
        QMessageBox.critical(self.ui, "AutoTune", message)

    def apply_autotune_profile(self, profile_name: str):
        profile = self.ui.autotune.profile_by_name(profile_name)
        if not profile:
            QMessageBox.warning(self.ui, "AutoTune", "Profile not available")
            return
        reply = QMessageBox.question(
            self.ui,
            "Apply AutoTune profile",
            f"Apply {profile_name}?\n\n" + " ".join(profile.command),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        values = candidate_to_settings_values(profile.candidate)
        self._loading_preset = True
        try:
            self.config.apply_values_to_ui(self.ui, values)
            if profile.candidate.extra_args:
                merged = merge_extra_args(
                    self.ui.extra_args.text(), " ".join(profile.candidate.extra_args)
                )
                self.ui.extra_args.setText(merged)
                self.config.settings.extra_args = merged
        finally:
            self._loading_preset = False

        self.update_cli_preview()
        self._mark_restart_needed()
        self.save_settings()
        self.log_mgr.append(f"AutoTune profile applied: {profile_name}")
        QMessageBox.information(self.ui, "AutoTune", f"{profile_name} applied to UI")

    def _reveal_force_stop(self):
        """Показать кнопку Force Stop, если сервер всё ещё запущен."""
        if self.server.is_server_running():
            self.ui.force_stop_btn.setVisible(True)

    def open_autotune_results_folder(self):
        output_dir = self.ui.autotune.last_output_dir()
        if not output_dir or not os.path.isdir(output_dir):
            QMessageBox.information(self.ui, "AutoTune", "No results folder yet")
            return
        if sys.platform.startswith("win"):
            os.startfile(output_dir)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", output_dir])
        else:
            subprocess.Popen(["xdg-open", output_dir])

    def stop_work(self):
        if self.launcher.cancel_pending():
            self.log_mgr.append("Restart cancelled")
        if self.server.is_server_running():
            self.server.stop_server()
            # Если сервер не остановился за 5 сек — показать Force Stop (Kill).
            self._force_stop_reveal_timer = QTimer(self.ui)
            self._force_stop_reveal_timer.setSingleShot(True)
            self._force_stop_reveal_timer.timeout.connect(self._reveal_force_stop)
            self._force_stop_reveal_timer.start(5000)
        if self.server.is_bench_running():
            self.server.stop_bench()
        if self.autotune and self.autotune.isRunning():
            self.autotune.cancel()
        if self.scanner and self.scanner.isRunning():
            self.scanner.requestInterruption()
        self.update_action_buttons()

    def force_stop_server(self):
        if self.launcher.cancel_pending():
            self.log_mgr.append("Restart cancelled")
        has_owned = self.server.is_server_running()
        external = [] if has_owned else self._external_llama_processes()
        if not has_owned and not external:
            self.log_mgr.append("Force stop skipped: no llama processes found", "warn")
            self.update_action_buttons()
            return
        now = time.monotonic()
        if now - self._last_force_stop_confirmed_at > 6.0:
            target_text = (
                "the current llama-server process tree"
                if has_owned
                else f"{len(external)} external llama process(es)"
            )
            if not confirm_destructive_action(
                self.ui,
                "Force Stop",
                "Force Stop immediately kills llama.cpp processes and may interrupt a model load or request.\n\n"
                f"Kill {target_text} now?",
            ):
                return
            self._last_force_stop_confirmed_at = now
        if self.server.is_server_running():
            self.log_mgr.append(
                "Force stop requested: killing llama-server now", "error"
            )
            self.server.force_stop_server()
            self.update_action_buttons()
            return
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

    def update_action_buttons(self, busy=False):
        srv = self.server.is_server_running()
        bnch = self.server.is_bench_running()
        scan = self.scanner and self.scanner.isRunning()
        upd = self.updater and self.updater.isRunning()
        tune = self._autotune_running or (self.autotune and self.autotune.isRunning())
        busy = srv or bnch or scan or tune or self.launcher.is_pending
        show_reload = srv or self.launcher.is_pending
        self.ui.start_btn.setVisible(not show_reload)
        self.ui.reload_btn.setVisible(show_reload)
        self.ui.stop_btn.setEnabled(busy)
        self.ui.force_stop_btn.setEnabled(True)
        # Force Stop скрыт, пока сервер не «зависнет» (показывается таймером
        # в stop_work). Когда сервер остановлен — прячем снова.
        self.ui.force_stop_btn.setVisible(srv)
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
        # Шаблон читается только при следующем запуске llama-server. Его можно
        # безопасно выбрать/заменить заранее, не останавливая текущую модель.
        for w in (
            self.ui.use_chat_template,
            self.ui.chat_template_file,
            self.ui.chat_template_btn,
        ):
            w.setEnabled(True)
        self.ui.reload_btn.setEnabled(
            srv
            and not bnch
            and not scan
            and not upd
            and not tune
            and not self.launcher.is_pending
            and not self.server.server_stop_requested
        )
        self.ui.test_btn.setEnabled(not busy and not upd)
        self.ui.autotune.start_btn.setEnabled(not busy and not upd)
        self.ui.autotune.cancel_btn.setEnabled(bool(tune))
        if not srv and not self.launcher.is_pending:
            self._reset_restart_indicator()
        self._refresh_overview()

    def auto_check_llamacpp_updates(self):
        """Passive startup check: is a newer llama.cpp build available for
        CUDA 12 and/or 13? Does not download — just flags it in the UI so the
        user knows before clicking Update."""
        exe = self.ui.exe_path.text().strip()
        if not exe:
            return
        if self._update_checker and self._update_checker.isRunning():
            return
        self._update_checker = LlamaCppUpdateChecker(exe)
        self._update_checker.checked.connect(self._on_llamacpp_update_checked)
        self._update_checker.error.connect(self._on_llamacpp_update_check_error)
        if not (self.updater and self.updater.isRunning()):
            self.ui.update_status.setText("Checking for llama.cpp updates...")
            self.ui.update_status.setStyleSheet("color: " + STATUS_COLOR_MUTED + ";")
            self.ui.update_progress.setRange(0, 0)  # indeterminate
            self.ui.update_progress.setVisible(True)
        self._update_checker.start()

    def _on_llamacpp_update_checked(self, result: dict):
        self._llamacpp_update_info = result
        parts = []
        any_update = False
        for version in ("12", "13"):
            info = result.get(version)
            if not info:
                continue
            current, latest = info.get("current"), info.get("latest")
            if current is None:
                parts.append(f"CUDA {version}: not installed")
            elif current < latest:
                parts.append(f"CUDA {version}: update available (b{current} → b{latest})")
                any_update = True
            else:
                parts.append(f"CUDA {version}: up to date (b{current})")
        if not (self.updater and self.updater.isRunning()):
            self.ui.update_status.setText(" · ".join(parts))
            color = STATUS_COLOR_WARNING if any_update else STATUS_COLOR_MUTED
            self.ui.update_status.setStyleSheet("color: " + color + ";")
            self.ui.update_progress.setRange(0, 100)
            self.ui.update_progress.setVisible(False)
        self._update_cuda_status()

    def _on_llamacpp_update_check_error(self, message: str):
        self.log_mgr.append(f"llama.cpp update check: {message}", "warning")
        if not (self.updater and self.updater.isRunning()):
            self.ui.update_status.setText(f"Update check failed: {message}")
            self.ui.update_status.setStyleSheet("color: " + STATUS_COLOR_WARNING + ";")
            self.ui.update_progress.setRange(0, 100)
            self.ui.update_progress.setVisible(False)

    def update_llamacpp(self):
        if self.server.is_server_running() or self.server.is_bench_running():
            QMessageBox.warning(self.ui, "Updater", "Stop processes before updating.")
            return
        if self.updater and self.updater.isRunning():
            QMessageBox.information(self.ui, "Updater", "Update is already running.")
            return
        if self.updater:
            self.updater.deleteLater()
            self.updater = None
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
            lambda: (
                self.ui.update_progress.setVisible(False)
                or self.update_action_buttons()
            )
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

        # Размер контекста: ручное значение из UI либо авто-опрос сервера.
        max_context = self.ui.integration_max_context.value()
        if not max_context or max_context <= 0:
            max_context = self.query_server_context_window(self.ui.current_base_url())

        mgr = IntegrationManager(base_url=self.ui.current_base_url())
        result = mgr.add_model(
            config_path, target, model_id, max_context=max_context or 0
        )
        self.ui.integration_status.setText(result.message)
        if result.success:
            self.ui.integration_models_list.clear()
            self.ui.integration_models_list.addItems(result.model_ids)
        else:
            QMessageBox.warning(self.ui, "Integration Error", result.message)

    def query_server_context_window(self, base_url: str) -> int:
        """Живой опрос сервера llama.cpp: GET /slots -> max(n_ctx).

        /slots — нативный endpoint у корня (не под /v1), поэтому /v1
        отрезается. Возвращает 0 при недоступности сервера.
        """
        import json
        import urllib.request
        import urllib.error

        root = (base_url or "").rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        url = f"{root}/slots"
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and data:
                return max(
                    (int(s.get("n_ctx", 0)) for s in data if isinstance(s, dict)),
                    default=0,
                )
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
            pass
        return 0

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
        self.shutdown_background_work()
        QApplication.instance().quit()


logger = logging.getLogger("llamaserver")


def _configure_logging() -> None:
    """Уровень логов llamaserver: WARNING по умолчанию, DEBUG с --debug."""
    level = logging.DEBUG if "--debug" in sys.argv else logging.WARNING
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    root = logging.getLogger("llamaserver")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


def _apply_theme(app: QApplication) -> None:
    """Глобальная QSS-тема (Этап 3.3). Точечные стили виджетов главнее."""
    theme_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    theme_path = theme_root / "src" / "ui" / "theme.qss"
    if not theme_path.exists():
        theme_path = Path(__file__).resolve().parent / "src" / "ui" / "theme.qss"
    try:
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.debug("theme.qss not applied: %s", exc)


def _selected_language() -> str:
    """Язык интерфейса: --lang=ru имеет приоритет, затем QSettings, затем en."""
    for arg in sys.argv[1:]:
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1].strip().lower()
    settings = QSettings("LlamaServerGUI", "UIState")
    return str(settings.value("language", "en") or "en").strip().lower()


def _install_translator(app: QApplication) -> str:
    """Подключает QTranslator с compiled-каталогом translations/ (Этап 4).

    Возвращает применённый язык ("en" — без перевода, исходные строки).
    """
    lang = _selected_language()
    if not lang or lang == "en":
        return "en"
    translator = QTranslator(app)
    for root in (
        Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)),
        Path(__file__).resolve().parent,
        Path.cwd(),
    ):
        qm_path = root / "translations" / f"llamaserver_{lang}.qm"
        if qm_path.exists() and translator.load(str(qm_path)):
            app.installTranslator(translator)
            return lang
    logger.debug("translation catalog not found for language %r", lang)
    return "en"


def main():
    _configure_logging()
    previous_native_crash = consume_previous_native_crash()
    native_log_path, native_log_stream = start_native_crash_capture()
    app = QApplication(sys.argv)
    _apply_theme(app)
    _install_translator(app)
    icon_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    icon_path = icon_root / "assets" / "llama_server_icon.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    gui = LlamaGUI()
    sys.excepthook = gui.handle_unhandled_exception
    qInstallMessageHandler(gui.handle_qt_message)
    app.aboutToQuit.connect(gui.shutdown_background_work)
    gui.ui.show()
    if previous_native_crash:
        gui.show_previous_native_crash(previous_native_crash)
    exit_code = 0
    try:
        exit_code = app.exec()
    finally:
        qInstallMessageHandler(None)
        finish_native_crash_capture(native_log_path, native_log_stream)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
