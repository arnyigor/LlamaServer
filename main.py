"""LlamaServer GUI - точка входа."""

import json
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

from PySide6.QtCore import Qt, QTimer, QtMsgType, qInstallMessageHandler
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
from src.core.cli_parser import parse_llama_server_command
from src.core.config import ConfigManager
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
    MAX_ACTIVE_TIME_DT,
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
    format_bytes,
    list_all_local_model_entries,
    list_all_partial_downloads,
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
        self.hf_downloaders = {}
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
        self._shutting_down = False
        self.last_diagnostic_path = ""
        self.last_diagnostic_summary = ""

        self.log_mgr = LogManager(self.ui.logs)
        self.log_mgr.speed_updated.connect(self._on_log_speed_updated)
        self.log_mgr.timing_updated.connect(self._on_log_timing_updated)
        self.ui.autoscroll_logs.toggled.connect(
            lambda checked: setattr(self.log_mgr, "autoscroll", checked)
        )
        self.ui.copy_last_error_btn.clicked.connect(self.copy_last_error)
        self.ui.open_diagnostics_btn.clicked.connect(self.open_diagnostics_folder)
        self.metrics = MetricsPoller(poll_interval_ms=250)
        self.metrics.slot_metrics_updated.connect(self._on_slot_metrics_updated)
        self.metrics.server_metrics_updated.connect(self._on_server_metrics_updated)
        self._latest_token_total = 0
        self._latest_prompt_total = 0
        self._latest_predicted_total = 0
        self._token_baseline_total = 0
        self._saved_token_total = 0
        self._saved_last_total = 0
        # Baseline-смещения "сессии": total/task токены и активное время
        # отображаются относительно точки Reset session, поэтому следующий
        # опрос /metrics не вернёт старые значения на экран.
        self._session_base_prompt = 0
        self._session_base_predicted = 0
        self._session_base_total = 0
        self._session_base_active_pp = 0.0
        self._session_base_active_tg = 0.0
        self._slot_prompt_total = 0
        self._slot_predicted_total = 0
        self._slot_token_seen = {}
        self._last_slots = []
        self._request_token_base = {}
        # Кумулятивные значения из /metrics (с момента старта сервера).
        # Используются только для "догоняющей" синхронизации вверх, т.к.
        # llama.cpp обновляет их лишь по завершении запросов.
        self._metrics_prompt_total = 0
        self._metrics_predicted_total = 0
        # Активное время работы модели (секунды PP/TG).
        #   _active_*    — total: сумма интервалов /slots за запуск сервера;
        #   _cur_*       — current: время текущего/последнего запроса
        #                  (точное значение приходит из llama_print_timings).
        # /metrics для времени не годится (его *tokens_seconds — это
        # throughput, а не длительность).
        self._active_prompt_s = 0.0
        self._active_predicted_s = 0.0
        self._cur_prompt_s = 0.0
        self._cur_predicted_s = 0.0
        self._was_processing = False
        self._last_poll_time = None

        self.config.load()
        self.config.apply_to_ui(self.ui)
        self._normalize_llamacpp_path_ui()
        self.auto_detect_bench()
        self._connect_signals()
        self._setup_tray()
        self._update_cuda_status()
        QTimer.singleShot(250, self.auto_scan_models)
        QTimer.singleShot(350, self._refresh_hf_partial_status)

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
        u.ctx_size.valueChanged.connect(self.on_ctx_changed)
        u.preset_name_combo.activated.connect(
            lambda _index: self._on_preset_selected()
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
        u.claude_config_path.editingFinished.connect(self._on_config_path_changed)
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
        u._browse_claude_clicked = self.browse_claude_config
        u._browse_chat_template_clicked = self.browse_chat_template
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
        # Без разделителей тысяч: пользователь счёл запятые сбивающими.
        return f"{max(int(value or 0), 0)}"

    def _server_metrics_url(self) -> str:
        host = str(self.ui.host.text() or "127.0.0.1").strip() or "127.0.0.1"
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{self.ui.port.value()}"

    def _on_log_speed_updated(self, text: str):
        # Скорость llama_print_timings из логов — приоритетный источник:
        # точный замер завершённого запроса. Дельты /slots занижают скорость
        # (теряют хвост генерации и включают время HTTP-опроса), поэтому
        # когда есть замер из логов — показываем именно его.
        self.ui.speed_label.setText(text)

    def _on_log_timing_updated(self, pp_seconds: float, tg_seconds: float):
        # Точное время завершённого запроса из llama_print_timings — им
        # заменяем current (живой подсчёт теряет первый интервал опроса).
        # Total остаётся живой суммой интервалов /slots.
        self._cur_prompt_s = pp_seconds
        self._cur_predicted_s = tg_seconds
        self._refresh_current_time_label()

    def _start_metrics_polling(self):
        self.metrics.set_url(self._server_metrics_url())
        self._slot_prompt_total = 0
        self._slot_predicted_total = 0
        self._slot_token_seen = {}
        self._last_slots = []
        self._request_token_base = {}
        self._metrics_prompt_total = 0
        self._metrics_predicted_total = 0
        self._session_base_prompt = 0
        self._session_base_predicted = 0
        self._session_base_total = 0
        self._session_base_active_pp = 0.0
        self._session_base_active_tg = 0.0
        self._token_baseline_total = 0
        self._active_prompt_s = 0.0
        self._active_predicted_s = 0.0
        self._cur_prompt_s = 0.0
        self._cur_predicted_s = 0.0
        self._was_processing = False
        self._last_poll_time = None
        self._refresh_active_time_label()
        self._refresh_current_time_label()
        self.metrics.start()
        self.ui.speed_label.setText("Speed: waiting for /slots...")
        self.ui.request_tokens_label.setText("Request: -")

    def _stop_metrics_polling(self):
        self.metrics.stop()
        self.ui.speed_label.setText("Speed: -")
        self.ui.request_tokens_label.setText("Request: -")

    def _refresh_token_label(self):
        total = max(self._latest_token_total - self._session_base_total, 0)
        task_total = max(self._latest_token_total - self._token_baseline_total, 0)
        prompt = max(self._latest_prompt_total - self._session_base_prompt, 0)
        generated = max(self._latest_predicted_total - self._session_base_predicted, 0)
        self.ui.tokens_label.setText(
            "Tokens: "
            + stat_sep().join(
                [
                    stat_kv(
                        "total",
                        self._fmt_counter(total),
                        STAT_COLOR_TOTAL,
                    ),
                    stat_kv("task", self._fmt_counter(task_total), STAT_COLOR_TASK),
                    stat_kv(
                        "prompt",
                        self._fmt_counter(prompt),
                        STAT_COLOR_PROMPT,
                    ),
                    stat_kv(
                        "generated",
                        self._fmt_counter(generated),
                        STAT_COLOR_GENERATED,
                    ),
                ]
            )
        )

    def _sync_latest_token_totals(self):
        slot_total = int(getattr(self, "_slot_prompt_total", 0) or 0) + int(
            getattr(self, "_slot_predicted_total", 0) or 0
        )
        metrics_total = int(getattr(self, "_metrics_prompt_total", 0) or 0) + int(
            getattr(self, "_metrics_predicted_total", 0) or 0
        )
        if slot_total <= 0 and metrics_total <= 0 and self._latest_token_total > 0:
            return
        self._apply_metrics_catch_up()
        self._latest_prompt_total = self._slot_prompt_total
        self._latest_predicted_total = self._slot_predicted_total
        self._latest_token_total = (
            self._latest_prompt_total + self._latest_predicted_total
        )

    def _reset_request_token_baseline(self):
        self._request_token_base = {}
        for slot in getattr(self, "_last_slots", []):
            slot_id = int(getattr(slot, "id", 0) or 0)
            self._request_token_base[slot_id] = (
                int(getattr(slot, "n_prompt_tokens", 0) or 0),
                int(getattr(slot, "n_decoded", 0) or 0),
            )

    def _request_counter_value(self, slot, attr_name: str, base_index: int) -> int:
        slot_id = int(getattr(slot, "id", 0) or 0)
        value = int(getattr(slot, attr_name, 0) or 0)
        base = getattr(self, "_request_token_base", {}).get(slot_id)
        if base is None:
            return value
        base_value = int(base[base_index] or 0)
        if value < base_value:
            base_values = list(base)
            base_values[base_index] = 0
            self._request_token_base[slot_id] = tuple(base_values)
            return value
        return max(value - base_value, 0)

    def _current_request_token_counts(self) -> tuple[int, int]:
        slots = list(getattr(self, "_last_slots", []) or [])
        active = [slot for slot in slots if getattr(slot, "is_processing", False)]
        visible = active or [
            slot
            for slot in slots
            if getattr(slot, "n_prompt_tokens", 0) or getattr(slot, "n_decoded", 0)
        ]
        if not visible:
            return 0, 0

        prompt = sum(
            self._request_counter_value(slot, "n_prompt_tokens", 0)
            for slot in visible
        )
        generated = sum(
            self._request_counter_value(slot, "n_decoded", 1) for slot in visible
        )
        return max(int(prompt), 0), max(int(generated), 0)

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
        self._sync_latest_token_totals()
        request_prompt, request_generated = self._current_request_token_counts()
        total = max(self._latest_token_total - self._session_base_total, 0)
        task_total = max(self._latest_token_total - self._token_baseline_total, 0)
        prompt = max(self._latest_prompt_total - self._session_base_prompt, 0)
        generated = max(
            self._latest_predicted_total - self._session_base_predicted, 0
        )
        active_prompt_s = max(self._active_prompt_s - self._session_base_active_pp, 0.0)
        active_generated_s = max(
            self._active_predicted_s - self._session_base_active_tg, 0.0
        )
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
            "tokens": {
                "total": int(total),
                "task": int(task_total),
                "prompt": int(prompt),
                "generated": int(generated),
                "request_prompt": int(request_prompt),
                "request_generated": int(request_generated),
                "saved_last": int(getattr(self, "_saved_last_total", 0) or 0),
                "saved_total": int(getattr(self, "_saved_token_total", 0) or 0),
            },
            "time_seconds": {
                "active_total": active_prompt_s + active_generated_s,
                "active_prompt": active_prompt_s,
                "active_generated": active_generated_s,
                "current_total": self._cur_prompt_s + self._cur_predicted_s,
                "current_prompt": self._cur_prompt_s,
                "current_generated": self._cur_predicted_s,
            },
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
        snapshot = self.runtime_stats_snapshot()
        tokens = snapshot["tokens"]
        times = snapshot["time_seconds"]
        model = snapshot["model"]
        server = snapshot["server"]
        lines = [
            "# LlamaServer Runtime Stats",
            "",
            f"- Exported: {snapshot['exported_at']}",
            f"- Model: {model.get('id') or '-'}",
            f"- Model path: {model.get('path') or '-'}",
            f"- Server: {server.get('base_url') or '-'}",
            f"- Running: {'yes' if server.get('running') else 'no'}",
            "",
            "## Tokens",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total | {tokens['total']} |",
            f"| Task | {tokens['task']} |",
            f"| Prompt | {tokens['prompt']} |",
            f"| Generated | {tokens['generated']} |",
            f"| Request prompt | {tokens['request_prompt']} |",
            f"| Request generated | {tokens['request_generated']} |",
            f"| Saved last | {tokens['saved_last']} |",
            f"| Saved total | {tokens['saved_total']} |",
            "",
            "## Time",
            "",
            "| Metric | Seconds | Formatted |",
            "|---|---:|---:|",
            f"| Active total | {times['active_total']:.3f} | {format_duration(times['active_total'])} |",
            f"| Active prompt | {times['active_prompt']:.3f} | {format_duration(times['active_prompt'])} |",
            f"| Active generated | {times['active_generated']:.3f} | {format_duration(times['active_generated'])} |",
            f"| Current total | {times['current_total']:.3f} | {format_duration(times['current_total'])} |",
            f"| Current prompt | {times['current_prompt']:.3f} | {format_duration(times['current_prompt'])} |",
            f"| Current generated | {times['current_generated']:.3f} | {format_duration(times['current_generated'])} |",
        ]
        return "\n".join(lines) + "\n"

    def _time_row_html(self, caption: str, pp_s: float, tg_s: float) -> str:
        """HTML-строка времени: `Caption: total (PP pp | TG tg)`."""
        inner = (
            f'<span style="color:{STAT_COLOR_CAPTION};">(</span>'
            + stat_sep().join(
                [
                    stat_kv("PP", format_duration(pp_s), STAT_COLOR_PROMPT),
                    stat_kv("TG", format_duration(tg_s), STAT_COLOR_GENERATED),
                ]
            )
            + f'<span style="color:{STAT_COLOR_CAPTION};">)</span>'
        )
        total = pp_s + tg_s
        return stat_kv(caption, format_duration(total), STAT_COLOR_TIME) + " " + inner

    def _refresh_active_time_label(self):
        """Отрисовка total активного времени (за текущую сессию)."""
        pp = max(self._active_prompt_s - self._session_base_active_pp, 0.0)
        tg = max(self._active_predicted_s - self._session_base_active_tg, 0.0)
        self.ui.active_time_label.setText(self._time_row_html("Active", pp, tg))

    def _refresh_current_time_label(self):
        """Отрисовка времени текущего/последнего запроса."""
        self.ui.current_time_label.setText(
            self._time_row_html("Current", self._cur_prompt_s, self._cur_predicted_s)
        )

    def _accumulate_active_time(self, dt: float, active):
        """Накопление активного времени по интервалу между опросами.

        dt — время между двумя последовательными опросами /slots, когда
        активен хотя бы один слот. Распределение между PP и TG — по долям
        мгновенных скоростей; точную разбивку завершённого запроса потом
        дают логи llama_print_timings (для current).
        """
        prompt_speed = sum(
            max(float(getattr(slot, "prompt_per_second", 0.0) or 0.0), 0.0)
            for slot in active
        )
        predicted_speed = sum(
            max(float(getattr(slot, "predicted_per_second", 0.0) or 0.0), 0.0)
            for slot in active
        )
        if prompt_speed > 0 and predicted_speed > 0:
            total_speed = prompt_speed + predicted_speed
            dt_pp = dt * prompt_speed / total_speed
            dt_tg = dt * predicted_speed / total_speed
        elif predicted_speed > 0:
            dt_pp, dt_tg = 0.0, dt
        elif prompt_speed > 0:
            dt_pp, dt_tg = dt, 0.0
        else:
            # Скорость ещё не измерилась (первый опрос запроса): интервал
            # пропускаем — current получит точное значение из логов.
            return
        self._active_prompt_s += dt_pp
        self._active_predicted_s += dt_tg
        self._cur_prompt_s += dt_pp
        self._cur_predicted_s += dt_tg
        self._refresh_active_time_label()
        self._refresh_current_time_label()

    def _apply_metrics_catch_up(self):
        """Синхронизация счётчиков с /metrics — только вверх.

        /metrics считает кумулятивно с момента старта сервера, но llama.cpp
        обновляет его счётчики только по завершении запроса (см. вызовы
        metrics.on_prediction / metrics.on_prompt_eval в server-context.cpp).
        Поэтому /metrics не может быть единственным источником (во время
        генерации числа замирают), а используется лишь как "догоняющий":
        если сервер знает больше — подтягиваем суммы вверх.
        """
        if self._metrics_prompt_total > self._slot_prompt_total:
            self._slot_prompt_total = self._metrics_prompt_total
        if self._metrics_predicted_total > self._slot_predicted_total:
            self._slot_predicted_total = self._metrics_predicted_total

    def _accumulate_slot_tokens(self, slots):
        """Накопление токенов из дельт /slots.

        Слоты накапливаются всегда: счётчики n_prompt_tokens_processed /
        n_decoded сохраняются у слота даже после завершения запроса
        (сбрасываются только при старте следующего), поэтому дельты
        покрывают и "быстрые" запросы, целиком уложившиеся между опросами.
        """
        for slot in slots:
            slot_id = int(getattr(slot, "id", 0) or 0)
            prompt_tokens = int(getattr(slot, "n_prompt_tokens_processed", 0) or 0)
            predicted_tokens = int(getattr(slot, "n_decoded", 0) or 0)
            previous_prompt, previous_predicted = self._slot_token_seen.get(
                slot_id, (0, 0)
            )
            prompt_delta = (
                prompt_tokens - previous_prompt
                if prompt_tokens >= previous_prompt
                else prompt_tokens
            )
            predicted_delta = (
                predicted_tokens - previous_predicted
                if predicted_tokens >= previous_predicted
                else predicted_tokens
            )
            self._slot_prompt_total += max(prompt_delta, 0)
            self._slot_predicted_total += max(predicted_delta, 0)
            self._slot_token_seen[slot_id] = (prompt_tokens, predicted_tokens)
        self._sync_latest_token_totals()
        self._refresh_token_label()

    def _on_slot_metrics_updated(self, slots):
        self._last_slots = list(slots)
        # Активное время: интервалы опросов, пока хоть один слот обрабатывает.
        # Большой зазор между опросами (> MAX_ACTIVE_TIME_DT) — пауза/простой,
        # его не считаем: точный current приходит из логов llama_print_timings.
        now = time.monotonic()
        active = [slot for slot in slots if getattr(slot, "is_processing", False)]
        if active and not self._was_processing:
            # Переход idle → processing: начался новый запрос, время текущего
            # запроса обнуляется.
            self._cur_prompt_s = 0.0
            self._cur_predicted_s = 0.0
            self._refresh_current_time_label()
        self._was_processing = bool(active)
        if self._last_poll_time is not None and active:
            dt = max(now - self._last_poll_time, 0.0)
            if dt <= MAX_ACTIVE_TIME_DT:
                self._accumulate_active_time(dt, active)
        self._last_poll_time = now

        if not slots:
            return
        visible = active or [
            slot
            for slot in slots
            if getattr(slot, "n_prompt_tokens", 0) or getattr(slot, "n_decoded", 0)
        ]
        if not visible:
            if not self.log_mgr.has_speed:
                self.ui.speed_label.setText("Speed: -")
            self.ui.request_tokens_label.setText("Request: -")
            return

        prompt_speed = sum(
            max(float(getattr(slot, "prompt_per_second", 0.0) or 0.0), 0.0)
            for slot in visible
        )
        predicted_speed = sum(
            max(float(getattr(slot, "predicted_per_second", 0.0) or 0.0), 0.0)
            for slot in visible
        )
        prompt_tokens = sum(
            self._request_counter_value(slot, "n_prompt_tokens", 0)
            for slot in visible
        )
        predicted_tokens = sum(
            self._request_counter_value(slot, "n_decoded", 1) for slot in visible
        )

        parts = []
        if prompt_speed > 0:
            parts.append(
                stat_kv("PP", f"{format_speed(prompt_speed)} tok/s", STAT_COLOR_PROMPT)
            )
        if predicted_speed > 0:
            parts.append(
                stat_kv(
                    "TG", f"{format_speed(predicted_speed)} tok/s", STAT_COLOR_GENERATED
                )
            )
        if not self.log_mgr.has_speed:
            self.ui.speed_label.setText(
                "Speed: " + (stat_sep().join(parts) if parts else "-")
            )
        if prompt_tokens or predicted_tokens:
            self.ui.request_tokens_label.setText(
                "Request: "
                + stat_sep().join(
                    [
                        stat_kv(
                            "prompt",
                            self._fmt_counter(prompt_tokens),
                            STAT_COLOR_PROMPT,
                        ),
                        stat_kv(
                            "generated",
                            self._fmt_counter(predicted_tokens),
                            STAT_COLOR_GENERATED,
                        ),
                    ]
                )
            )
        else:
            self.ui.request_tokens_label.setText("Request: -")
        self._accumulate_slot_tokens(slots)

    def _on_server_metrics_updated(self, metrics):
        prompt_total = int(getattr(metrics, "prompt_tokens_total", 0) or 0)
        predicted_total = int(getattr(metrics, "tokens_predicted_total", 0) or 0)
        total = prompt_total + predicted_total
        if total <= 0:
            return
        self._metrics_prompt_total = prompt_total
        self._metrics_predicted_total = predicted_total
        if total < self._token_baseline_total:
            self._token_baseline_total = 0
        self._apply_metrics_catch_up()
        # НЕ используем llamacpp:prompt_tokens_seconds / predicted_tokens_seconds
        # из /metrics как длительность: это throughput (токены/сек), а не время.
        # Точное время PP/TG берём из логов llama_print_timings.
        self._sync_latest_token_totals()
        self._refresh_token_label()

    def reset_task_tokens(self):
        """Сохранить текущую задачу в Saved и начать отсчёт новой с нуля.

        Обнуляет task-счётчик, Current time и Request. Total-токены и Active
        время (server-scope) не трогает — для них есть Reset session.
        """
        self._sync_latest_token_totals()
        task_total = max(self._latest_token_total - self._token_baseline_total, 0)
        self._saved_token_total += task_total
        self._token_baseline_total = self._latest_token_total
        self._reset_request_token_baseline()
        self._set_saved_label(last_total=task_total)
        self._cur_prompt_s = 0.0
        self._cur_predicted_s = 0.0
        self._last_poll_time = time.monotonic()
        self.log_mgr.reset_runtime_extractors(reset_speed=False, reset_timing=True)
        self._refresh_current_time_label()
        self.ui.request_tokens_label.setText("Request: -")
        self._refresh_token_label()
        self.log_mgr.append(
            f"Token counter reset: saved {task_total} tokens, "
            "Current time and Request reset"
        )

    def reset_session(self):
        """Обнулить все живые счётчики сессии (total/task, время, Request).

        Saved-история сохраняется. Реализовано через baseline-смещения,
        поэтому следующий опрос /metrics не вернёт старые значения на экран.
        """
        self._sync_latest_token_totals()
        self._session_base_prompt = self._latest_prompt_total
        self._session_base_predicted = self._latest_predicted_total
        self._session_base_total = self._latest_token_total
        self._session_base_active_pp = self._active_prompt_s
        self._session_base_active_tg = self._active_predicted_s
        self._token_baseline_total = self._latest_token_total
        self._reset_request_token_baseline()
        self._cur_prompt_s = 0.0
        self._cur_predicted_s = 0.0
        self._last_poll_time = time.monotonic()
        self.log_mgr.reset_runtime_extractors(reset_speed=True, reset_timing=True)
        self._refresh_token_label()
        self._refresh_active_time_label()
        self._refresh_current_time_label()
        self.ui.request_tokens_label.setText("Request: -")
        self.log_mgr.append("Session reset: tokens and time zeroed")

    def reset_saved_total(self):
        """Обнулить накопленную Saved-историю (last и total)."""
        self._saved_token_total = 0
        self._set_saved_label()
        self.log_mgr.append("Saved history reset")

    def _set_saved_label(self, last_total=None):
        last = max(int(last_total or 0), 0)
        self._saved_last_total = last
        self.ui.tokens_saved_label.setText(
            "Saved: "
            + stat_sep().join(
                [
                    stat_kv("last", self._fmt_counter(last), STAT_COLOR_SAVED),
                    stat_kv(
                        "total",
                        self._fmt_counter(self._saved_token_total),
                        STAT_COLOR_SAVED,
                    ),
                ]
            )
        )

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
                json.dump(self.runtime_stats_snapshot(), f, ensure_ascii=False, indent=2)
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

    def browse_claude_config(self):
        start = self.ui.claude_config_path.text().strip()
        f, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Select Claude Code settings.json",
            start,
            "JSON (*.json);;All files (*.*)",
        )
        if f:
            self.ui.claude_config_path.setText(f)
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
        for key, task in list(self.hf_downloaders.items()):
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
        self._restart_pending = False
        self._pending_restart_launch = None
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
            color = "#4CAF50"
        elif raw_path:
            parts = [f"CUDA {cuda_ver}: build not found"]
            color = "#FF9800"
        else:
            parts = [f"CUDA {cuda_ver}: select llama.cpp folder"]
            color = "#888"

        parts.append("model selected" if model_path else "select model")

        if cuda_ver == "13":
            note = "CUDA 13 requires NVIDIA driver 580+; best for RTX 50/Blackwell."
        else:
            note = "CUDA 12 is the safer default for RTX 30/40 and older stable builds."

        label.setText(" | ".join(parts))
        label.setToolTip(note)
        label.setStyleSheet(f"color: {color};")

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

    def _refresh_perf_preset_names(self, selected_name: str = ""):
        combo = getattr(self.ui, "preset_name_combo", None)
        if combo is None:
            return

        selected_name = (selected_name or combo.currentText()).strip() or "default"
        model_path = self._current_model_path()
        names = self.config.list_perf_preset_names(model_path) if model_path else ["default"]
        if selected_name not in names:
            names.append(selected_name)

        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(names)
            combo.setCurrentText(selected_name)
        finally:
            combo.blockSignals(False)

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
            self.ui.preset_status.setStyleSheet("color: #888;")

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
            preset_name = self._current_perf_preset_name()
            self.config.save_perf_preset(
                model_path,
                ctx,
                self.ui,
                autotune_params=autotune_params,
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
            self.ui.preset_status.setStyleSheet("color: #4CAF50;")
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

        preset_name = (preset_name or self._current_perf_preset_name()).strip() or "default"
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
            self.ui.preset_status.setStyleSheet("color: #4CAF50;")

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
            label.setStyleSheet("color: #FF9800;")

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
            self.ui.local_models_status.setText(
                f"No local GGUF models found in {model_dir}"
            )

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
            QMessageBox.warning(
                self.ui, "Hugging Face", "Вставьте repo id или URL модели"
            )
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
                self._upsert_hf_partial_task(
                    result.get("repo_id") or "",
                    file_info,
                    partial,
                    model_dir=self.ui.model_dir.text().strip(),
                )
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
            target_text = (
                f" → {lmstudio_repo_dir(Path(model_dir), result.get('repo_id', ''))}"
            )
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
            f"Незавершённые загрузки: {len(partials)}, сохранено {format_bytes(total)}. "
            "Они уже показаны в списке; Scan HF добавит полный размер и точный процент."
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
        self.ui.hf_local_files.clear()
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
        has_selection = bool(self.ui.hf_files.selectedItems())
        has_partial = bool(self._selected_hf_partial_info())
        count = len(self.ui.hf_files.selectedItems())
        self.ui.hf_download_btn.setText(
            f"{'Resume' if has_partial else 'Download'} selected models"
            + (f" ({count})" if count > 1 else "")
        )
        self.ui.hf_download_btn.setEnabled(has_selection)

        selected_task_keys = self._selected_hf_task_keys()
        selected_running = any(
            self.hf_downloaders.get(key, {}).get("worker")
            and self.hf_downloaders[key]["worker"].isRunning()
            for key in selected_task_keys
        )
        self.ui.hf_pause_btn.setEnabled(selected_running)
        self.ui.hf_cancel_btn.setEnabled(
            bool(selected_task_keys) or (has_partial and not selected_running)
        )

    @staticmethod
    def _hf_task_key(repo_id, file_info):
        filename = str(file_info.get("rfilename") or file_info.get("name") or "")
        return f"{repo_id}::{filename}"

    def _selected_hf_task_keys(self):
        return [
            str(item.data(Qt.ItemDataRole.UserRole) or "")
            for item in self.ui.hf_downloads.selectedItems()
            if item.data(Qt.ItemDataRole.UserRole)
        ]

    def _active_hf_downloads(self, repo_id=None):
        active = []
        for task in self.hf_downloaders.values():
            worker = task.get("worker")
            if worker and worker.isRunning() and (
                repo_id is None or task.get("repo_id") == repo_id
            ):
                active.append(task)
        return active

    def _set_hf_task_display(self, task_key):
        task = self.hf_downloaders.get(task_key)
        if not task:
            return
        item = task.get("item")
        if item is None:
            return
        name = task.get("name") or task_key
        percent_value = task.get("percent")
        percent = f"{int(percent_value):3d}%" if percent_value is not None else "  —%"
        status = task.get("status") or "queued"
        message = str(task.get("message") or "").strip()
        headline = f"{name}  |  {percent}  |  {status}"
        item.setText(f"{headline}\n{message}" if message else headline)
        item.setToolTip(message)

    def _upsert_hf_partial_task(self, repo_id, file_info, partial, model_dir=None):
        """Show a saved .part immediately, even before its repository is scanned."""
        if not repo_id or not file_info or not partial:
            return
        task_key = self._hf_task_key(repo_id, file_info)
        previous = self.hf_downloaders.get(task_key, {})
        worker = previous.get("worker")
        if worker and worker.isRunning():
            return

        item = previous.get("item")
        if item is None:
            self.ui.hf_downloads.addItem("")
            item = self.ui.hf_downloads.item(self.ui.hf_downloads.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, task_key)

        saved = int(partial.get("partial_size") or 0)
        expected = int(file_info.get("size") or partial.get("expected_size") or 0)
        percent = min(99, int(saved * 100 / expected)) if expected else None
        saved_text = partial.get("partial_size_text") or format_bytes(saved)
        total_text = format_bytes(expected) if expected else "размер уточняется"
        filename = str(file_info.get("rfilename") or file_info.get("name") or "")
        self.hf_downloaders[task_key] = {
            "worker": None,
            "repo_id": repo_id,
            "file_info": dict(file_info),
            "model_dir": model_dir or self.ui.model_dir.text().strip(),
            "name": f"{repo_id} / {filename}",
            "percent": percent,
            "status": "paused / resumable",
            "message": (
                f"Сохранено: {saved_text} / {total_text}\n"
                f"{partial.get('partial_path') or ''}"
            ),
            "item": item,
        }
        self._set_hf_task_display(task_key)

    def _refresh_hf_download_summary(self):
        active = self._active_hf_downloads()
        if active:
            average = int(sum(int(task.get("percent") or 0) for task in active) / len(active))
            self.ui.hf_progress.setRange(0, 100)
            self.ui.hf_progress.setValue(average)
            self.ui.hf_progress.setVisible(True)
            self.ui.hf_status.setText(
                f"Параллельные загрузки: {len(active)} | общий прогресс ~{average}%"
            )
        elif self.hf_downloaders:
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
            QMessageBox.warning(
                self.ui, "Hugging Face", "Сначала просканируйте репозиторий"
            )
            return
        selected = self.ui.hf_files.selectedItems()
        if not selected:
            QMessageBox.warning(
                self.ui, "Hugging Face", "Выберите GGUF файл для скачивания"
            )
            return
        model_dir = self.ui.model_dir.text().strip()
        if not model_dir:
            QMessageBox.warning(self.ui, "Hugging Face", "Укажите базовую папку Models")
            return

        files = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
        if self.ui.hf_include_mmproj.isChecked():
            projector = self._select_hf_projector()
            selected_names = {
                str(file_info.get("rfilename") or file_info.get("name") or "")
                for file_info in files
            }
            projector_name = str(
                (projector or {}).get("rfilename") or (projector or {}).get("name") or ""
            )
            if projector and projector_name not in selected_names:
                files.append(projector)

        repo_id = self.hf_scan_result.get("repo_id") or ""
        files = [
            file_info
            for file_info in files
            if file_info
            and not (
                self.hf_downloaders.get(self._hf_task_key(repo_id, file_info), {}).get("worker")
                and self.hf_downloaders[self._hf_task_key(repo_id, file_info)]["worker"].isRunning()
            )
        ]
        if not files:
            self.ui.hf_status.setText("Все выбранные файлы уже скачиваются")
            return
        target_root = lmstudio_repo_dir(Path(model_dir), repo_id)
        total_size = sum(int(f.get("size") or 0) for f in files)
        names = "\n".join(f"• {self._hf_file_display(f)}" for f in files)
        size_line = f"\nTotal: {format_bytes(total_size)}" if total_size else ""
        reply = QMessageBox.question(
            self.ui,
            "Download GGUF models",
            f"Запустить {len(files)} параллельных загрузок в LM Studio-compatible папку:\n"
            f"{target_root}\n\n{names}{size_line}",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.ui.hf_downloads.clearSelection()
        for file_info in files:
            self._start_hf_download_task(repo_id, file_info, model_dir)
        self._refresh_hf_download_summary()

    def _start_hf_download_task(self, repo_id, file_info, model_dir):
        task_key = self._hf_task_key(repo_id, file_info)
        previous = self.hf_downloaders.get(task_key, {})
        item = previous.get("item")
        if item is None:
            self.ui.hf_downloads.addItem("")
            item = self.ui.hf_downloads.item(self.ui.hf_downloads.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, task_key)

        worker = HfModelDownloader(repo_id, [file_info], model_dir)
        filename = str(file_info.get("rfilename") or file_info.get("name") or "")
        self.hf_downloaders[task_key] = {
            "worker": worker,
            "repo_id": repo_id,
            "file_info": file_info,
            "model_dir": model_dir,
            "name": filename,
            "percent": 0,
            "status": "starting",
            "message": "",
            "item": item,
        }
        item.setSelected(True)
        self._set_hf_task_display(task_key)
        worker.progress.connect(
            lambda message, key=task_key: self._on_hf_task_progress(key, message)
        )
        worker.percent.connect(
            lambda percent, key=task_key: self._on_hf_task_percent(key, percent)
        )
        worker.completed.connect(
            lambda ok, message, key=task_key: self._on_hf_task_completed(key, ok, message)
        )
        worker.finished.connect(lambda key=task_key: self._on_hf_task_finished(key))
        worker.start()

    def _on_hf_task_progress(self, task_key, message):
        task = self.hf_downloaders.get(task_key)
        if not task:
            return
        task["status"] = "downloading"
        task["message"] = message
        self._set_hf_task_display(task_key)

    def _on_hf_task_percent(self, task_key, percent):
        task = self.hf_downloaders.get(task_key)
        if not task:
            return
        task["percent"] = int(percent)
        self._set_hf_task_display(task_key)
        self._refresh_hf_download_summary()

    def pause_hf_download(self):
        paused = 0
        for key in self._selected_hf_task_keys():
            task = self.hf_downloaders.get(key, {})
            worker = task.get("worker")
            if worker and worker.isRunning():
                task["status"] = "pausing"
                worker.pause()
                self._set_hf_task_display(key)
                paused += 1
        if paused:
            self.ui.hf_status.setText(
                f"Пауза {paused} загрузок: сохраняю .part для докачки..."
            )

    def cancel_hf_download(self):
        selected_keys = self._selected_hf_task_keys()
        running_keys = [
            key
            for key in selected_keys
            if self.hf_downloaders.get(key, {}).get("worker")
            and self.hf_downloaders[key]["worker"].isRunning()
        ]
        if running_keys:
            reply = QMessageBox.question(
                self.ui,
                "Cancel downloads",
                f"Прервать выбранные загрузки ({len(running_keys)}) и удалить их .part?\n\n"
                "Если хотите продолжить позже — нажмите Pause вместо Cancel.",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            for key in running_keys:
                task = self.hf_downloaders[key]
                task["status"] = "cancelling"
                task["worker"].cancel_and_delete()
                self._set_hf_task_display(key)
            self.ui.hf_status.setText(
                f"Отмена {len(running_keys)} загрузок: удаляю частичные .part..."
            )
            return

        saved_partials = []
        for key in selected_keys:
            task = self.hf_downloaders.get(key, {})
            file_info = task.get("file_info") or {}
            filename = str(
                file_info.get("rfilename") or file_info.get("name") or ""
            )
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
            reply = QMessageBox.question(
                self.ui,
                "Delete partial downloads",
                f"Удалить сохранённые .part выбранных задач ({len(saved_partials)})?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            for key, partial_info in saved_partials:
                delete_file_safely(Path(partial_info.get("partial_path") or ""))
                task = self.hf_downloaders.get(key, {})
                task["status"] = "cancelled"
                task["percent"] = 0
                task["message"] = "Частичный .part удалён"
                self._set_hf_task_display(key)
            self.ui.hf_status.setText(
                f"Удалено частичных загрузок: {len(saved_partials)}"
            )
            self.refresh_hf_local_files(silent=True)
            self._update_hf_download_button()
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
        self.ui.hf_status.setText(
            "Частичный .part удалён. Следующая загрузка начнётся заново."
        )
        self.refresh_hf_local_files(silent=True)
        self._update_hf_download_button()

    def _on_hf_task_finished(self, task_key):
        self.refresh_hf_local_files(silent=True)
        self.refresh_local_model_manager(silent=True)
        self._refresh_hf_download_summary()

    def _on_hf_task_completed(self, task_key, ok, message):
        task = self.hf_downloaders.get(task_key)
        if task:
            task["status"] = "complete" if ok else "stopped"
            task["message"] = message
            if ok:
                task["percent"] = 100
            self._set_hf_task_display(task_key)
        self.log_mgr.append(message, "info" if ok else "error")
        self.refresh_hf_local_files(silent=True)
        self.refresh_local_model_manager(silent=True)
        if ok:
            self.scan_models(silent=True)
        self._refresh_hf_download_summary()

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
        cached_draft = str(info.get("mtp_draft_path") or "").strip()
        if cached_draft and (
            not os.path.isfile(cached_draft)
            or not is_mtp_draft_file(cached_draft)
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

    @staticmethod
    def _mtp_model_key(model_path):
        text = str(model_path or "").strip()
        if not text:
            return ""
        return os.path.normcase(os.path.abspath(text))

    def _mtp_info_model_key(self, info=None):
        info = info or {}
        model_path = (
            info.get("path")
            or info.get("_model_path")
            or self._current_model_path()
        )
        return self._mtp_model_key(model_path)

    def _is_mtp_draft_auto_disabled(self, info=None):
        model_key = self._mtp_info_model_key(info)
        disabled = getattr(
            self.config.settings, "spec_draft_auto_disabled_models", []
        )
        if not model_key or not isinstance(disabled, list):
            return False
        return model_key in {self._mtp_model_key(path) for path in disabled}

    def _set_mtp_draft_auto_disabled(self, disabled, info=None):
        model_key = self._mtp_info_model_key(info)
        if not model_key:
            return
        saved = getattr(
            self.config.settings, "spec_draft_auto_disabled_models", []
        )
        saved = saved if isinstance(saved, list) else []
        normalized = {
            self._mtp_model_key(path) for path in saved if str(path or "").strip()
        }
        if disabled:
            normalized.add(model_key)
        else:
            normalized.discard(model_key)
        self.config.settings.spec_draft_auto_disabled_models = sorted(normalized)

    def _mtp_manual_draft_path(self, info=None):
        model_key = self._mtp_info_model_key(info)
        saved = getattr(self.config.settings, "spec_draft_manual_paths", {})
        if not model_key or not isinstance(saved, dict):
            return ""
        for saved_model, draft_path in saved.items():
            if self._mtp_model_key(saved_model) == model_key:
                return str(draft_path or "").strip()
        return ""

    def _set_mtp_manual_draft_path(self, draft_path, info=None):
        model_key = self._mtp_info_model_key(info)
        if not model_key:
            return
        saved = getattr(self.config.settings, "spec_draft_manual_paths", {})
        saved = dict(saved) if isinstance(saved, dict) else {}
        normalized = {
            self._mtp_model_key(saved_model): str(path or "").strip()
            for saved_model, path in saved.items()
            if str(saved_model or "").strip() and str(path or "").strip()
        }
        text = str(draft_path or "").strip()
        if text:
            normalized[model_key] = text
        else:
            normalized.pop(model_key, None)
        self.config.settings.spec_draft_manual_paths = normalized
        self._set_mtp_draft_auto_disabled(not bool(text), info)

    def _on_mtp_draft_path_edited(self, text):
        # textEdited is emitted only for a user edit, not for automatic setText().
        # Therefore an empty value is an explicit request not to auto-add draft.
        self._set_mtp_manual_draft_path(text)
        if str(text or "").strip() and os.path.isfile(str(text).strip()):
            self.ui.speculative_mtp.setChecked(True)
        self.config.read_from_ui(self.ui)
        self.config.save()

    def _uses_embedded_mtp_mode(self, info):
        """True when llama.cpp should use --spec-type draft-mtp without --model-draft."""
        arch = str(info.get("architecture") or "").lower()
        name_text = " ".join(
            str(info.get(k) or "") for k in ("path", "name", "display", "_model_path")
        ).lower()
        return (
            arch.startswith(("gemma4", "qwen"))
            and bool(info.get("mtp_capable"))
            and not info.get("is_qat")
            and "qat" not in name_text
        )

    def _auto_mtp_supported(self, info):
        """Auto-enable known embedded MTP or an available non-disabled nearby draft."""
        if self._uses_embedded_mtp_mode(info):
            return True
        draft_path = str(info.get("mtp_draft_path") or "").strip()
        return bool(
            draft_path
            and os.path.isfile(draft_path)
            and not self._is_mtp_draft_auto_disabled(info)
        )

    def _auto_mtp_draft_path(self, info):
        if not self._auto_mtp_supported(info) or self._uses_embedded_mtp_mode(info):
            return ""
        manual_draft = self._mtp_manual_draft_path(info)
        if manual_draft and os.path.isfile(manual_draft):
            return manual_draft
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

        self._autotune_best_applied = False
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
            self.ui.cli_status.setStyleSheet("color: #FF9800;")
            self.ui.cli_preview.setStyleSheet(
                "background-color: #1f2933; color: #ffffff; font-family: Consolas; padding: 4px;"
            )
        else:
            self.ui.cli_status.setText("Generated from UI")
            self.ui.cli_status.setStyleSheet("color: #888;")
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
        parsed = parse_llama_server_command(self.ui.cli_preview.text())
        if parsed.warnings and not parsed.settings:
            self.ui.cli_status.setText("; ".join(parsed.warnings))
            self.ui.cli_status.setStyleSheet("color: #f44336;")
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
            settings["extra_args"] = parsed.extra_args
            self.config.apply_values_to_ui(self.ui, settings)
            if "spec_draft_model_path" in settings:
                self._set_mtp_manual_draft_path(settings["spec_draft_model_path"])
        finally:
            self._applying_cli = False

        self._autotune_best_applied = False
        self._mark_preset_modified()
        self.update_cli_preview(force=False)
        self.save_settings()
        self._mark_restart_needed()

        if parsed.warnings:
            status = "Applied with warnings: " + "; ".join(parsed.warnings)
            color = "#FF9800"
        else:
            status = "CLI applied"
            color = "#4CAF50"
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
            self.ui.cli_status.setStyleSheet("color: #f44336;")
            return

        QApplication.clipboard().setText(text)
        self.ui.cli_status.setText("CLI copied with relative paths")
        self.ui.cli_status.setStyleSheet("color: #4CAF50;")
        self.log_mgr.append("CLI copied to clipboard with relative paths")

    def import_cli_from_clipboard(self):
        text = QApplication.clipboard().text().strip()
        if not text:
            self.ui.cli_status.setText("Clipboard is empty")
            self.ui.cli_status.setStyleSheet("color: #f44336;")
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
        self._mtp_draft_error_seen = True
        self._mtp_failure_reason = reason
        if fatal and not self._mtp_auto_abort_requested:
            self._mtp_auto_abort_requested = True
            QTimer.singleShot(0, self._abort_bad_mtp_launch)

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
        if (
            "--spec-type" not in args
            and "--model-draft" not in args
            and "-md" not in args
        ):
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
            result["action"] += f" Не удалось записать отчёт: {exc}"

        summary = format_diagnostic_summary(result, report_path)
        self.last_diagnostic_path = report_path
        self.last_diagnostic_summary = summary
        self.ui.copy_last_error_btn.setEnabled(True)
        self.log_mgr.append(summary, "error")
        self.ui.tabs.setCurrentWidget(self.ui.log_tab)
        return result

    def _record_preflight_diagnostic(self, cause: str, action: str):
        result = {
            "cause": cause,
            "action": action,
            "exit_code": "процесс не запущен",
        }
        summary = format_diagnostic_summary(result)
        self.last_diagnostic_path = ""
        self.last_diagnostic_summary = summary
        self.ui.copy_last_error_btn.setEnabled(True)
        self.log_mgr.append(summary, "error")
        self.ui.tabs.setCurrentWidget(self.ui.log_tab)

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
        QTimer.singleShot(1500, lambda: self.ui.copy_last_error_btn.setText("Copy last error"))

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
            "❌ ДИАГНОСТИКА\n"
            "Причина: в предыдущем сеансе аварийно завершилось само приложение.\n"
            "Что сделать: откройте отчёт; там сохранены Python-потоки на момент сбоя.\n"
            f"Полный отчёт: {report_path}"
        )
        self.ui.copy_last_error_btn.setEnabled(True)
        self.log_mgr.append(self.last_diagnostic_summary, "error")
        self.ui.tabs.setCurrentWidget(self.ui.log_tab)

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
            "❌ ДИАГНОСТИКА\n"
            f"Причина: необработанная ошибка интерфейса: {exc_value}\n"
            "Что сделать: скопируйте отчёт кнопкой Copy last error.\n"
            f"Полный отчёт: {self.last_diagnostic_path or 'записать не удалось'}"
        )
        self.ui.copy_last_error_btn.setEnabled(True)
        self.log_mgr.append(self.last_diagnostic_summary, "error")
        self.log_mgr.append(exception_text, "error")
        self.ui.tabs.setCurrentWidget(self.ui.log_tab)

    def handle_qt_message(self, message_type, context, message):
        location = ""
        if getattr(context, "file", None):
            location = f" ({context.file}:{getattr(context, 'line', 0)})"
        text = f"Qt: {message}{location}"
        serious = message_type in {
            QtMsgType.QtCriticalMsg,
            QtMsgType.QtFatalMsg,
        } or "destroyed while thread is still running" in message.lower()
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
            "❌ ДИАГНОСТИКА\n"
            f"Причина: критическая ошибка Qt: {message}\n"
            "Что сделать: откройте отчёт; в нём сохранён стек приложения.\n"
            f"Полный отчёт: {self.last_diagnostic_path or 'записать не удалось'}"
        )
        self.ui.copy_last_error_btn.setEnabled(True)

    def _on_server_stopped(self):
        """Обработчик остановки сервера."""
        self._stop_metrics_polling()
        exit_code = self.server.server_last_exit_code
        self._record_server_diagnostic(exit_code)
        if self._restart_pending:
            self._reset_mem_viz("Сервер остановлен, перезапуск с новыми параметрами...")
            QTimer.singleShot(150, self._start_pending_restart)
            return
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
            self._record_preflight_diagnostic(
                "модель не запущена: сейчас выполняется benchmark",
                "Остановите benchmark и повторите запуск.",
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
                "llama-server.exe не найден в выбранной CUDA-сборке",
                "Проверьте папку llama.cpp и выбранную версию CUDA.",
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
                f"ошибка параметров запуска: {e}",
                "Исправьте указанный параметр в Launch settings и повторите запуск.",
            )
            QMessageBox.warning(self.ui, "Error", str(e))
            return None
        if not args:
            self._record_preflight_diagnostic(
                "не удалось сформировать команду llama-server",
                "Проверьте выбранную модель и Launch settings.",
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
            self.ui.spec_draft_p_min.setValue(
                float(params.get("spec_draft_p_min", self.ui.spec_draft_p_min.value()))
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
            preset_name = self._current_perf_preset_name()
            self.config.save_perf_preset(
                model_path,
                ctx,
                self.ui,
                metadata=metadata,
                autotune_params=self._best_autotune_params(),
                preset_name=preset_name,
            )
        except (ValueError, OSError) as e:
            QMessageBox.warning(self.ui, "AutoTune", str(e))
            return
        self._refresh_perf_preset_names(preset_name)
        self.log_mgr.append(
            f"AutoTune preset saved: {preset_name} | {Path(model_path).name} | ctx={ctx:,}"
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
            if target == "claude":
                # Tool use through llama-server's Anthropic-compatible endpoint
                # requires Jinja chat templates.
                self.ui.jinja.setChecked(True)
                self.save_settings()
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
        self.shutdown_background_work()
        QApplication.instance().quit()


def main():
    previous_native_crash = consume_previous_native_crash()
    native_log_path, native_log_stream = start_native_crash_capture()
    app = QApplication(sys.argv)
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
