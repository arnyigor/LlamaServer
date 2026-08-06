"""Главное окно и панели интерфейса."""

import os
import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QSpinBox,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QGroupBox,
    QMessageBox,
    QListWidget,
    QDoubleSpinBox,
    QCheckBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QGridLayout,
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QFont, QIcon

from src.ui.widgets import CollapsiblePanel
from src.ui.mem_viz_widget import MemoryVisualizationWidget
from src.ui.autotune_widget import AutoTuneWidget


class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Llama Server Studio")
        self.setGeometry(100, 100, 1280, 720)
        self.setMinimumSize(1050, 560)
        self._apply_app_icon()

        self.models = []
        self.models_by_path = {}
        self.loading_profile = False

        self.ui_settings = QSettings("LlamaServerGUI", "UIState")

        self._setup_ui()
        self._setup_tooltips()
        self._load_ui_state()

    def _apply_app_icon(self):
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
        icon_path = root / "assets" / "llama_server_icon.svg"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        left = self._build_left_panel()
        right = self._build_right_panel()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([820, 730])
        main_layout.addWidget(splitter)

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(720)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # === 0. Кнопки управления (вверху, чтобы не скролить) ===
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Server")
        self.start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        self.reload_btn = QPushButton("Restart", enabled=False)
        self.reload_btn.setVisible(False)
        self.reload_btn.setToolTip(
            "Restart the running server and apply the current model parameters"
        )
        self.reload_btn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
        )
        self.stop_btn = QPushButton("Stop", enabled=False)
        self.stop_btn.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; padding: 8px;"
        )
        self.force_stop_btn = QPushButton("Force Stop", enabled=True)
        self.force_stop_btn.setToolTip(
            "Immediately kills llama-server process tree if normal stop is stuck"
        )
        self.force_stop_btn.setStyleSheet(
            "background-color: #8B0000; color: white; font-weight: bold; padding: 8px;"
        )
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.reload_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.force_stop_btn)
        lay.addLayout(btn_row)

        # === 1. Пути ===
        g_paths = QGroupBox("Paths")
        lp = QVBoxLayout(g_paths)
        lp.setContentsMargins(12, 18, 12, 12)
        lp.setSpacing(8)

        self.exe_path = QLineEdit(placeholderText="Base folder with llama.cpp builds")
        self.exe_path.setToolTip(
            "Select the folder that contains version folders, e.g.\n"
            "G:/AIModels/llamacpp/\n\n"
            "Expected subfolders are auto-detected by CUDA version:\n"
            "llama-win-cuda-12.4-x64 / llama-win-cuda-13.3-x64"
        )
        self.bench_path = QLineEdit(placeholderText="Auto-detected llama-bench.exe")
        self.bench_path.setVisible(False)
        self.model_dir = QLineEdit(placeholderText="Base folder with models")

        for line, label, slot in [
            (self.exe_path, "Llama.cpp:", "_browse_exe_clicked"),
            (self.model_dir, "Models:", "_browse_model_dir_clicked"),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(line, 1)
            btn = QPushButton("...")
            btn.setFixedWidth(32)
            btn.clicked.connect(lambda _checked=False, s=slot: getattr(self, s)())
            row.addWidget(btn)
            lp.addLayout(row)

        upd_row = QHBoxLayout()
        self.cuda_version_combo = QComboBox()
        self.cuda_version_combo.addItem("CUDA 12", "12")
        self.cuda_version_combo.addItem("CUDA 13", "13")
        self.cuda_version_combo.setMaximumWidth(110)
        self.cuda_version_combo.setToolTip(
            "CUDA major version for llama.cpp builds.\n"
            "CUDA 13 also downloads additional cudart DLLs.\n"
            "Minor version (12.4 / 13.3) is auto-detected from release."
        )
        self.update_llama_btn = QPushButton("Update llama.cpp")
        self.update_status = QLabel("idle", wordWrap=True)
        upd_row.addWidget(self.cuda_version_combo)
        upd_row.addWidget(self.update_llama_btn)
        upd_row.addWidget(self.update_status, 1)
        lp.addLayout(upd_row)

        self.update_progress = QProgressBar(visible=False, minimum=0, maximum=100)
        lp.addWidget(self.update_progress)
        self.paths_panel = CollapsiblePanel("Advanced: Paths and llama.cpp")
        self.paths_panel.add_widget(g_paths)

        # === 2. Модель ===
        g_model = QGroupBox("Model")
        lm = QVBoxLayout(g_model)
        lm.setContentsMargins(12, 18, 12, 12)
        lm.setSpacing(8)

        scan_row = QHBoxLayout()
        self.scan_btn = QPushButton("Scan")
        scan_row.addWidget(self.scan_btn)
        lm.addLayout(scan_row)

        self.scan_status = QLabel("Models not scanned")
        self.scan_progress = QProgressBar(visible=False, minimum=0, maximum=0)
        lm.addWidget(self.scan_status)
        lm.addWidget(self.scan_progress)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumHeight(30)
        self.model_combo.setMaxVisibleItems(25)
        self.model_combo.setMinimumContentsLength(80)
        self.model_combo.setStyleSheet(
            "QComboBox { padding-left: 6px; padding-right: 34px; } "
            "QComboBox::drop-down { width: 30px; }"
        )
        if self.model_combo.lineEdit():
            self.model_combo.lineEdit().setTextMargins(6, 0, 34, 0)

        lm.addWidget(QLabel("Found GGUF:"))
        lm.addWidget(self.model_combo)

        self.auto_params = QCheckBox("Auto setup ctx/GPU/cache by GGUF")
        self.auto_params.setChecked(True)
        lm.addWidget(self.auto_params)

        mmproj_row = QHBoxLayout()
        self.use_mmproj = QCheckBox("Use mmproj")
        self.use_mmproj.setChecked(True)
        self.mmproj_offload = QCheckBox("mmproj offload")
        self.mmproj_offload.setChecked(True)
        mmproj_row.addWidget(self.use_mmproj)
        mmproj_row.addWidget(self.mmproj_offload)
        lm.addLayout(mmproj_row)

        self.model_info = QLabel("Select model")
        self.model_info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.model_id_label = QLabel("")
        self.model_id_label.setStyleSheet("color: #888; font-size: 10px;")
        self.model_id_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.model_id_label.setWordWrap(True)
        self.copy_model_btn = QPushButton("Copy model path")
        self.copy_model_btn.setFixedHeight(22)
        self.copy_model_btn.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        mrow = QHBoxLayout()
        mrow.addWidget(self.copy_model_btn)
        mrow.addWidget(self.model_id_label, 1)
        lm.addWidget(self.model_info)
        lm.addLayout(mrow)
        lay.addWidget(g_model)

        # === 2a. Локальные модели + загрузка с Hugging Face ===
        self.models_panel = CollapsiblePanel("Local model manager and download")
        local = self.models_panel.content_layout

        local_row = QHBoxLayout()
        self.local_models_refresh_btn = QPushButton("Refresh local models")
        self.local_models_delete_btn = QPushButton("Delete selected")
        self.local_models_delete_btn.setEnabled(False)
        self.local_models_delete_btn.setToolTip(
            "Safely delete the selected model folder/file from the Models base folder."
        )
        local_row.addWidget(QLabel("All models under Models:"))
        local_row.addStretch(1)
        local_row.addWidget(self.local_models_refresh_btn)
        local_row.addWidget(self.local_models_delete_btn)
        local.addLayout(local_row)

        self.local_models_list = QListWidget()
        self.local_models_list.setMaximumHeight(130)
        self.local_models_list.setToolTip(
            "Shows every local model folder/file found under the Models path, not only HF downloads. "
            "Projectors/MTP drafts are included in folder deletion but not shown as standalone models."
        )
        local.addWidget(self.local_models_list)

        self.local_models_status = QLabel("Refresh to list local models")
        self.local_models_status.setWordWrap(True)
        local.addWidget(self.local_models_status)

        # --- Подблок загрузки с Hugging Face (можно скрыть) ---
        self.show_hf_download = QCheckBox("Show Hugging Face download")
        self.show_hf_download.setChecked(True)
        self.show_hf_download.setToolTip(
            "Uncheck to hide the Hugging Face download section and keep the panel compact."
        )
        local.addWidget(self.show_hf_download)

        self.hf_section = QWidget()
        hf = QVBoxLayout(self.hf_section)
        hf.setContentsMargins(0, 0, 0, 0)
        hf.setSpacing(6)

        self.hf_repo = QLineEdit(
            placeholderText="repo or URL, e.g. unsloth/Qwen3.6-27B-MTP-GGUF"
        )
        self.hf_repo.setToolTip(
            "Paste Hugging Face repo id or model URL. Files are saved as:\n"
            "<Models>/<author>/<model>/<file>.gguf, compatible with LM Studio."
        )
        hf.addWidget(self.hf_repo)

        hf_filter_row = QHBoxLayout()
        self.hf_quant_filter = QLineEdit(placeholderText="filter: Q4_K_M or Q3-BF16")
        self.hf_quant_filter.setToolTip(
            "Optional filter. Examples: Q4_K_M, IQ4, Q3-BF16.\n"
            "Q3-BF16 means show quants from Q3 up to BF16."
        )
        self.hf_scan_btn = QPushButton("Scan HF")
        hf_filter_row.addWidget(self.hf_quant_filter, 1)
        hf_filter_row.addWidget(self.hf_scan_btn)
        hf.addLayout(hf_filter_row)

        hf_opts_row = QHBoxLayout()
        self.hf_include_mmproj = QCheckBox("also vision/mmproj")
        self.hf_include_mmproj.setChecked(True)
        self.hf_download_btn = QPushButton("Download selected")
        self.hf_download_btn.setEnabled(False)
        self.hf_pause_btn = QPushButton("Pause")
        self.hf_pause_btn.setEnabled(False)
        self.hf_cancel_btn = QPushButton("Cancel")
        self.hf_cancel_btn.setEnabled(False)
        hf_opts_row.addWidget(self.hf_include_mmproj)
        hf_opts_row.addWidget(self.hf_download_btn)
        hf_opts_row.addWidget(self.hf_pause_btn)
        hf_opts_row.addWidget(self.hf_cancel_btn)
        hf_opts_row.addStretch(1)
        hf.addLayout(hf_opts_row)

        self.hf_files = QListWidget()
        self.hf_files.setMaximumHeight(120)
        self.hf_files.setToolTip(
            "Select one main GGUF. Vision projector is paired automatically if found."
        )
        hf.addWidget(self.hf_files)

        self.hf_status = QLabel("Paste repo and scan")
        self.hf_status.setWordWrap(True)
        self.hf_progress = QProgressBar(visible=False, minimum=0, maximum=100)
        hf.addWidget(self.hf_status)
        hf.addWidget(self.hf_progress)

        hf_local_row = QHBoxLayout()
        self.hf_refresh_local_btn = QPushButton("Refresh local")
        self.hf_delete_local_folder_btn = QPushButton("Delete local folder")
        self.hf_delete_local_folder_btn.setEnabled(False)
        hf_local_row.addWidget(QLabel("Local files:"))
        hf_local_row.addStretch(1)
        hf_local_row.addWidget(self.hf_refresh_local_btn)
        hf_local_row.addWidget(self.hf_delete_local_folder_btn)
        hf.addLayout(hf_local_row)

        self.hf_local_files = QListWidget()
        self.hf_local_files.setMaximumHeight(90)
        self.hf_local_files.setToolTip(
            "Files already present in <Models>/<author>/<model>. Delete local folder removes the whole repo folder including mmproj/vision files."
        )
        hf.addWidget(self.hf_local_files)
        local.addWidget(self.hf_section)
        self.show_hf_download.toggled.connect(self.hf_section.setVisible)

        self.runtime_stats_group = QGroupBox("Runtime stats")
        stats = QGridLayout(self.runtime_stats_group)
        stats.setContentsMargins(12, 18, 12, 12)
        stats.setHorizontalSpacing(10)
        stats.setVerticalSpacing(6)
        self.speed_label = QLabel("Speed: -")
        self.speed_label.setTextFormat(Qt.RichText)
        self.speed_label.setStyleSheet(
            "color: #1a1a1a; font-family: Consolas; font-weight: bold;"
        )
        self.tokens_label = QLabel("Tokens: total 0 | task 0")
        self.tokens_label.setTextFormat(Qt.RichText)
        self.tokens_label.setStyleSheet("font-family: Consolas; color: #1a1a1a;")
        self.request_tokens_label = QLabel("Request: -")
        self.request_tokens_label.setTextFormat(Qt.RichText)
        self.request_tokens_label.setStyleSheet(
            "font-family: Consolas; color: #1a1a1a;"
        )
        self.tokens_saved_label = QLabel("Saved: 0")
        self.tokens_saved_label.setTextFormat(Qt.RichText)
        self.tokens_saved_label.setStyleSheet("color: #1a1a1a;")
        self.active_time_label = QLabel("Active: 0:00 (PP 0:00 | TG 0:00)")
        self.active_time_label.setTextFormat(Qt.RichText)
        self.active_time_label.setStyleSheet("font-family: Consolas; color: #1a1a1a;")
        self.active_time_label.setToolTip(
            "Active — суммарное время работы модели (prompt processing + "
            "token generation) с момента старта сервера. Простой и ожидание "
            "в очереди не считаются. Сбрасывается при каждом старте сервера."
        )
        self.current_time_label = QLabel("Current: 0:00 (PP 0:00 | TG 0:00)")
        self.current_time_label.setTextFormat(Qt.RichText)
        self.current_time_label.setStyleSheet("font-family: Consolas; color: #1a1a1a;")
        self.current_time_label.setToolTip(
            "Current — время последнего запроса (prompt processing + "
            "token generation). Точное значение приходит из llama_print_timings "
            "по завершении запроса."
        )
        self.tokens_reset_btn = QPushButton("Reset task tokens")
        self.tokens_reset_btn.setToolTip(
            "Save the current task token count and start counting the next task from zero."
        )
        stats.addWidget(self.speed_label, 0, 0, 1, 2)
        stats.addWidget(self.tokens_label, 1, 0, 1, 2)
        stats.addWidget(self.request_tokens_label, 2, 0)
        stats.addWidget(self.tokens_saved_label, 2, 1)
        stats.addWidget(self.active_time_label, 3, 0, 1, 2)
        stats.addWidget(self.current_time_label, 4, 0, 1, 2)
        stats.addWidget(self.tokens_reset_btn, 0, 2, 3, 1)
        stats.setColumnStretch(1, 1)
        lay.addWidget(self.runtime_stats_group)

        # === 3. Производительность ===
        g_launch = QGroupBox("Launch settings")
        launch = QVBoxLayout(g_launch)
        launch.setContentsMargins(12, 18, 12, 12)
        launch.setSpacing(8)

        self.adv_panel = CollapsiblePanel("Advanced: Memory, Sampling, Server")
        lperf = self.adv_panel.content_layout
        lperf.setContentsMargins(8, 6, 8, 6)
        lperf.setSpacing(8)
        # Launch settings groupbox — первая секция общего спойлера Advanced
        self.adv_panel.add_widget(g_launch)

        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(0, 999)
        self.gpu_layers.setValue(33)
        self.gpu_auto = QCheckBox("auto")
        self.gpu_auto.setChecked(True)
        self.gpu_layers_all = QCheckBox("all")

        def sync_gpu_layer_controls():
            self.gpu_layers.setDisabled(
                self.gpu_auto.isChecked() or self.gpu_layers_all.isChecked()
            )
            self.gpu_auto.setDisabled(self.gpu_layers_all.isChecked())

        self.gpu_auto.toggled.connect(lambda _checked: sync_gpu_layer_controls())
        self.gpu_layers_all.toggled.connect(lambda _checked: sync_gpu_layer_controls())
        sync_gpu_layer_controls()
        self.cpu_moe_layers = QSpinBox()
        self.cpu_moe_layers.setRange(-1, 200)
        self.cpu_moe_layers.setValue(-1)
        self.cpu_moe_layers.setSpecialValueText("auto")

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("GPU offload (-ngl):"))
        r1.addWidget(self.gpu_layers)
        r1.addWidget(self.gpu_auto)
        r1.addWidget(self.gpu_layers_all)
        r1.addStretch(1)

        self.ctx_size = QSpinBox()
        self.ctx_size.setRange(-1, 1048576)
        self.ctx_size.setSingleStep(512)
        self.ctx_size.setValue(-1)
        self.ctx_size.setSpecialValueText("auto")

        self.save_preset_btn = QPushButton("Save Preset")
        self.save_preset_btn.setToolTip(
            "Save parameters (ngl, ncmoe, etc.) for current model and context"
        )

        self.preset_status = QLabel("Preset: none")
        self.preset_status.setStyleSheet("color: #888;")

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Context Size (-c):"))
        r2.addWidget(self.ctx_size)

        self.ctx_quick_buttons = []
        for label, value in [
            ("8K", 8192),
            ("16K", 16384),
            ("24K", 24576),
            ("32K", 32768),
            ("41K", 40960),
            ("65K", 65536),
            ("128K", 131072),
            ("256K", 262144),
        ]:
            btn = QPushButton(label)
            btn.setFixedWidth(42 if len(label) <= 3 else 50)
            btn.setFixedHeight(24)
            btn.setToolTip(f"Set Context Size to {value}")
            btn.setProperty("ctx_value", value)
            self.ctx_quick_buttons.append(btn)
            r2.addWidget(btn)

        r2.addStretch(1)
        launch.addLayout(r2)
        launch.addLayout(r1)

        r2b = QHBoxLayout()
        r2b.addWidget(self.save_preset_btn)
        r2b.addWidget(self.preset_status, 1)
        launch.addLayout(r2b)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(-1, 32768)
        self.batch_size.setSingleStep(128)
        self.batch_size.setValue(-1)
        self.batch_size.setSpecialValueText("auto")
        self.ubatch_size = QSpinBox()
        self.ubatch_size.setRange(-1, 8192)
        self.ubatch_size.setSingleStep(64)
        self.ubatch_size.setValue(-1)
        self.ubatch_size.setSpecialValueText("auto")
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Batch / UBatch (-b / -ub):"))
        r3.addWidget(self.batch_size)
        r3.addWidget(self.ubatch_size)
        launch.addLayout(r3)

        self.threads = QSpinBox()
        self.threads.setRange(1, 64)
        self.threads.setValue(os.cpu_count() or 4)
        self.threads_batch = QSpinBox()
        self.threads_batch.setRange(0, 64)
        self.threads_batch.setSpecialValueText("same")
        self.threads_batch.setValue(0)
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Threads gen / batch (-t / -tb):"))
        r4.addWidget(self.threads)
        r4.addWidget(self.threads_batch)
        perf_header = QLabel("Performance and Memory:")
        perf_header.setStyleSheet("font-weight: bold; color: #666;")
        lperf.addWidget(perf_header)
        lperf.addLayout(r4)

        self.cache_type_k = QComboBox()
        self.cache_type_v = QComboBox()
        for ct in ["f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "f32"]:
            self.cache_type_k.addItem(ct)
            self.cache_type_v.addItem(ct)
        r5 = QHBoxLayout()
        r5.addWidget(QLabel("KV Cache K / V (-ctk / -ctv):"))
        r5.addWidget(self.cache_type_k)
        r5.addWidget(self.cache_type_v)
        launch.addLayout(r5)

        self.flash_attn = QCheckBox("Flash Attention (-fa)")
        self.flash_attn.setChecked(True)
        self.fit_off = QCheckBox("Fit off (--fit off)")
        self.fit_off.setChecked(True)
        r6 = QHBoxLayout()
        r6.addWidget(self.flash_attn)
        r6.addWidget(self.fit_off)
        launch.addLayout(r6)

        r_cpu = QHBoxLayout()
        r_cpu.addWidget(QLabel("CPU MoE offload (-ncmoe):"))
        r_cpu.addWidget(self.cpu_moe_layers)
        r_cpu.addStretch(1)
        lperf.addLayout(r_cpu)

        self.reasoning_mode = QComboBox()
        self.reasoning_mode.addItems(["off", "auto", "on"])
        self.reasoning_mode.setCurrentText("off")
        self.enable_thinking = QComboBox()
        self.enable_thinking.addItems(["off", "false", "true"])
        self.enable_thinking.setCurrentText("off")
        r7 = QHBoxLayout()
        r7.addWidget(QLabel("Reasoning (--reasoning):"))
        r7.addWidget(self.reasoning_mode)
        r7.addSpacing(10)
        r7.addWidget(QLabel("Thinking:"))
        r7.addWidget(self.enable_thinking)
        lperf.addLayout(r7)

        self.host = QLineEdit(placeholderText="127.0.0.1")
        self.host.setText("127.0.0.1")
        self.host.setMaximumWidth(120)
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(8080)
        self.parallel_slots = QSpinBox()
        self.parallel_slots.setRange(-1, 16)
        self.parallel_slots.setValue(-1)
        self.parallel_slots.setSpecialValueText("auto")
        r8 = QHBoxLayout()
        r8.addWidget(QLabel("Host:"))
        r8.addWidget(self.host)
        r8.addWidget(QLabel("Port:"))
        r8.addWidget(self.port)
        r8.addSpacing(10)
        r8.addWidget(QLabel("Slots (-np):"))
        r8.addWidget(self.parallel_slots)
        lperf.addLayout(r8)

        self.kv_unified = QCheckBox("KV unified (-kvu)")
        self.speculative_mtp = QCheckBox("MTP speculative")
        self.spec_draft_n_max = QSpinBox()
        self.spec_draft_n_max.setRange(1, 8)
        self.spec_draft_n_max.setValue(3)
        self.spec_draft_gpu_layers = QLineEdit(placeholderText="all")
        self.spec_draft_gpu_layers.setText("all")
        self.spec_draft_gpu_layers.setMaximumWidth(60)
        r8b = QHBoxLayout()
        r8b.addWidget(self.kv_unified)
        r8b.addWidget(self.speculative_mtp)
        r8b.addSpacing(10)
        r8b.addWidget(QLabel("MTP n/draft ngl:"))
        r8b.addWidget(self.spec_draft_n_max)
        r8b.addWidget(self.spec_draft_gpu_layers)
        lperf.addLayout(r8b)

        r8b2 = QHBoxLayout()
        r8b2.addWidget(QLabel("MTP draft GGUF:"))
        self.spec_draft_model_path = QLineEdit(
            placeholderText="Auto-detected, or browse for separate MTP GGUF"
        )
        self.spec_draft_model_path.setToolTip(
            "Optional separate MTP/draft GGUF. Gemma 4 packages often include it in an MTP folder; Qwen3.6 may require a separate file."
        )
        self.spec_draft_model_btn = QPushButton("...")
        self.spec_draft_model_btn.setFixedWidth(32)
        self.spec_draft_model_btn.clicked.connect(
            lambda _checked=False: self._browse_mtp_draft_clicked()
        )
        r8b2.addWidget(self.spec_draft_model_path, 1)
        r8b2.addWidget(self.spec_draft_model_btn)
        lperf.addLayout(r8b2)

        self.cuda_device = QLineEdit(placeholderText="CUDA0")
        self.cuda_device.setMaximumWidth(80)
        self.spec_draft_device = QLineEdit(placeholderText="CUDA0")
        self.spec_draft_device.setMaximumWidth(80)
        self.split_mode = QComboBox()
        self.split_mode.addItems(["", "none", "layer", "row"])
        self.main_gpu = QSpinBox()
        self.main_gpu.setRange(-1, 16)
        self.main_gpu.setSpecialValueText("auto")
        self.main_gpu.setValue(-1)
        r8c = QHBoxLayout()
        r8c.addWidget(QLabel("Device:"))
        r8c.addWidget(self.cuda_device)
        r8c.addWidget(QLabel("Draft device:"))
        r8c.addWidget(self.spec_draft_device)
        r8c.addWidget(QLabel("Split:"))
        r8c.addWidget(self.split_mode)
        r8c.addWidget(QLabel("Main GPU:"))
        r8c.addWidget(self.main_gpu)
        lperf.addLayout(r8c)

        self.ctx_checkpoints = QSpinBox()
        self.ctx_checkpoints.setRange(-1, 128)
        self.ctx_checkpoints.setSpecialValueText("default")
        self.ctx_checkpoints.setValue(-1)
        self.cache_ram = QSpinBox()
        self.cache_ram.setRange(-2, 262144)
        self.cache_ram.setSpecialValueText("default")
        self.cache_ram.setValue(-2)
        r9 = QHBoxLayout()
        r9.addWidget(QLabel("Ctx Checkpoints:"))
        r9.addWidget(self.ctx_checkpoints)
        r9.addSpacing(10)
        r9.addWidget(QLabel("Cache RAM (MiB):"))
        r9.addWidget(self.cache_ram)
        lperf.addLayout(r9)
        # === 4. Sampling / Debug / Server (общий спойлер Advanced) ===
        samp_header = QLabel("Sampling, Debug, Server:")
        samp_header.setStyleSheet("font-weight: bold; color: #666;")
        self.adv_panel.add_widget(samp_header)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(-1.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(-1.0)
        self.temperature.setDecimals(2)
        self.temperature.setSpecialValueText("auto")
        self.repeat_penalty = QDoubleSpinBox()
        self.repeat_penalty.setRange(-1.0, 2.0)
        self.repeat_penalty.setSingleStep(0.01)
        self.repeat_penalty.setValue(-1.0)
        self.repeat_penalty.setDecimals(2)
        self.repeat_penalty.setSpecialValueText("auto")
        s1 = QHBoxLayout()
        s1.addWidget(QLabel("Temperature:"))
        s1.addWidget(self.temperature)
        s1.addSpacing(10)
        s1.addWidget(QLabel("Repeat Penalty:"))
        s1.addWidget(self.repeat_penalty)
        self.adv_panel.add_layout(s1)

        self.use_mmap = QCheckBox("mmap")
        self.use_mmap.setChecked(True)
        self.use_mlock = QCheckBox("mlock")
        self.verbose = QCheckBox("verbose")
        self.log_timestamps = QCheckBox("log timestamps")
        s2 = QHBoxLayout()
        for w in [self.use_mmap, self.use_mlock, self.verbose, self.log_timestamps]:
            s2.addWidget(w)
        self.adv_panel.add_layout(s2)

        self.cuda_visible_devices = QLineEdit(placeholderText="CUDA_VISIBLE_DEVICES")
        self.cuda_visible_devices.setMaximumWidth(120)
        self.cuda_module_loading = QLineEdit(placeholderText="CUDA_MODULE_LOADING")
        self.cuda_module_loading.setText("LAZY")
        self.cuda_module_loading.setMaximumWidth(80)
        s_cuda = QHBoxLayout()
        s_cuda.addWidget(QLabel("CUDA env:"))
        s_cuda.addWidget(self.cuda_visible_devices)
        s_cuda.addWidget(self.cuda_module_loading)
        s_cuda.addStretch(1)
        self.adv_panel.add_layout(s_cuda)

        self.cont_batching = QCheckBox("cont batching")
        self.cont_batching.setChecked(True)
        self.cache_prompt = QCheckBox("cache prompt")
        self.cache_prompt.setChecked(True)
        self.context_shift = QCheckBox("context shift")
        self.no_webui = QCheckBox("no webui")
        self.jinja = QCheckBox("jinja")
        s3 = QHBoxLayout()
        for w in [
            self.cont_batching,
            self.cache_prompt,
            self.context_shift,
            self.no_webui,
            self.jinja,
        ]:
            s3.addWidget(w)
        self.adv_panel.add_layout(s3)

        s_tpl = QHBoxLayout()
        self.use_chat_template = QCheckBox("--chat-template-file")
        self.chat_template_file = QLineEdit(
            placeholderText="Path to .jinja chat template"
        )
        self.chat_template_file.setToolTip(
            "Override the model's built-in chat template with an external .jinja file. "
            "Required for Qwen3.6 tool calls when using the relaxed template."
        )
        self.chat_template_btn = QPushButton("...")
        self.chat_template_btn.setFixedWidth(32)
        self.chat_template_btn.clicked.connect(
            lambda _checked=False: self._browse_chat_template_clicked()
        )
        s_tpl.addWidget(self.use_chat_template)
        s_tpl.addWidget(self.chat_template_file, 1)
        s_tpl.addWidget(self.chat_template_btn)
        self.adv_panel.add_layout(s_tpl)

        self.adv_panel.add_widget(QLabel("Extra params:"))
        self.extra_args = QLineEdit(placeholderText="--top-p 0.9 --min-p 0.05 ...")
        self.adv_panel.add_widget(self.extra_args)

        # === 5. Интеграция ===
        self.int_panel = CollapsiblePanel("Integration (OpenCode / PI)")

        oc_layout = QHBoxLayout()
        oc_layout.addWidget(QLabel("OpenCode JSON:"))
        self.opencode_config_path = QLineEdit(placeholderText="Path to opencode.json")
        oc_btn = QPushButton("...")
        oc_btn.clicked.connect(self._browse_opencode_clicked)
        oc_layout.addWidget(self.opencode_config_path)
        oc_layout.addWidget(oc_btn)
        self.int_panel.add_layout(oc_layout)

        pi_layout = QHBoxLayout()
        pi_layout.addWidget(QLabel("PI JSON:"))
        self.pi_config_path = QLineEdit(placeholderText="Path to PI config.json")
        pi_btn = QPushButton("...")
        pi_btn.clicked.connect(self._browse_pi_clicked)
        pi_layout.addWidget(self.pi_config_path)
        pi_layout.addWidget(pi_btn)
        self.int_panel.add_layout(pi_layout)

        tgt_layout = QHBoxLayout()
        tgt_layout.addWidget(QLabel("Target:"))
        self.integration_target = QComboBox()
        self.integration_target.addItem("OpenCode", "opencode")
        self.integration_target.addItem("PI", "pi")
        tgt_layout.addWidget(self.integration_target)
        self.integration_check_btn = QPushButton("Check")
        tgt_layout.addWidget(self.integration_check_btn)
        self.int_panel.add_layout(tgt_layout)

        self.integration_model_label = QLabel(
            "Model to add: not selected", wordWrap=True
        )
        self.int_panel.add_widget(self.integration_model_label)

        self.integration_models_list = QListWidget()
        self.integration_models_list.setMinimumHeight(80)
        self.int_panel.add_widget(self.integration_models_list)

        act_layout = QHBoxLayout()
        self.integration_add_btn = QPushButton("Add")
        self.integration_remove_btn = QPushButton("Remove")
        act_layout.addWidget(self.integration_add_btn)
        act_layout.addWidget(self.integration_remove_btn)
        self.int_panel.add_layout(act_layout)

        self.integration_status = QLabel(
            "Specify config path and click Check", wordWrap=True
        )
        self.int_panel.add_widget(self.integration_status)

        # === 6. Бенчмарк ===
        self.bench_panel = CollapsiblePanel("Benchmark")
        bp_layout = QHBoxLayout()
        bp_layout.addWidget(QLabel("Prompt (-p):"))
        self.bench_prompt = QSpinBox()
        self.bench_prompt.setRange(16, 4096)
        self.bench_prompt.setValue(128)
        self.bench_prompt.setSingleStep(64)
        bp_layout.addWidget(self.bench_prompt)
        bp_layout.addSpacing(10)
        bp_layout.addWidget(QLabel("Gen (-n):"))
        self.bench_gen = QSpinBox()
        self.bench_gen.setRange(16, 4096)
        self.bench_gen.setValue(256)
        self.bench_gen.setSingleStep(64)
        bp_layout.addWidget(self.bench_gen)
        self.bench_panel.add_layout(bp_layout)

        bench_buttons = QHBoxLayout()
        self.test_btn = QPushButton("Test Speed")
        self.test_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 6px;"
        )
        self.autotune_btn = QPushButton("AutoTune...")
        self.autotune_btn.setToolTip(
            "Open AutoTune, build candidates from current model/context, then run automatic benchmark"
        )
        self.autotune_btn.setStyleSheet(
            "background-color: #673AB7; color: white; font-weight: bold; padding: 6px;"
        )
        bench_buttons.addWidget(self.test_btn)
        bench_buttons.addWidget(self.autotune_btn)
        self.bench_panel.add_layout(bench_buttons)

        self.autotune = AutoTuneWidget()
        self.bench_panel.add_widget(self.autotune)

        # === 7. Preview CLI ===
        g_cli = QGroupBox("CLI Preview")
        self.cli_preview = QLineEdit(
            placeholderText="Command will be displayed here...", readOnly=True
        )
        self.cli_preview.setStyleSheet(
            "background-color: #2a2a2a; color: #b5cea8; font-family: Consolas; padding: 4px;"
        )
        g_cli.setLayout(QVBoxLayout())
        g_cli.layout().addWidget(self.cli_preview)

        # === 8. Итоговый порядок блоков ===
        lay.addWidget(self.paths_panel)
        lay.addWidget(self.adv_panel)
        lay.addWidget(self.models_panel)
        lay.addWidget(self.int_panel)
        lay.addWidget(self.bench_panel)
        lay.addWidget(g_cli)
        lay.addStretch()

        # Collect all widgets that must be locked while server/bench/autotune runs
        self._runtime_lockable = [
            self.model_combo,
            self.auto_params,
            self.use_mmproj,
            self.mmproj_offload,
            self.scan_btn,
            self.gpu_layers,
            self.gpu_auto,
            self.gpu_layers_all,
            self.cpu_moe_layers,
            self.ctx_size,
            self.batch_size,
            self.ubatch_size,
            self.threads,
            self.threads_batch,
            self.cache_type_k,
            self.cache_type_v,
            self.flash_attn,
            self.fit_off,
            self.temperature,
            self.repeat_penalty,
            self.use_mmap,
            self.use_mlock,
            self.verbose,
            self.log_timestamps,
            self.parallel_slots,
            self.kv_unified,
            self.speculative_mtp,
            self.spec_draft_n_max,
            self.spec_draft_gpu_layers,
            self.spec_draft_model_path,
            self.spec_draft_model_btn,
            self.spec_draft_device,
            self.cuda_device,
            self.split_mode,
            self.main_gpu,
            self.ctx_checkpoints,
            self.cache_ram,
            self.host,
            self.port,
            self.cont_batching,
            self.cache_prompt,
            self.context_shift,
            self.no_webui,
            self.jinja,
            self.reasoning_mode,
            self.enable_thinking,
            self.extra_args,
            self.cuda_visible_devices,
            self.cuda_module_loading,
            self.bench_prompt,
            self.bench_gen,
            self.save_preset_btn,
        ]
        self._runtime_lockable.extend(getattr(self, "ctx_quick_buttons", []))

        scroll = QScrollArea(
            widgetResizable=True,
            horizontalScrollBarPolicy=Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        scroll.setWidget(panel)
        scroll.setMinimumWidth(720)
        scroll.setMaximumWidth(940)
        return scroll

    def _build_right_panel(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(4, 4, 4, 4)

        # Вкладки: логи и AutoTune. MemoryVisualizationWidget остаётся скрытым
        # техническим виджетом для совместимости с существующим парсером логов;
        # пользовательские данные по памяти выводятся в Logs.
        from PySide6.QtWidgets import QTabWidget

        self.tabs = QTabWidget()

        # Вкладка логов
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(0, 0, 0, 0)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Logs:"))
        self.autoscroll_logs = QCheckBox("Auto-scroll", checked=True)
        hdr.addWidget(self.autoscroll_logs)
        log_layout.addLayout(hdr)
        self.logs = QTextEdit(readOnly=True, font=QFont("Consolas", 9))
        self.logs.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        log_layout.addWidget(self.logs)
        clr = QPushButton("Clear")
        clr.clicked.connect(self.logs.clear)
        log_layout.addWidget(clr)
        self.tabs.addTab(log_tab, "Logs")

        # Скрытая визуализация памяти: не добавляем вкладку, чтобы не показывать
        # пустой экран "Модель не выбрана" на сборках llama.cpp без buffer logs.
        self.mem_viz = MemoryVisualizationWidget()
        self.mem_viz.setVisible(False)

        lay.addWidget(self.tabs)
        return panel

    def _setup_tooltips(self):
        tips = {
            self.exe_path: (
                "Base folder containing llama-win-cuda-* build folders, e.g. "
                "G:/AIModels/llamacpp/"
            ),
            self.bench_path: "Auto-detected llama-bench.exe",
            self.model_dir: "Root folder for .gguf search",
            self.scan_btn: "Scans models folder",
            self.model_combo: "Selected GGUF model",
            self.auto_params: "Automatically sets parameters",
            self.start_btn: "Starts llama-server",
            self.autotune_btn: "Opens AutoTune and builds a plan from current settings",
            self.speculative_mtp: "Enable MTP speculative decoding when the selected model/package supports it",
            self.spec_draft_model_path: "Separate MTP/draft GGUF path, auto-detected when possible",
            self.reload_btn: "Restarts llama-server with current parameters",
            self.stop_btn: "Stops server",
            self.force_stop_btn: "Force kills llama-server immediately",
            self.autoscroll_logs: "Auto-scroll logs",
            self.tokens_reset_btn: "Save current task token count and reset task counter",
        }
        for widget, text in tips.items():
            if widget:
                widget.setToolTip(text)

    def _load_ui_state(self):
        geo = self.ui_settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        state = self.ui_settings.value("windowState")
        if state:
            self.restoreState(state)

    def save_ui_state(self):
        self.ui_settings.setValue("geometry", self.saveGeometry())
        self.ui_settings.setValue("windowState", self.saveState())

    # === Placeholders ===
    def _browse_exe_clicked(self):
        pass

    def _browse_bench_clicked(self):
        pass

    def _browse_model_dir_clicked(self):
        pass

    def _browse_opencode_clicked(self):
        pass

    def _browse_pi_clicked(self):
        pass

    def _browse_chat_template_clicked(self):
        pass

    def current_config_target(self):
        return self.integration_target.currentData() or "opencode"

    def current_config_path(self):
        return (
            self.pi_config_path.text().strip()
            if self.current_config_target() == "pi"
            else self.opencode_config_path.text().strip()
        )

    def current_base_url(self):
        return f"http://127.0.0.1:{self.port.value()}/v1"

    def current_model_id(self):
        p = self.model_combo.currentData()
        if p:
            return Path(p).stem
        t = self.model_combo.currentText().strip()
        return Path(t).stem if t.lower().endswith(".gguf") and os.path.exists(t) else t
