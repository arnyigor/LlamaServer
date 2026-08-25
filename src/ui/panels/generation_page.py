"""Generation settings builder.

Builds the widgets for the Launch, Sampling and Server nav pages plus the CLI
Preview group. These three pages intentionally share ``CollapsiblePanel``
instances (``sampling_panel``, ``adv_panel``, ``server_panel``) that are created
here and populated across the Launch/Sampling/Server pages, so they are kept
together in a single builder rather than split into independent page classes
(splitting would require risky widget re-parenting).

All widgets are created directly on ``mw`` (the ``MainWindowUI`` instance) so
``main.py`` and the helper methods keep working unchanged.
"""

import os

from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QGridLayout,
    QToolButton,
    QPushButton,
    QProgressBar,
    QTextEdit,
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from src.core.constants import (
    AUTO_SENTINEL,
    STATUS_COLOR_MUTED,
    STATUS_COLOR_MUTED_DARK,
    STATUS_COLOR_PENDING,
    SAMPLING_AUTO_FLOAT,
    SAMPLING_AUTO_INT,
    SAMPLING_LAST_N_AUTO,
    SAMPLING_PENALTY_AUTO,
    SAMPLING_SEED_AUTO,
    SERVER_DEFAULT_SENTINEL,
)
from src.ui.widgets import CollapsiblePanel


class GenerationBuilder:
    """Builds model/launch/sampling/server/CLI widgets onto ``mw``."""

    def __init__(self, mw):
        self.mw = mw
        self._build_model_section()
        self._build_performance_section()
        self._build_sampling_section()
        self._build_cli_section()

    def _build_model_section(self):
        mw = self.mw
        # === 2. Модель ===
        g_model = QGroupBox(mw.tr("Model"))
        lm = QVBoxLayout(g_model)
        lm.setContentsMargins(12, 18, 12, 12)
        lm.setSpacing(8)

        scan_row = QHBoxLayout()
        mw.scan_btn = QPushButton(mw.tr("Scan"))
        scan_row.addWidget(mw.scan_btn)
        lm.addLayout(scan_row)

        mw.scan_status = QLabel(mw.tr("Models not scanned"))
        mw.scan_progress = QProgressBar(visible=False, minimum=0, maximum=0)
        lm.addWidget(mw.scan_status)
        lm.addWidget(mw.scan_progress)

        mw.model_combo = QComboBox()
        mw.model_combo.setEditable(False)
        mw.model_combo.setMinimumHeight(30)
        mw.model_combo.setMaxVisibleItems(25)
        mw.model_combo.setMinimumContentsLength(80)
        mw.model_combo.setStyleSheet(
            "QComboBox { padding-left: 6px; padding-right: 34px; } "
            "QComboBox::drop-down { width: 30px; }"
        )

        lm.addWidget(QLabel(mw.tr("Found GGUF:")))
        lm.addWidget(mw.model_combo)

        mw.auto_params = QCheckBox(mw.tr("Auto setup ctx/GPU/cache by GGUF"))
        mw.auto_params.setChecked(True)
        lm.addWidget(mw.auto_params)

        mw.model_info = QLabel(mw.tr("Select model"))
        mw.model_info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        mw.model_id_label = QLabel(mw.tr(""))
        mw.model_id_label.setStyleSheet("color: #888; font-size: 10px;")
        mw.model_id_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        mw.model_id_label.setWordWrap(True)
        mw.copy_model_btn = QPushButton(mw.tr("Copy model path"))
        mw.copy_model_btn.setFixedHeight(22)
        mw.copy_model_btn.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        mrow = QHBoxLayout()
        mrow.addWidget(mw.copy_model_btn)
        mrow.addWidget(mw.model_id_label, 1)
        lm.addWidget(mw.model_info)
        lm.addLayout(mrow)
        mw.model_group = g_model

    def _build_performance_section(self):
        mw = self.mw
        # === 3. Производительность ===
        mw.g_launch = QGroupBox(mw.tr("Launch settings"))
        launch = QVBoxLayout(mw.g_launch)
        launch.setContentsMargins(12, 18, 12, 12)
        launch.setSpacing(8)

        mw.adv_panel = CollapsiblePanel(
            mw.tr("Память (KV-кэш)"),
            settings_key="panel_adv",
            collapsible=False,
        )
        lperf = mw.adv_panel.content_layout
        lperf.setContentsMargins(8, 6, 8, 6)
        lperf.setSpacing(8)
        mw.sampling_panel = CollapsiblePanel(
            mw.tr("Generation: Sampling and Penalties"),
            settings_key="panel_sampling",
            collapsible=False,
        )
        sampling = mw.sampling_panel.content_layout
        sampling.setContentsMargins(8, 6, 8, 6)
        sampling.setSpacing(8)
        mw.server_panel = CollapsiblePanel(
            mw.tr("Server, Templates and Diagnostics"),
            settings_key="panel_server",
            collapsible=False,
        )
        server_opts = mw.server_panel.content_layout
        server_opts.setContentsMargins(8, 6, 8, 6)
        server_opts.setSpacing(8)
        # Launch settings (g_launch) keeps only context, vision and CUDA.
        # GPU offload, KV cache type, attention and the Memory (KV-cache)
        # panel are moved to the Sampling page as separate blocks.

        mw.launch_summary_group = QGroupBox(mw.tr("Launch preflight"))
        launch_summary = QVBoxLayout(mw.launch_summary_group)
        launch_summary.setContentsMargins(12, 18, 12, 12)
        launch_summary.setSpacing(6)
        mw.preflight_status = QLabel(
            mw.tr("Select a model to estimate launch readiness")
        )
        mw.preflight_status.setWordWrap(True)
        mw.preflight_status.setStyleSheet(
            "font-weight: bold; color: " + STATUS_COLOR_MUTED_DARK + ";"
        )
        mw.preflight_model = QLabel(mw.tr("Model: -"))
        mw.preflight_context = QLabel(mw.tr("Context: -"))
        mw.preflight_kv = QLabel(mw.tr("KV: -"))
        mw.preflight_gpu = QLabel(mw.tr("GPU offload: -"))
        mw.preflight_mtp = QLabel(mw.tr("MTP: -"))
        mw.preflight_endpoint = QLabel(mw.tr("Endpoint: -"))
        mw.preflight_warning = QLabel(mw.tr(""))
        mw.preflight_warning.setWordWrap(True)
        mw.preflight_warning.setStyleSheet("color: " + STATUS_COLOR_PENDING + ";")
        for label in [
            mw.preflight_model,
            mw.preflight_context,
            mw.preflight_kv,
            mw.preflight_gpu,
            mw.preflight_mtp,
            mw.preflight_endpoint,
        ]:
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            launch_summary.addWidget(label)
        launch_summary.addWidget(mw.preflight_warning)

        mw.gpu_layers = QSpinBox()
        mw.gpu_layers.setRange(0, 999)
        mw.gpu_layers.setValue(33)
        mw.gpu_auto = QCheckBox(mw.tr("auto"))
        mw.gpu_auto.setChecked(True)
        mw.gpu_layers_all = QCheckBox(mw.tr("all"))

        def sync_gpu_layer_controls():
            mw.gpu_layers.setDisabled(
                mw.gpu_auto.isChecked() or mw.gpu_layers_all.isChecked()
            )
            mw.gpu_auto.setDisabled(mw.gpu_layers_all.isChecked())

        mw.gpu_auto.toggled.connect(lambda _checked: sync_gpu_layer_controls())
        mw.gpu_layers_all.toggled.connect(lambda _checked: sync_gpu_layer_controls())
        sync_gpu_layer_controls()
        mw.cpu_moe_layers = QSpinBox()
        mw.cpu_moe_layers.setRange(AUTO_SENTINEL, 200)
        mw.cpu_moe_layers.setValue(AUTO_SENTINEL)
        mw.cpu_moe_layers.setSpecialValueText("auto")

        mw.cuda_status_label = QLabel(mw.tr("CUDA build: not checked"))
        mw.cuda_status_label.setWordWrap(True)
        mw.cuda_status_label.setStyleSheet("color: " + STATUS_COLOR_MUTED_DARK + ";")
        mw.cuda_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        r_cuda = QHBoxLayout()
        r_cuda.addWidget(QLabel(mw.tr("CUDA build:")))
        r_cuda.addWidget(mw.launch_cuda_version_combo)
        r_cuda.addWidget(mw.cuda_status_label, 1)
        launch.addLayout(r_cuda)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel(mw.tr("GPU offload (-ngl):")))
        r1.addWidget(mw.gpu_layers)
        r1.addWidget(mw.gpu_auto)
        r1.addWidget(mw.gpu_layers_all)
        r1.addStretch(1)

        mw.ctx_size = QSpinBox()
        mw.ctx_size.setRange(AUTO_SENTINEL, 1048576)
        mw.ctx_size.setSingleStep(512)
        mw.ctx_size.setValue(AUTO_SENTINEL)
        mw.ctx_size.setSpecialValueText("auto")

        r2a = QHBoxLayout()
        r2a.addWidget(QLabel(mw.tr("Context Size (-c):")))
        r2a.addWidget(mw.ctx_size)
        mw.ctx_help_btn = QToolButton()
        mw.ctx_help_btn.setText("?")
        mw.ctx_help_btn.setToolTip("Open detailed context/VRAM guidance")
        r2a.addWidget(mw.ctx_help_btn)
        r2a.addSpacing(10)
        r2a.addWidget(QLabel(mw.tr("CPU MoE (-ncmoe):")))
        r2a.addWidget(mw.cpu_moe_layers)
        mw.ncmoe_help_btn = QToolButton()
        mw.ncmoe_help_btn.setText("?")
        mw.ncmoe_help_btn.setToolTip("Open detailed CPU MoE/VRAM guidance")
        r2a.addWidget(mw.ncmoe_help_btn)
        r2a.addStretch(1)
        launch.addLayout(r2a)

        # Быстрые кнопки контекста — на отдельном ряду, чтобы не перегружать
        # основной ряд spinbox'ов. Стандартные степени 2 (8K..256K); нестандартные
        # 24K/41K/65K убраны (вводят в заблуждение: 40960 = 40K, 65536 = 64K).
        r2b = QHBoxLayout()
        r2b.addWidget(QLabel(mw.tr("Quick:")))
        mw.ctx_quick_buttons = []
        for label, value in [
            ("8K", 8192),
            ("16K", 16384),
            ("32K", 32768),
            ("64K", 65536),
            ("128K", 131072),
            ("256K", 262144),
        ]:
            btn = QPushButton(label)
            btn.setFixedWidth(42 if len(label) <= 3 else 50)
            btn.setFixedHeight(24)
            btn.setToolTip(f"Set Context Size to {value}")
            btn.setProperty("ctx_value", value)
            mw.ctx_quick_buttons.append(btn)
            r2b.addWidget(btn)
            if label == "32K":
                r2b.addSpacing(10)
        r2b.addStretch(1)
        launch.addLayout(r2b)

        # Vision (mmproj) — перенесено из секции модели на страницу Запуск.
        mmproj_row = QHBoxLayout()
        mw.use_mmproj = QCheckBox(mw.tr("Use mmproj"))
        mw.use_mmproj.setChecked(True)
        mw.mmproj_offload = QCheckBox(mw.tr("mmproj offload"))
        mw.mmproj_offload.setChecked(True)
        mmproj_row.addWidget(mw.use_mmproj)
        mmproj_row.addWidget(mw.mmproj_offload)
        launch.addLayout(mmproj_row)

        # GPU offload (-ngl) — блок на странице Сэмплинг.
        gpu_offload_box = QGroupBox(mw.tr("GPU offload (-ngl)"))
        gpu_offload_box.setLayout(r1)
        sampling.addWidget(gpu_offload_box)

        mw.batch_size = QSpinBox()
        mw.batch_size.setRange(AUTO_SENTINEL, 32768)
        mw.batch_size.setSingleStep(128)
        mw.batch_size.setValue(AUTO_SENTINEL)
        mw.batch_size.setSpecialValueText("auto")
        mw.ubatch_size = QSpinBox()
        mw.ubatch_size.setRange(AUTO_SENTINEL, 8192)
        mw.ubatch_size.setSingleStep(64)
        mw.ubatch_size.setValue(AUTO_SENTINEL)
        mw.ubatch_size.setSpecialValueText("auto")

        mw.cache_type_k = QComboBox()
        mw.cache_type_v = QComboBox()
        for ct in ["f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "f32"]:
            mw.cache_type_k.addItem(ct)
            mw.cache_type_v.addItem(ct)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel(mw.tr("KV K / V:")))
        r3.addWidget(mw.cache_type_k)
        r3.addWidget(mw.cache_type_v)
        r3.addStretch(1)
        kv_cache_box = QGroupBox(mw.tr("KV cache type"))
        kv_cache_box.setLayout(r3)
        sampling.addWidget(kv_cache_box)

        # Batch / UBatch — генеративные параметры, на страницу Сэмплинг.
        r_batch = QHBoxLayout()
        r_batch.addWidget(QLabel(mw.tr("Batch / UBatch (-b / -ub):")))
        r_batch.addWidget(mw.batch_size)
        r_batch.addWidget(mw.ubatch_size)
        r_batch.addStretch(1)
        sampling.addLayout(r_batch)

        mw.threads = QSpinBox()
        mw.threads.setRange(1, 64)
        mw.threads.setValue(os.cpu_count() or 4)
        mw.threads_batch = QSpinBox()
        mw.threads_batch.setRange(0, 64)
        mw.threads_batch.setSpecialValueText("same")
        mw.threads_batch.setValue(0)
        r4 = QHBoxLayout()
        r4.addWidget(QLabel(mw.tr("Threads gen / batch (-t / -tb):")))
        r4.addWidget(mw.threads)
        r4.addWidget(mw.threads_batch)
        sampling.addLayout(r4)

        mw.flash_attn = QCheckBox(mw.tr("Flash Attention (-fa)"))
        mw.flash_attn.setChecked(True)
        mw.fit_off = QCheckBox(mw.tr("Fit off (--fit off)"))
        mw.fit_off.setChecked(True)
        r6 = QHBoxLayout()
        r6.addWidget(mw.flash_attn)
        r6.addWidget(mw.fit_off)
        r6.addStretch(1)
        attention_box = QGroupBox(mw.tr("Attention / Fit"))
        attention_box.setLayout(r6)
        sampling.addWidget(attention_box)

        mw.reasoning_mode = QComboBox()
        mw.reasoning_mode.addItems(["off", "auto", "on"])
        mw.reasoning_mode.setCurrentText("off")
        mw.enable_thinking = QComboBox()
        mw.enable_thinking.addItems(["off", "false", "true"])
        mw.enable_thinking.setCurrentText("off")
        mw.reasoning_effort = QComboBox()
        mw.reasoning_effort.addItems(["", "low", "medium", "xhigh"])
        mw.reasoning_effort.setCurrentText("")
        mw.reasoning_preserve = QComboBox()
        mw.reasoning_preserve.addItems(["off", "preserve", "no-preserve"])
        mw.reasoning_preserve.setCurrentText("off")
        mw.reasoning_budget = QSpinBox()
        mw.reasoning_budget.setRange(0, 32767)
        mw.reasoning_budget.setValue(0)
        mw.reasoning_budget_message = QLineEdit(placeholderText="optional")
        r7 = QHBoxLayout()
        r7.addWidget(QLabel(mw.tr("Reasoning (--reasoning):")))
        r7.addWidget(mw.reasoning_mode)
        r7.addSpacing(10)
        r7.addWidget(QLabel(mw.tr("Thinking:")))
        r7.addWidget(mw.enable_thinking)
        r7.addSpacing(10)
        r7.addWidget(QLabel(mw.tr("Effort (--reasoning-effort):")))
        r7.addWidget(mw.reasoning_effort)
        r7.addSpacing(10)
        r7.addWidget(QLabel(mw.tr("Preserve (--reasoning-preserve):")))
        r7.addWidget(mw.reasoning_preserve)
        sampling.addLayout(r7)

        # Budget и Budget msg — вертикально ("друг под другом"): текстовое
        # поле рядом со спинбоксом неудобно, а сообщение удобнее на всю ширину.
        r7b = QVBoxLayout()
        r7b.setSpacing(4)
        row_budget = QHBoxLayout()
        row_budget.addWidget(QLabel(mw.tr("Budget (--reasoning-budget):")))
        row_budget.addWidget(mw.reasoning_budget)
        row_budget.addStretch(1)
        r7b.addLayout(row_budget)
        row_budget_msg = QHBoxLayout()
        row_budget_msg.addWidget(
            QLabel(mw.tr("Budget msg (--reasoning-budget-message):"))
        )
        row_budget_msg.addWidget(mw.reasoning_budget_message, 1)
        r7b.addLayout(row_budget_msg)
        sampling.addLayout(r7b)

        mw.host = QLineEdit(placeholderText="127.0.0.1")
        mw.host.setText("127.0.0.1")
        mw.host.setMaximumWidth(120)
        mw.port = QSpinBox()
        mw.port.setRange(1024, 65535)
        mw.port.setValue(8080)
        mw.parallel_slots = QSpinBox()
        mw.parallel_slots.setRange(AUTO_SENTINEL, 16)
        mw.parallel_slots.setValue(AUTO_SENTINEL)
        mw.parallel_slots.setSpecialValueText("auto")
        r8 = QHBoxLayout()
        r8.addWidget(QLabel(mw.tr("Host:")))
        r8.addWidget(mw.host)
        r8.addWidget(QLabel(mw.tr("Port:")))
        r8.addWidget(mw.port)
        r8.addSpacing(10)
        r8.addWidget(QLabel(mw.tr("Slots (-np):")))
        r8.addWidget(mw.parallel_slots)
        server_opts.addLayout(r8)

        mw.kv_unified = QCheckBox(mw.tr("KV unified (-kvu)"))
        mw.speculative_mtp = QCheckBox(mw.tr("MTP speculative"))
        mw.spec_draft_n_max = QSpinBox()
        mw.spec_draft_n_max.setRange(1, 32)
        mw.spec_draft_n_max.setValue(8)
        mw.spec_draft_n_max.setToolTip(
            "Maximum speculative MTP tokens. Coding default: 8; conservative: 2-4; aggressive: 16."
        )
        mw.spec_draft_p_min = QDoubleSpinBox()
        mw.spec_draft_p_min.setRange(0.0, 1.0)
        mw.spec_draft_p_min.setDecimals(2)
        mw.spec_draft_p_min.setSingleStep(0.05)
        mw.spec_draft_p_min.setValue(0.8)
        mw.spec_draft_p_min.setToolTip(
            "Minimum MTP confidence. 0.8 avoids expensive long speculation when the draft head is uncertain."
        )
        mw.spec_draft_gpu_layers = QLineEdit(placeholderText="all")
        mw.spec_draft_gpu_layers.setText("all")
        mw.spec_draft_gpu_layers.setMaximumWidth(60)
        r8b = QHBoxLayout()
        r8b.addWidget(mw.kv_unified)
        r8b.addWidget(mw.speculative_mtp)
        r8b.addSpacing(10)
        r8b.addWidget(QLabel(mw.tr("MTP n-max / p-min / draft ngl:")))
        r8b.addWidget(mw.spec_draft_n_max)
        r8b.addWidget(mw.spec_draft_p_min)
        r8b.addWidget(mw.spec_draft_gpu_layers)
        sampling.addLayout(r8b)

        r8b2 = QHBoxLayout()
        r8b2.addWidget(QLabel(mw.tr("MTP draft GGUF:")))
        mw.spec_draft_model_path = QLineEdit(
            placeholderText="Auto-detected, or browse for separate MTP GGUF"
        )
        mw.spec_draft_model_path.setToolTip(
            "Optional separate MTP/draft GGUF. Gemma 4 packages often include it in an MTP folder; Qwen3.6 may require a separate file."
        )
        mw.spec_draft_model_btn = QPushButton(mw.tr("..."))
        mw.spec_draft_model_btn.setFixedWidth(32)
        mw.spec_draft_model_btn.clicked.connect(
            lambda _checked=False: mw._browse_mtp_draft_clicked()
        )
        r8b2.addWidget(mw.spec_draft_model_path, 1)
        r8b2.addWidget(mw.spec_draft_model_btn)
        sampling.addLayout(r8b2)

        mw.cuda_device = QLineEdit(placeholderText="CUDA0")
        mw.cuda_device.setMaximumWidth(80)
        mw.spec_draft_device = QLineEdit(placeholderText="CUDA0")
        mw.spec_draft_device.setMaximumWidth(80)
        mw.split_mode = QComboBox()
        mw.split_mode.addItems(["", "none", "layer", "row"])
        mw.main_gpu = QSpinBox()
        mw.main_gpu.setRange(AUTO_SENTINEL, 16)
        mw.main_gpu.setSpecialValueText("auto")
        mw.main_gpu.setValue(AUTO_SENTINEL)
        r8c = QHBoxLayout()
        r8c.addWidget(QLabel(mw.tr("Device:")))
        r8c.addWidget(mw.cuda_device)
        r8c.addWidget(QLabel(mw.tr("Draft device:")))
        r8c.addWidget(mw.spec_draft_device)
        r8c.addWidget(QLabel(mw.tr("Split:")))
        r8c.addWidget(mw.split_mode)
        r8c.addWidget(QLabel(mw.tr("Main GPU:")))
        r8c.addWidget(mw.main_gpu)
        sampling.addLayout(r8c)

        mw.ctx_checkpoints = QSpinBox()
        mw.ctx_checkpoints.setRange(AUTO_SENTINEL, 128)
        mw.ctx_checkpoints.setSpecialValueText("default")
        mw.ctx_checkpoints.setValue(AUTO_SENTINEL)
        mw.cache_ram = QSpinBox()
        mw.cache_ram.setRange(SERVER_DEFAULT_SENTINEL, 262144)
        mw.cache_ram.setSpecialValueText("default")
        mw.cache_ram.setValue(SERVER_DEFAULT_SENTINEL)
        r9 = QHBoxLayout()
        r9.addWidget(QLabel(mw.tr("Ctx Checkpoints:")))
        r9.addWidget(mw.ctx_checkpoints)
        r9.addSpacing(10)
        r9.addWidget(QLabel(mw.tr("Cache RAM (MiB):")))
        r9.addWidget(mw.cache_ram)
        lperf.addLayout(r9)

    def _build_sampling_section(self):
        mw = self.mw
        # === 4. Generation / Sampling ===
        # Layout-объекты панелей создаются в performance-секции; здесь и
        # ниже (server/diagnostics-часть) используем те же самые объекты.
        lperf = mw.adv_panel.content_layout
        sampling = mw.sampling_panel.content_layout
        server_opts = mw.server_panel.content_layout
        mw.temperature = QDoubleSpinBox()
        mw.temperature.setRange(SAMPLING_AUTO_FLOAT, 2.0)
        mw.temperature.setSingleStep(0.1)
        mw.temperature.setValue(SAMPLING_AUTO_FLOAT)
        mw.temperature.setDecimals(2)
        mw.temperature.setSpecialValueText("auto")
        mw.repeat_penalty = QDoubleSpinBox()
        mw.repeat_penalty.setRange(SAMPLING_AUTO_FLOAT, 2.0)
        mw.repeat_penalty.setSingleStep(0.01)
        mw.repeat_penalty.setValue(SAMPLING_AUTO_FLOAT)
        mw.repeat_penalty.setDecimals(2)
        mw.repeat_penalty.setSpecialValueText("auto")
        mw.top_k = QSpinBox()
        mw.top_k.setRange(SAMPLING_AUTO_INT, 10000)
        mw.top_k.setValue(SAMPLING_AUTO_INT)
        mw.top_k.setSpecialValueText("auto")
        mw.top_p = QDoubleSpinBox()
        mw.top_p.setRange(SAMPLING_AUTO_FLOAT, 1.0)
        mw.top_p.setSingleStep(0.01)
        mw.top_p.setDecimals(3)
        mw.top_p.setValue(SAMPLING_AUTO_FLOAT)
        mw.top_p.setSpecialValueText("auto")
        mw.min_p = QDoubleSpinBox()
        mw.min_p.setRange(SAMPLING_AUTO_FLOAT, 1.0)
        mw.min_p.setSingleStep(0.01)
        mw.min_p.setDecimals(3)
        mw.min_p.setValue(SAMPLING_AUTO_FLOAT)
        mw.min_p.setSpecialValueText("auto")
        mw.typical_p = QDoubleSpinBox()
        mw.typical_p.setRange(SAMPLING_AUTO_FLOAT, 1.0)
        mw.typical_p.setSingleStep(0.01)
        mw.typical_p.setDecimals(3)
        mw.typical_p.setValue(SAMPLING_AUTO_FLOAT)
        mw.typical_p.setSpecialValueText("auto")
        mw.repeat_last_n = QSpinBox()
        mw.repeat_last_n.setRange(SAMPLING_LAST_N_AUTO, 1048576)
        mw.repeat_last_n.setValue(SAMPLING_LAST_N_AUTO)
        mw.repeat_last_n.setSpecialValueText("auto")
        mw.presence_penalty = QDoubleSpinBox()
        mw.presence_penalty.setRange(SAMPLING_PENALTY_AUTO, 2.0)
        mw.presence_penalty.setSingleStep(0.05)
        mw.presence_penalty.setDecimals(2)
        mw.presence_penalty.setValue(SAMPLING_PENALTY_AUTO)
        mw.presence_penalty.setSpecialValueText("auto")
        mw.frequency_penalty = QDoubleSpinBox()
        mw.frequency_penalty.setRange(SAMPLING_PENALTY_AUTO, 2.0)
        mw.frequency_penalty.setSingleStep(0.05)
        mw.frequency_penalty.setDecimals(2)
        mw.frequency_penalty.setValue(SAMPLING_PENALTY_AUTO)
        mw.frequency_penalty.setSpecialValueText("auto")
        mw.seed = QSpinBox()
        mw.seed.setRange(SAMPLING_SEED_AUTO, 2147483647)
        mw.seed.setValue(SAMPLING_SEED_AUTO)
        mw.seed.setSpecialValueText("auto")

        sampling_grid = QGridLayout()
        sampling_grid.setHorizontalSpacing(10)
        sampling_grid.setVerticalSpacing(6)
        sampling_fields = [
            ("Temperature:", mw.temperature),
            ("Top K:", mw.top_k),
            ("Top P:", mw.top_p),
            ("Min P:", mw.min_p),
            ("Typical P:", mw.typical_p),
            ("Seed:", mw.seed),
            ("Repeat penalty:", mw.repeat_penalty),
            ("Repeat last N:", mw.repeat_last_n),
            ("Presence penalty:", mw.presence_penalty),
            ("Frequency penalty:", mw.frequency_penalty),
        ]
        for index, (label, widget) in enumerate(sampling_fields):
            row, column = divmod(index, 2)
            sampling_grid.addWidget(QLabel(label), row, column * 2)
            sampling_grid.addWidget(widget, row, column * 2 + 1)
        sampling_help = {
            mw.temperature: "Randomness (0.0–2.0).\nCLI: --temp\nauto = server default",
            mw.top_k: "Keep K most likely tokens; 0 disables.\nCLI: --top-k",
            mw.top_p: "Nucleus sampling threshold; 1.0 disables.\nCLI: --top-p",
            mw.min_p: "Min probability relative to best token; 0 disables.\nCLI: --min-p",
            mw.typical_p: "Locally typical sampling; 1.0 disables.\nCLI: --typical",
            mw.seed: "RNG seed. -1 = random; auto omits the flag.\nCLI: --seed",
            mw.repeat_penalty: "Penalty for repeated token sequences; 1.0 disables.\nCLI: --repeat-penalty",
            mw.repeat_last_n: "Recent tokens penalized; -1 = full context.\nCLI: --repeat-last-n",
            mw.presence_penalty: "Penalty based on token presence; 0 disables.\nCLI: --presence-penalty",
            mw.frequency_penalty: "Penalty based on token repetition count; 0 disables.\nCLI: --frequency-penalty",
        }
        for widget, help_text in sampling_help.items():
            widget.setToolTip(help_text)
        sampling.addLayout(sampling_grid)

        mw.use_mlock = QCheckBox(mw.tr("mlock"))
        mw.verbose = QCheckBox(mw.tr("verbose"))
        mw.log_timestamps = QCheckBox(mw.tr("log timestamps"))
        memory_flags = QHBoxLayout()
        memory_flags.addWidget(mw.use_mlock)
        memory_flags.addStretch(1)
        lperf.addLayout(memory_flags)
        diagnostics_flags = QHBoxLayout()
        diagnostics_flags.addWidget(mw.verbose)
        diagnostics_flags.addWidget(mw.log_timestamps)
        diagnostics_flags.addStretch(1)
        server_opts.addLayout(diagnostics_flags)

        mw.cuda_visible_devices = QLineEdit(placeholderText="CUDA_VISIBLE_DEVICES")
        mw.cuda_visible_devices.setMaximumWidth(120)
        mw.cuda_module_loading = QLineEdit(placeholderText="CUDA_MODULE_LOADING")
        mw.cuda_module_loading.setText("LAZY")
        mw.cuda_module_loading.setMaximumWidth(80)
        s_cuda = QHBoxLayout()
        s_cuda.addWidget(QLabel(mw.tr("CUDA env:")))
        s_cuda.addWidget(mw.cuda_visible_devices)
        s_cuda.addWidget(mw.cuda_module_loading)
        s_cuda.addStretch(1)
        lperf.addLayout(s_cuda)

        mw.context_shift = QCheckBox(mw.tr("context shift"))
        mw.no_webui = QCheckBox(mw.tr("no webui"))
        mw.jinja = QCheckBox(mw.tr("jinja"))
        s3 = QHBoxLayout()
        for w in [
            mw.context_shift,
            mw.no_webui,
            mw.jinja,
        ]:
            s3.addWidget(w)
        server_opts.addLayout(s3)

        s_tpl = QHBoxLayout()
        mw.use_chat_template = QCheckBox(mw.tr("--chat-template-file"))
        mw.chat_template_file = QLineEdit(
            placeholderText="Path to .jinja chat template"
        )
        mw.chat_template_file.setToolTip(
            "Override the model's built-in chat template with an external .jinja file. "
            "Required for Qwen3.6 tool calls when using the relaxed template."
        )
        mw.chat_template_btn = QPushButton(mw.tr("..."))
        mw.chat_template_btn.setFixedWidth(32)
        mw.chat_template_btn.clicked.connect(
            lambda _checked=False: mw._browse_chat_template_clicked()
        )
        s_tpl.addWidget(mw.use_chat_template)
        s_tpl.addWidget(mw.chat_template_file, 1)
        s_tpl.addWidget(mw.chat_template_btn)
        server_opts.addLayout(s_tpl)

        sampling.addWidget(
            QLabel(mw.tr("Extra params (only uncommon llama-server flags):"))
        )
        mw.extra_args = QLineEdit()
        mw.extra_args.setPlaceholderText(
            "--dry-multiplier 0.8 --xtc-probability 0.1 ..."
        )
        sampling.addWidget(mw.extra_args)

    def _build_cli_section(self):
        mw = self.mw
        # === 7. Preview CLI ===
        mw.cli_group = QGroupBox(mw.tr("CLI Preview"))
        g_cli = mw.cli_group
        cli_layout = QVBoxLayout()
        cli_controls = QHBoxLayout()
        mw.cli_manual_mode = QCheckBox(mw.tr("Edit CLI"))
        mw.cli_manual_mode.setToolTip(
            "Enable direct command editing. Apply CLI parses known flags back into UI and keeps unknown flags in Extra params."
        )
        mw.cli_apply_btn = QPushButton(mw.tr("Apply CLI"))
        mw.cli_apply_btn.setEnabled(False)
        mw.cli_apply_btn.setToolTip(
            "Parse the edited command: known flags update UI controls, unknown flags go to Extra params."
        )
        mw.cli_copy_btn = QPushButton(mw.tr("Copy CLI"))
        mw.cli_copy_btn.setToolTip("Copy the current generated command line.")
        mw.cli_import_btn = QPushButton(mw.tr("Import CLI"))
        mw.cli_import_btn.setToolTip(
            "Read a llama-server command line from the clipboard and apply it to settings."
        )
        mw.cli_status = QLabel(mw.tr("Generated from UI"))
        mw.cli_status.setStyleSheet("color: " + STATUS_COLOR_MUTED + ";")
        cli_controls.addWidget(mw.cli_manual_mode)
        cli_controls.addWidget(mw.cli_apply_btn)
        cli_controls.addWidget(mw.cli_copy_btn)
        cli_controls.addWidget(mw.cli_import_btn)
        cli_controls.addWidget(mw.cli_status, 1)
        mw.cli_preview = QTextEdit()
        mw.cli_preview.setReadOnly(True)
        mw.cli_preview.setPlaceholderText("Command will be displayed here...")
        mw.cli_preview.setMinimumHeight(60)
        mw.cli_preview.setMaximumHeight(100)
        mw.cli_preview.setFont(QFont("Consolas", 9))
        mw.cli_preview.setStyleSheet(
            "background-color: #2a2a2a; color: #b5cea8; padding: 4px;"
        )
        g_cli.setLayout(cli_layout)
        cli_layout.addLayout(cli_controls)
        cli_layout.addWidget(mw.cli_preview)
