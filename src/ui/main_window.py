"""Главное окно и панели интерфейса."""

import os
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
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QFont

from src.ui.widgets import CollapsiblePanel


class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLama.cpp GUI Manager")
        self.setGeometry(100, 100, 1150, 720)
        self.setMinimumSize(900, 560)

        self.models = []
        self.models_by_path = {}
        self.loading_profile = False

        self.ui_settings = QSettings("LlamaServerGUI", "UIState")

        self._setup_ui()
        self._setup_tooltips()
        self._load_ui_state()

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
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([420, 730])
        main_layout.addWidget(splitter)

    def _build_left_panel(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        # === 1. Пути ===
        g_paths = QGroupBox("📁 Пути")
        lp = QVBoxLayout(g_paths)

        self.exe_path = QLineEdit(placeholderText="Путь к llama-server.exe")
        self.bench_path = QLineEdit(placeholderText="Путь к llama-bench.exe (авто)")
        self.model_dir = QLineEdit(placeholderText="Базовая папка с моделями")

        for line, label, slot in [
            (self.exe_path, "Exe:", "_browse_exe_clicked"),
            (self.bench_path, "Bench:", "_browse_bench_clicked"),
            (self.model_dir, "Models:", "_browse_model_dir_clicked"),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(line, 1)
            btn = QPushButton("📂")
            btn.setFixedWidth(32)
            btn.clicked.connect(getattr(self, slot))
            row.addWidget(btn)
            lp.addLayout(row)

        upd_row = QHBoxLayout()
        self.update_llama_btn = QPushButton("🔄 Update llama.cpp")
        self.update_status = QLabel("idle", wordWrap=True)
        upd_row.addWidget(self.update_llama_btn)
        upd_row.addWidget(self.update_status, 1)
        lp.addLayout(upd_row)

        self.update_progress = QProgressBar(visible=False, minimum=0, maximum=100)
        lp.addWidget(self.update_progress)
        lay.addWidget(g_paths)

        # === 2. Модель ===
        g_model = QGroupBox("🤖 Модель")
        lm = QVBoxLayout(g_model)

        scan_row = QHBoxLayout()
        self.scan_btn = QPushButton("🔍 Сканировать")
        scan_row.addWidget(self.scan_btn)
        lm.addLayout(scan_row)

        self.scan_status = QLabel("Модели не сканировались")
        self.scan_progress = QProgressBar(visible=False, minimum=0, maximum=0)
        lm.addWidget(self.scan_status)
        lm.addWidget(self.scan_progress)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        lm.addWidget(QLabel("Найденные GGUF:"))
        lm.addWidget(self.model_combo)

        self.auto_params = QCheckBox("Автонастройка ctx/GPU/cache по GGUF")
        self.auto_params.setChecked(True)
        lm.addWidget(self.auto_params)

        mmproj_row = QHBoxLayout()
        self.use_mmproj = QCheckBox("Использовать mmproj")
        self.use_mmproj.setChecked(True)
        self.mmproj_offload = QCheckBox("mmproj offload")
        self.mmproj_offload.setChecked(True)
        mmproj_row.addWidget(self.use_mmproj)
        mmproj_row.addWidget(self.mmproj_offload)
        lm.addLayout(mmproj_row)

        self.model_info = QLabel("Выберите модель")
        lm.addWidget(self.model_info)
        lay.addWidget(g_model)

        self.speed_label = QLabel("Скорость: —")
        self.speed_label.setStyleSheet(
            "color: #4ec9b0; font-family: Consolas; font-weight: bold;"
        )
        lay.addWidget(self.speed_label)

        # === 3. Производительность ===
        g_perf = QGroupBox("🚀 Производительность и память")
        lperf = QVBoxLayout(g_perf)

        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(0, 200)
        self.gpu_layers.setValue(33)
        self.gpu_auto = QCheckBox("auto")
        self.gpu_auto.setChecked(True)
        self.gpu_auto.toggled.connect(self.gpu_layers.setDisabled)
        self.gpu_layers.setDisabled(True)
        self.cpu_moe_layers = QSpinBox()
        self.cpu_moe_layers.setRange(-1, 200)
        self.cpu_moe_layers.setValue(-1)
        self.cpu_moe_layers.setSpecialValueText("auto")

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("GPU Layers (-ngl):"))
        r1.addWidget(self.gpu_layers)
        r1.addWidget(self.gpu_auto)
        r1.addSpacing(10)
        r1.addWidget(QLabel("CPU MoE (-ncmoe):"))
        r1.addWidget(self.cpu_moe_layers)
        lperf.addLayout(r1)

        self.ctx_size = QSpinBox()
        self.ctx_size.setRange(-1, 1048576)
        self.ctx_size.setSingleStep(512)
        self.ctx_size.setValue(-1)
        self.ctx_size.setSpecialValueText("auto")
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Context Size (-c):"))
        r2.addWidget(self.ctx_size)
        lperf.addLayout(r2)

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
        lperf.addLayout(r3)

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
        lperf.addLayout(r5)

        self.flash_attn = QCheckBox("Flash Attention (-fa)")
        self.flash_attn.setChecked(True)
        self.fit_off = QCheckBox("Fit off (--fit off)")
        self.fit_off.setChecked(True)
        r6 = QHBoxLayout()
        r6.addWidget(self.flash_attn)
        r6.addWidget(self.fit_off)
        lperf.addLayout(r6)

        self.reasoning_mode = QComboBox()
        self.reasoning_mode.addItems(["off", "auto", "on"])
        self.reasoning_mode.setCurrentText("off")
        r7 = QHBoxLayout()
        r7.addWidget(QLabel("Reasoning (-rea):"))
        r7.addWidget(self.reasoning_mode)
        lperf.addLayout(r7)

        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(8080)
        self.parallel_slots = QSpinBox()
        self.parallel_slots.setRange(-1, 16)
        self.parallel_slots.setValue(-1)
        self.parallel_slots.setSpecialValueText("auto")
        r8 = QHBoxLayout()
        r8.addWidget(QLabel("Port:"))
        r8.addWidget(self.port)
        r8.addSpacing(10)
        r8.addWidget(QLabel("Slots (-np):"))
        r8.addWidget(self.parallel_slots)
        lperf.addLayout(r8)

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
        lay.addWidget(g_perf)

        # === 4. Спойлеры ===
        self.adv_panel = CollapsiblePanel("⚙️ Сэмплинг, отладка и сервер")
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

        self.adv_panel.add_widget(QLabel("Доп. параметры:"))
        self.extra_args = QLineEdit(placeholderText="--top-p 0.9 --min-p 0.05 ...")
        self.adv_panel.add_widget(self.extra_args)
        lay.addWidget(self.adv_panel)

        # === 5. Интеграция ===
        self.int_panel = CollapsiblePanel("🔌 Интеграция (OpenCode / PI)")

        oc_layout = QHBoxLayout()
        oc_layout.addWidget(QLabel("OpenCode JSON:"))
        self.opencode_config_path = QLineEdit(placeholderText="Путь к opencode.json")
        oc_btn = QPushButton("📂")
        oc_btn.clicked.connect(self._browse_opencode_clicked)
        oc_layout.addWidget(self.opencode_config_path)
        oc_layout.addWidget(oc_btn)
        self.int_panel.add_layout(oc_layout)

        pi_layout = QHBoxLayout()
        pi_layout.addWidget(QLabel("PI JSON:"))
        self.pi_config_path = QLineEdit(placeholderText="Путь к PI config.json")
        pi_btn = QPushButton("📂")
        pi_btn.clicked.connect(self._browse_pi_clicked)
        pi_layout.addWidget(self.pi_config_path)
        pi_layout.addWidget(pi_btn)
        self.int_panel.add_layout(pi_layout)

        tgt_layout = QHBoxLayout()
        tgt_layout.addWidget(QLabel("Цель:"))
        self.integration_target = QComboBox()
        self.integration_target.addItem("OpenCode", "opencode")
        self.integration_target.addItem("PI", "pi")
        tgt_layout.addWidget(self.integration_target)
        self.integration_check_btn = QPushButton("Проверить")
        tgt_layout.addWidget(self.integration_check_btn)
        self.int_panel.add_layout(tgt_layout)

        self.integration_model_label = QLabel(
            "Модель для добавления: не выбрана", wordWrap=True
        )
        self.int_panel.add_widget(self.integration_model_label)

        self.integration_models_list = QListWidget()
        self.integration_models_list.setMinimumHeight(80)
        self.int_panel.add_widget(self.integration_models_list)

        act_layout = QHBoxLayout()
        self.integration_add_btn = QPushButton("Добавить")
        self.integration_remove_btn = QPushButton("Удалить")
        act_layout.addWidget(self.integration_add_btn)
        act_layout.addWidget(self.integration_remove_btn)
        self.int_panel.add_layout(act_layout)

        self.integration_status = QLabel(
            "Укажите путь к конфигу и нажмите Проверить", wordWrap=True
        )
        self.int_panel.add_widget(self.integration_status)
        lay.addWidget(self.int_panel)

        # === 6. Бенчмарк ===
        self.bench_panel = CollapsiblePanel("🧪 Бенчмарк")
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

        self.test_btn = QPushButton("🧪 Тестировать скорость")
        self.test_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 6px;"
        )
        self.bench_panel.add_widget(self.test_btn)
        lay.addWidget(self.bench_panel)

        # === 7. Preview CLI ===
        g_cli = QGroupBox("👁️ Preview CLI")
        self.cli_preview = QLineEdit(
            placeholderText="Команда будет отображаться здесь...", readOnly=True
        )
        self.cli_preview.setStyleSheet(
            "background-color: #2a2a2a; color: #b5cea8; font-family: Consolas; padding: 4px;"
        )
        g_cli.setLayout(QVBoxLayout())
        g_cli.layout().addWidget(self.cli_preview)
        lay.addWidget(g_cli)

        # === 8. Кнопки управления ===
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ Старт Server")
        self.start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        self.stop_btn = QPushButton("⏹ Стоп", enabled=False)
        self.stop_btn.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; padding: 8px;"
        )
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        lay.addLayout(btn_row)
        lay.addStretch()

        scroll = QScrollArea(
            widgetResizable=True,
            horizontalScrollBarPolicy=Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        scroll.setWidget(panel)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(520)
        return scroll

    def _build_right_panel(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(4, 4, 4, 4)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Логи:"))
        self.autoscroll_logs = QCheckBox("Автоскролл", checked=True)
        hdr.addWidget(self.autoscroll_logs)
        lay.addLayout(hdr)
        self.logs = QTextEdit(readOnly=True, font=QFont("Consolas", 9))
        self.logs.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        lay.addWidget(self.logs)
        clr = QPushButton("🧹 Очистить логи")
        clr.clicked.connect(self.logs.clear)
        lay.addWidget(clr)
        return panel

    def _setup_tooltips(self):
        tips = {
            self.exe_path: "Путь к llama-server.exe",
            self.bench_path: "Путь к llama-bench.exe",
            self.model_dir: "Корневая папка для поиска .gguf",
            self.scan_btn: "Сканирует папку моделей",
            self.model_combo: "Выбранная GGUF-модель",
            self.auto_params: "Автоматически выставляет параметры",
            self.start_btn: "Запускает llama-server",
            self.stop_btn: "Останавливает сервер",
            self.autoscroll_logs: "Автопрокрутка логов",
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
