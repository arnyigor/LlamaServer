"""LlamaServer GUI - точка входа.

Модульное PySide6 приложение для управления llama-server и llama-bench.
"""

import sys
import json
import os
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import (
    QApplication,
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
    QSystemTrayIcon,
    QMenu,
    QSplitter,
)
from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import (
    QFont,
    QColor,
    QTextCharFormat,
    QTextCursor,
    QAction,
    QIcon,
)

# Импорты из модулей проекта
from src.core.constants import (
    DEFAULT_LOCAL_BASE_URL,
    KILL_TIMEOUT_BENCHMARK,
    KILL_TIMEOUT_SERVER,
    LLAMA_ALLOWED_FLAGS,
    LLAMACPP_PROVIDER_ID,
    MAX_LOG_LINES,
)
from src.core.gguf_parser import extract_model_info
from src.ui.widgets import CollapsiblePanel
from src.services.threads import ModelScanner, LlamaCppUpdater
from src.services.integration import (
    ensure_opencode_llamacpp_provider,
    ensure_pi_llamacpp_provider,
    get_model_ids,
)
from src.utils.file_utils import (
    load_or_create_json,
    validate_path,
    write_json_file_safely,
)


class LlamaGUI(QMainWindow):
    """Главное окно приложения LlamaServer GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLama.cpp GUI Manager")
        self.setGeometry(100, 100, 1150, 720)
        self.setMinimumSize(900, 560)

        # Процессы
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.stateChanged.connect(self.handle_state)

        self.bench_process = QProcess()
        self.bench_process.readyReadStandardOutput.connect(self.handle_bench_stdout)
        self.bench_process.readyReadStandardError.connect(self.handle_bench_stderr)
        self.bench_process.finished.connect(self.handle_bench_finished)

        # Файлы и данные
        self.profiles_file = "profiles.json"
        self.settings_file = "settings.json"
        self.profiles: Dict[str, Any] = {}
        self.settings: Dict[str, Any] = {}
        self.models: List[Dict[str, Any]] = []
        self.models_by_path: Dict[str, Dict[str, Any]] = {}

        # Потоки
        self.scanner: Optional[ModelScanner] = None
        self.updater: Optional[LlamaCppUpdater] = None

        # Состояние
        self.loading_profile = False
        self.server_stop_requested = False
        self.bench_stop_requested = False
        self.scan_cancel_requested = False

        # Tray icon
        self.tray_icon: Optional[QSystemTrayIcon] = None
        self.tray_menu: Optional[QMenu] = None

        # Batch logging
        self.log_buffer: List[tuple] = []
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.flush_log_buffer)
        self.log_timer.start(100)  # 100ms

        self.setup_ui()
        self.setup_tray()
        self.load_data()
        QTimer.singleShot(250, self.auto_scan_models)

    def setup_ui(self):
        """Настройка пользовательского интерфейса."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # === ЛЕВАЯ ПАНЕЛЬ ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # 1. Пути
        path_group = QGroupBox("📁 Пути")
        path_layout = QVBoxLayout(path_group)

        exe_layout = QHBoxLayout()
        self.exe_path = QLineEdit()
        self.exe_path.setPlaceholderText("Путь к llama-server.exe")
        self.exe_path.textChanged.connect(self.auto_detect_bench)
        exe_btn = QPushButton("Обзор")
        exe_btn.clicked.connect(self.browse_exe)
        exe_layout.addWidget(self.exe_path)
        exe_layout.addWidget(exe_btn)
        path_layout.addLayout(exe_layout)

        bench_layout = QHBoxLayout()
        self.bench_path = QLineEdit()
        self.bench_path.setPlaceholderText("Путь к llama-bench.exe (авто)")
        bench_btn = QPushButton("Обзор")
        bench_btn.clicked.connect(self.browse_bench)
        bench_layout.addWidget(QLabel("Benchmark:"))
        bench_layout.addWidget(self.bench_path)
        bench_layout.addWidget(bench_btn)
        path_layout.addLayout(bench_layout)

        update_layout = QHBoxLayout()
        self.update_llama_btn = QPushButton("Update llama.cpp")
        self.update_llama_btn.clicked.connect(self.update_llamacpp)
        self.update_status = QLabel("llama.cpp updater idle")
        self.update_status.setWordWrap(True)
        self.update_progress = QProgressBar()
        self.update_progress.setRange(0, 100)
        self.update_progress.setVisible(False)
        update_layout.addWidget(self.update_llama_btn)
        update_layout.addWidget(self.update_status, 1)
        path_layout.addLayout(update_layout)
        path_layout.addWidget(self.update_progress)

        model_dir_layout = QHBoxLayout()
        self.model_dir = QLineEdit()
        self.model_dir.setPlaceholderText("Базовая папка с моделями")
        model_dir_btn = QPushButton("Обзор")
        model_dir_btn.clicked.connect(self.browse_model_dir)
        self.scan_btn = QPushButton("🔍 Сканировать")
        self.scan_btn.clicked.connect(self.scan_models)
        model_dir_layout.addWidget(self.model_dir)
        model_dir_layout.addWidget(model_dir_btn)
        model_dir_layout.addWidget(self.scan_btn)
        path_layout.addLayout(model_dir_layout)

        self.scan_status = QLabel("Модели не сканировались")
        self.scan_progress = QProgressBar()
        self.scan_progress.setRange(0, 0)
        self.scan_progress.setVisible(False)
        path_layout.addWidget(self.scan_status)
        path_layout.addWidget(self.scan_progress)
        left_layout.addWidget(path_group)

        # 2. Модель
        model_group = QGroupBox("🤖 Модель")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        model_layout.addWidget(QLabel("Найденные GGUF:"))
        model_layout.addWidget(self.model_combo)
        self.auto_params = QCheckBox("Автонастройка ctx/GPU/cache по GGUF")
        self.auto_params.setChecked(True)
        model_layout.addWidget(self.auto_params)
        mmproj_layout = QHBoxLayout()
        self.use_mmproj = QCheckBox("Использовать mmproj, если найден")
        self.use_mmproj.setChecked(True)
        self.mmproj_offload = QCheckBox("mmproj offload")
        self.mmproj_offload.setChecked(True)
        mmproj_layout.addWidget(self.use_mmproj)
        mmproj_layout.addWidget(self.mmproj_offload)
        model_layout.addLayout(mmproj_layout)
        self.model_info = QLabel("Выберите модель")
        self.model_info.setWordWrap(True)
        model_layout.addWidget(self.model_info)
        left_layout.addWidget(model_group)

        # 3. Производительность (ВСЕГДА ВИДИМО)
        perf_group = QGroupBox("🚀 Производительность и память")
        perf_layout = QVBoxLayout(perf_group)

        gpu_moe_layout = QHBoxLayout()
        gpu_moe_layout.addWidget(QLabel("GPU Layers (-ngl):"))
        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(0, 200)
        self.gpu_layers.setValue(33)
        self.gpu_auto = QCheckBox("auto")
        self.gpu_auto.setChecked(True)
        self.gpu_auto.toggled.connect(self.gpu_layers.setDisabled)
        self.gpu_layers.setDisabled(True)
        gpu_moe_layout.addWidget(self.gpu_layers)
        gpu_moe_layout.addWidget(self.gpu_auto)
        gpu_moe_layout.addSpacing(10)
        gpu_moe_layout.addWidget(QLabel("CPU MoE (-ncmoe):"))
        self.cpu_moe_layers = QSpinBox()
        self.cpu_moe_layers.setRange(0, 200)
        self.cpu_moe_layers.setValue(0)
        gpu_moe_layout.addWidget(self.cpu_moe_layers)
        perf_layout.addLayout(gpu_moe_layout)

        ctx_layout = QHBoxLayout()
        ctx_layout.addWidget(QLabel("Context Size (-c):"))
        self.ctx_size = QSpinBox()
        self.ctx_size.setRange(512, 1048576)
        self.ctx_size.setSingleStep(512)
        self.ctx_size.setValue(4096)
        ctx_layout.addWidget(self.ctx_size)
        perf_layout.addLayout(ctx_layout)

        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch / UBatch (-b / -ub):"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(128, 32768)
        self.batch_size.setSingleStep(128)
        self.batch_size.setValue(2048)
        self.ubatch_size = QSpinBox()
        self.ubatch_size.setRange(64, 8192)
        self.ubatch_size.setSingleStep(64)
        self.ubatch_size.setValue(2048)
        batch_layout.addWidget(self.batch_size)
        batch_layout.addWidget(self.ubatch_size)
        perf_layout.addLayout(batch_layout)

        threads_layout = QHBoxLayout()
        threads_layout.addWidget(QLabel("Threads gen / batch (-t / -tb):"))
        self.threads = QSpinBox()
        self.threads.setRange(1, 64)
        self.threads.setValue(os.cpu_count() or 4)
        self.threads_batch = QSpinBox()
        self.threads_batch.setRange(0, 64)
        self.threads_batch.setSpecialValueText("same")
        self.threads_batch.setValue(0)
        threads_layout.addWidget(self.threads)
        threads_layout.addWidget(self.threads_batch)
        perf_layout.addLayout(threads_layout)

        kv_layout = QHBoxLayout()
        kv_layout.addWidget(QLabel("KV Cache K / V (-ctk / -ctv):"))
        self.cache_type_k = QComboBox()
        self.cache_type_v = QComboBox()
        for ct in ["f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "f32"]:
            self.cache_type_k.addItem(ct)
            self.cache_type_v.addItem(ct)
        self.cache_type_k.setCurrentText("f16")
        self.cache_type_v.setCurrentText("f16")
        kv_layout.addWidget(self.cache_type_k)
        kv_layout.addWidget(self.cache_type_v)
        perf_layout.addLayout(kv_layout)

        flags_layout = QHBoxLayout()
        self.flash_attn = QCheckBox("Flash Attention (-fa)")
        self.flash_attn.setChecked(True)
        self.fit_off = QCheckBox("Fit off (--fit off)")
        self.fit_off.setChecked(True)
        flags_layout.addWidget(self.flash_attn)
        flags_layout.addWidget(self.fit_off)
        perf_layout.addLayout(flags_layout)

        rea_layout = QHBoxLayout()
        rea_layout.addWidget(QLabel("Reasoning (-rea):"))
        self.reasoning_mode = QComboBox()
        for m in ("off", "auto", "on"):
            self.reasoning_mode.addItem(m)
        self.reasoning_mode.setCurrentText("off")
        rea_layout.addWidget(self.reasoning_mode)
        perf_layout.addLayout(rea_layout)

        port_slot_layout = QHBoxLayout()
        port_slot_layout.addWidget(QLabel("Port:"))
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(8080)
        port_slot_layout.addWidget(self.port)
        port_slot_layout.addSpacing(10)
        port_slot_layout.addWidget(QLabel("Slots (-np):"))
        self.parallel_slots = QSpinBox()
        self.parallel_slots.setRange(1, 16)
        self.parallel_slots.setValue(1)
        port_slot_layout.addWidget(self.parallel_slots)
        perf_layout.addLayout(port_slot_layout)

        chk_ram_layout = QHBoxLayout()
        chk_ram_layout.addWidget(QLabel("Ctx Checkpoints:"))
        self.ctx_checkpoints = QSpinBox()
        self.ctx_checkpoints.setRange(-1, 128)
        self.ctx_checkpoints.setSpecialValueText("default")
        self.ctx_checkpoints.setValue(-1)
        chk_ram_layout.addWidget(self.ctx_checkpoints)
        chk_ram_layout.addSpacing(10)
        chk_ram_layout.addWidget(QLabel("Cache RAM (MiB):"))
        self.cache_ram = QSpinBox()
        self.cache_ram.setRange(-2, 262144)
        self.cache_ram.setSpecialValueText("default")
        self.cache_ram.setValue(-2)
        chk_ram_layout.addWidget(self.cache_ram)
        perf_layout.addLayout(chk_ram_layout)
        left_layout.addWidget(perf_group)

        # 4. Спойлер: Сэмплинг и отладка
        adv_panel = CollapsiblePanel("⚙️ Сэмплинг, отладка и сервер")
        samp_layout = QHBoxLayout()
        samp_layout.addWidget(QLabel("Temperature:"))
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.7)
        self.temperature.setDecimals(2)
        samp_layout.addWidget(self.temperature)
        samp_layout.addSpacing(10)
        samp_layout.addWidget(QLabel("Repeat Penalty:"))
        self.repeat_penalty = QDoubleSpinBox()
        self.repeat_penalty.setRange(0.0, 2.0)
        self.repeat_penalty.setSingleStep(0.01)
        self.repeat_penalty.setValue(1.1)
        self.repeat_penalty.setDecimals(2)
        samp_layout.addWidget(self.repeat_penalty)
        adv_panel.add_layout(samp_layout)

        mem_dbg_layout = QHBoxLayout()
        self.use_mmap = QCheckBox("mmap")
        self.use_mmap.setChecked(True)
        self.use_mlock = QCheckBox("mlock")
        self.verbose = QCheckBox("verbose")
        self.log_timestamps = QCheckBox("log timestamps")
        mem_dbg_layout.addWidget(self.use_mmap)
        mem_dbg_layout.addWidget(self.use_mlock)
        mem_dbg_layout.addWidget(self.verbose)
        mem_dbg_layout.addWidget(self.log_timestamps)
        adv_panel.add_layout(mem_dbg_layout)

        srv_layout = QHBoxLayout()
        self.cont_batching = QCheckBox("cont batching")
        self.cont_batching.setChecked(True)
        self.cache_prompt = QCheckBox("cache prompt")
        self.cache_prompt.setChecked(True)
        self.context_shift = QCheckBox("context shift")
        self.no_webui = QCheckBox("no webui")
        srv_layout.addWidget(self.cont_batching)
        srv_layout.addWidget(self.cache_prompt)
        srv_layout.addWidget(self.context_shift)
        srv_layout.addWidget(self.no_webui)
        adv_panel.add_layout(srv_layout)

        adv_panel.add_widget(QLabel("Доп. параметры:"))
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText(
            "--top-p 0.9 --min-p 0.05 --rope-scaling yarn ..."
        )
        adv_panel.add_widget(self.extra_args)
        left_layout.addWidget(adv_panel)

        # 5. Спойлер: Интеграция
        int_panel = CollapsiblePanel("🔌 Интеграция (OpenCode / PI)")
        oc_layout = QHBoxLayout()
        oc_layout.addWidget(QLabel("OpenCode JSON:"))
        self.opencode_config_path = QLineEdit()
        self.opencode_config_path.setPlaceholderText("Путь к opencode.json")
        self.opencode_config_path.editingFinished.connect(self.save_settings)
        oc_btn = QPushButton("📂")
        oc_btn.clicked.connect(self.browse_opencode_config)
        oc_layout.addWidget(self.opencode_config_path)
        oc_layout.addWidget(oc_btn)
        int_panel.add_layout(oc_layout)

        pi_layout = QHBoxLayout()
        pi_layout.addWidget(QLabel("PI JSON:"))
        self.pi_config_path = QLineEdit()
        self.pi_config_path.setPlaceholderText("Путь к PI config.json")
        self.pi_config_path.editingFinished.connect(self.save_settings)
        pi_btn = QPushButton("📂")
        pi_btn.clicked.connect(self.browse_pi_config)
        pi_layout.addWidget(self.pi_config_path)
        pi_layout.addWidget(pi_btn)
        int_panel.add_layout(pi_layout)

        tgt_layout = QHBoxLayout()
        tgt_layout.addWidget(QLabel("Цель:"))
        self.integration_target = QComboBox()
        self.integration_target.addItem("OpenCode", "opencode")
        self.integration_target.addItem("PI", "pi")
        self.integration_target.currentIndexChanged.connect(
            lambda *_: self.check_integration_models(silent=True)
        )
        tgt_layout.addWidget(self.integration_target)
        self.integration_check_btn = QPushButton("Проверить")
        self.integration_check_btn.clicked.connect(self.check_integration_models)
        tgt_layout.addWidget(self.integration_check_btn)
        int_panel.add_layout(tgt_layout)

        self.integration_model_label = QLabel("Модель для добавления: не выбрана")
        self.integration_model_label.setWordWrap(True)
        int_panel.add_widget(self.integration_model_label)
        self.integration_models_list = QListWidget()
        self.integration_models_list.setMinimumHeight(80)
        int_panel.add_widget(self.integration_models_list)

        act_layout = QHBoxLayout()
        self.integration_add_btn = QPushButton("Добавить")
        self.integration_add_btn.clicked.connect(self.add_model_to_integration)
        self.integration_remove_btn = QPushButton("Удалить")
        self.integration_remove_btn.clicked.connect(self.remove_model_from_integration)
        act_layout.addWidget(self.integration_add_btn)
        act_layout.addWidget(self.integration_remove_btn)
        int_panel.add_layout(act_layout)

        self.integration_status = QLabel("Укажите путь к конфигу и нажмите Проверить")
        self.integration_status.setWordWrap(True)
        int_panel.add_widget(self.integration_status)
        left_layout.addWidget(int_panel)

        # 6. Спойлер: Бенчмарк
        bench_panel = CollapsiblePanel("🧪 Бенчмарк")
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
        bench_panel.add_layout(bp_layout)
        self.test_btn = QPushButton("🧪 Тестировать скорость")
        self.test_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 6px;"
        )
        self.test_btn.clicked.connect(self.run_benchmark)
        bench_panel.add_widget(self.test_btn)
        left_layout.addWidget(bench_panel)

        # 7. Preview CLI
        preview_group = QGroupBox("👁️ Preview CLI")
        preview_layout = QVBoxLayout(preview_group)
        self.cli_preview = QLineEdit()
        self.cli_preview.setReadOnly(True)
        self.cli_preview.setPlaceholderText("Команда будет отображаться здесь...")
        self.cli_preview.setStyleSheet(
            "background-color: #2a2a2a; color: #b5cea8; font-family: Consolas; padding: 4px;"
        )
        preview_layout.addWidget(self.cli_preview)
        left_layout.addWidget(preview_group)

        # 8. Кнопки управления
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Старт Server")
        self.start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        self.start_btn.clicked.connect(self.start_server)
        self.stop_btn = QPushButton("⏹ Стоп")
        self.stop_btn.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; padding: 8px;"
        )
        self.stop_btn.clicked.connect(self.stop_work)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        left_layout.addLayout(btn_layout)

        left_layout.addStretch()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(380)
        left_scroll.setMaximumWidth(520)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # === ПРАВАЯ ПАНЕЛЬ (ЛОГИ) ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Логи:"))
        self.autoscroll_logs = QCheckBox("Автоскролл")
        self.autoscroll_logs.setChecked(True)
        log_header.addWidget(self.autoscroll_logs)
        right_layout.addLayout(log_header)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setFont(QFont("Consolas", 9))
        self.logs.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        right_layout.addWidget(self.logs)

        clear_btn = QPushButton("🧹 Очистить логи")
        clear_btn.clicked.connect(self.logs.clear)
        right_layout.addWidget(clear_btn)

        # QSplitter для изменяемых пропорций
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([420, 730])
        main_layout.addWidget(splitter)

        self.setup_tooltips()

    def setup_tray(self):
        """Настройка системного трея."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setToolTip("LlamaServer GUI")

        # Меню трея
        self.tray_menu = QMenu()
        show_action = QAction("Показать", self)
        show_action.triggered.connect(self.showNormal)
        self.tray_menu.addAction(show_action)

        hide_action = QAction("Скрыть", self)
        hide_action.triggered.connect(self.hide)
        self.tray_menu.addAction(hide_action)

        self.tray_menu.addSeparator()

        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        self.tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def on_tray_activated(self, reason):
        """Обработка активации иконки трея."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.showNormal()
                self.raise_()
                self.activateWindow()

    def setup_tooltips(self):
        """Установка подсказок для виджетов."""
        tips = {
            self.exe_path: "Путь к llama-server.exe. Основной сервер llama.cpp.",
            self.bench_path: "Путь к llama-bench.exe. Используется только для теста скорости.",
            self.model_dir: "Корневая папка для рекурсивного поиска .gguf и mmproj.",
            self.scan_btn: "Сканирует папку моделей в фоне. Повторный клик отменяет обход.",
            self.model_combo: "Выбранная GGUF-модель. Можно выбрать из списка или ввести путь вручную.",
            self.auto_params: "Автоматически выставляет ctx, KV cache и batch по GGUF metadata.",
            self.use_mmproj: "Добавляет -mm с найденным projector-файлом для vision/multimodal моделей.",
            self.mmproj_offload: "Разрешает offload mmproj. Если выключить, добавляется --no-mmproj-offload.",
            self.gpu_layers: "Слои весов на GPU. Главный рычаг скорости и VRAM.",
            self.gpu_auto: "Для server: -ngl auto. Для bench: 99.",
            self.cpu_moe_layers: "CPU MoE layers (-ncmoe). Критично для MoE-моделей (например Qwen3-A3B).",
            self.ctx_size: "Размер контекста. Резко влияет на KV cache и VRAM.",
            self.batch_size: "Batch size (-b). Влияет на обработку промпта.",
            self.ubatch_size: "Micro-batch (-ub). Ключевой параметр для скорости prefill.",
            self.threads: "CPU-потоки генерации (-t).",
            self.threads_batch: "CPU-потоки обработки промпта (-tb). 0 = same as -t.",
            self.cache_type_k: "Тип KV cache Key. q8_0 экономит ~50% VRAM на длинных контекстах.",
            self.cache_type_v: "Тип KV cache Value.",
            self.flash_attn: "Flash Attention. Ускоряет и экономит память на современных GPU.",
            self.fit_off: "Отключает авто-подгонку контекста под VRAM (--fit off).",
            self.reasoning_mode: "Режим reasoning (-rea). Влияет на thinking-токены у MoE/Reasoning моделей.",
            self.port: "HTTP-порт llama-server.",
            self.parallel_slots: "Количество слотов (-np). Каждый слот увеличивает потребление KV cache.",
            self.ctx_checkpoints: "Ctx checkpoints. Снижает пиковое RAM при длинных контекстах.",
            self.cache_ram: "Явный лимит RAM под cache (MiB).",
            self.temperature: "Sampling temperature. Влияет на разнообразие, не на скорость.",
            self.repeat_penalty: "Штраф повторов. Обычно 1.05-1.20.",
            self.use_mmap: "Memory mapping модели. Обычно ускоряет загрузку.",
            self.use_mlock: "Удерживает модель в RAM. Требует достаточно памяти.",
            self.verbose: "Подробный лог llama.cpp.",
            self.log_timestamps: "Добавляет timestamps в лог.",
            self.cont_batching: "Continuous batching. Полезно для нескольких одновременных запросов.",
            self.cache_prompt: "Prompt cache. Ускоряет повторное использование префикса.",
            self.context_shift: "Context shift при заполнении контекста.",
            self.no_webui: "Отключает встроенный Web UI.",
            self.extra_args: "Дополнительные параметры. Разбираются через shlex.",
            self.bench_prompt: "Токены промпта для бенчмарка (-p).",
            self.bench_gen: "Генерируемые токены для бенчмарка (-n).",
            self.test_btn: "Запускает llama-bench с текущими параметрами.",
            self.start_btn: "Запускает llama-server.",
            self.stop_btn: "Останавливает сервер, бенчмарк или сканирование.",
            self.autoscroll_logs: "Автоматическая прокрутка логов вниз.",
            self.opencode_config_path: "Путь к opencode.json для интеграции.",
            self.pi_config_path: "Путь к config.json PI для интеграции.",
            self.integration_target: "Выбор целевого конфига: OpenCode или PI.",
            self.integration_check_btn: "Проверить модели в выбранном конфиге.",
            self.integration_add_btn: "Добавить текущую модель в конфиг.",
            self.integration_remove_btn: "Удалить модель из конфига.",
        }
        for widget, text in tips.items():
            if widget:
                widget.setToolTip(text)

    # === Методы управления процессами ===

    def update_cli_preview(self):
        """Обновление preview командной строки."""
        try:
            args = self.build_args(for_benchmark=False)
            if args:
                exe = self.exe_path.text() or "llama-server.exe"
                cmd = f"{exe} {' '.join(args)}"
                self.cli_preview.setText(cmd)
            else:
                self.cli_preview.setText("")
        except Exception:
            self.cli_preview.setText("")

    def start_server(self):
        """Запуск llama-server."""
        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(
                self, "Benchmark запущен", "Остановите benchmark перед запуском сервера"
            )
            return
        exe = self.exe_path.text()
        if not exe or not os.path.exists(exe):
            QMessageBox.critical(self, "Ошибка", "Укажите путь к llama-server.exe")
            return
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.log("⚠️ Сервер уже запущен")
            return
        args = self.build_args(for_benchmark=False)
        if not args:
            return
        self.log(f"▶ Запуск сервера: {exe}\n   Аргументы: {' '.join(args)}")
        self.server_stop_requested = False
        self.process.start(exe, args)
        self.start_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # Обновляем tray tooltip
        if self.tray_icon:
            self.tray_icon.setToolTip(
                f"LlamaServer GUI - Running on port {self.port.value()}"
            )

    def stop_work(self):
        """Остановка текущей работы."""
        stopped = False
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.stop_server()
            stopped = True
        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            self.stop_benchmark()
            stopped = True
        if self.scanner and self.scanner.isRunning():
            self.cancel_scan()
            stopped = True
        if not stopped:
            self.update_action_buttons()

    def stop_server(self):
        """Остановка сервера с graceful shutdown."""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            if self.server_stop_requested:
                return
            self.server_stop_requested = True
            self.log("⏹ Остановка сервера...")
            self.process.terminate()
            QTimer.singleShot(KILL_TIMEOUT_SERVER, self.kill_server_if_running)

    def kill_server_if_running(self):
        """Принудительная остановка сервера."""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.log("⚠️ Сервер не завершился штатно, принудительная остановка")
            self.process.kill()

    def run_benchmark(self):
        """Запуск бенчмарка."""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(
                self, "Сервер запущен", "Остановите сервер перед запуском benchmark"
            )
            return
        bench_exe = self.bench_path.text()
        if not bench_exe or not os.path.exists(bench_exe):
            self.auto_detect_bench()
            bench_exe = self.bench_path.text()
            if not bench_exe or not os.path.exists(bench_exe):
                QMessageBox.critical(self, "Ошибка", "Укажите путь к llama-bench.exe")
                return
        args = self.build_args(for_benchmark=True)
        if not args:
            return
        self.log(
            f"🧪 Запуск бенчмарка: {os.path.basename(bench_exe)}\n"
            f"   Модель: {self.model_combo.currentText()}\n"
            f"   Параметры: {' '.join(args)}"
        )
        if self.bench_process.state() == QProcess.ProcessState.Running:
            self.stop_benchmark()
            return
        self.bench_stop_requested = False
        self.test_btn.setEnabled(False)
        self.test_btn.setText("⏳ Тестирование...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.bench_process.start(bench_exe, args)

    def stop_benchmark(self):
        """Остановка бенчмарка с graceful shutdown."""
        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            self.bench_stop_requested = True
            self.log("⏹ Остановка benchmark...")
            self.bench_process.terminate()
            QTimer.singleShot(KILL_TIMEOUT_BENCHMARK, self.kill_benchmark_if_running)

    def kill_benchmark_if_running(self):
        """Принудительная остановка бенчмарка."""
        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            self.log("⚠️ Benchmark не завершился штатно, принудительная остановка")
            self.bench_process.kill()

    # === Обработчики вывода процессов ===

    def handle_stdout(self):
        """Обработка stdout сервера."""
        data = (
            self.process.readAllStandardOutput().data().decode("utf-8", errors="ignore")
        )
        self.log(data, "info")

    def handle_stderr(self):
        """Обработка stderr сервера."""
        data = (
            self.process.readAllStandardError().data().decode("utf-8", errors="ignore")
        )
        self.log(data, "error")

    def handle_state(self, state):
        """Обработка изменения состояния процесса."""
        if state == QProcess.ProcessState.NotRunning:
            self.start_btn.setEnabled(True)
            self.test_btn.setEnabled(
                self.bench_process.state() == QProcess.ProcessState.NotRunning
            )
            self.update_action_buttons()
            if self.server_stop_requested:
                self.log("⏹ Сервер остановлен")
            else:
                self.log(f"⏹ Сервер остановлен (код: {self.process.exitCode()})")
            self.server_stop_requested = False
            # Обновляем tray tooltip
            if self.tray_icon:
                self.tray_icon.setToolTip("LlamaServer GUI - Stopped")

    def handle_bench_stdout(self):
        """Обработка stdout бенчмарка."""
        data = (
            self.bench_process.readAllStandardOutput()
            .data()
            .decode("utf-8", errors="ignore")
        )
        self.log(data, "bench")

    def handle_bench_stderr(self):
        """Обработка stderr бенчмарка."""
        data = (
            self.bench_process.readAllStandardError()
            .data()
            .decode("utf-8", errors="ignore")
        )
        self.log(data, "error")

    def handle_bench_finished(self, exit_code):
        """Обработка завершения бенчмарка."""
        self.test_btn.setEnabled(True)
        self.test_btn.setText("🧪 Тестировать скорость")
        self.start_btn.setEnabled(
            self.process.state() == QProcess.ProcessState.NotRunning
        )
        self.update_action_buttons()
        if self.bench_stop_requested:
            self.log("⏹ Тестирование остановлено")
        elif exit_code == 0:
            self.log("✅ Тестирование завершено успешно")
        else:
            self.log(f"❌ Ошибка тестирования (код: {exit_code})", "error")
        self.bench_stop_requested = False

    # === Метрики и логирование ===

    def log(self, text: str, level: str = "info") -> None:
        """Добавление записи в лог через буфер (batch-логирование).

        Args:
            text: Текст для добавления.
            level: Уровень логирования (info, error, bench).
        """
        self.log_buffer.append((text, level))

    def flush_log_buffer(self) -> None:
        """Сброс буфера логов в UI (вызывается таймером каждые 100ms)."""
        if not self.log_buffer:
            return

        cursor = self.logs.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        for text, level in self.log_buffer:
            fmt = QTextCharFormat()
            if level == "error":
                fmt.setForeground(QColor("#f48771"))
            elif level == "bench":
                fmt.setForeground(QColor("#4ec9b0"))
            elif "error" in text.lower() or "failed" in text.lower():
                fmt.setForeground(QColor("#f48771"))
            elif "loading model" in text.lower():
                fmt.setForeground(QColor("#4ec9b0"))
            elif "server started" in text.lower():
                fmt.setForeground(QColor("#b5cea8"))
            cursor.insertText(text, fmt)

        self.log_buffer.clear()

        # Ограничение размера логов
        if self.logs.document().blockCount() > MAX_LOG_LINES:
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.movePosition(
                QTextCursor.MoveOperation.Down,
                QTextCursor.MoveMode.KeepAnchor,
                self.logs.document().blockCount() - MAX_LOG_LINES,
            )
            cursor.removeSelectedText()

        if self.autoscroll_logs.isChecked():
            self.logs.setTextCursor(cursor)
            self.logs.ensureCursorVisible()
            self.logs.verticalScrollBar().setValue(
                self.logs.verticalScrollBar().maximum()
            )

    # === Методы управления UI ===

    def update_action_buttons(self):
        """Обновление состояния кнопок."""
        srv = self.process.state() != QProcess.ProcessState.NotRunning
        bnch = self.bench_process.state() != QProcess.ProcessState.NotRunning
        scan = self.scanner is not None and self.scanner.isRunning()
        upd = self.updater is not None and self.updater.isRunning()
        busy = srv or bnch or scan
        self.stop_btn.setEnabled(busy)
        self.update_llama_btn.setEnabled(not srv and not bnch and not upd)
        if upd:
            self.start_btn.setEnabled(False)
            self.test_btn.setEnabled(False)
        if not srv and not bnch and not upd:
            self.start_btn.setEnabled(True)
            self.test_btn.setEnabled(True)

    # === Методы сканирования моделей ===

    def auto_scan_models(self):
        """Автоматическое сканирование моделей при запуске."""
        if self.models:
            self.scan_status.setText(
                f"Кэш моделей: {len(self.models)}. Фоновая проверка..."
            )
        base_path = self.model_dir.text()
        if base_path and os.path.exists(base_path):
            self.scan_models(silent=True)

    def scan_models(self, silent=False):
        """Сканирование моделей."""
        base_path = self.model_dir.text()
        if not base_path or not os.path.exists(base_path):
            if not silent:
                QMessageBox.warning(
                    self, "Ошибка", "Укажите существующую базовую папку"
                )
            return
        if self.scanner and self.scanner.isRunning():
            if not silent:
                self.cancel_scan()
            return
        self.scan_cancel_requested = False
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("⏹ Отменить")
        self.scan_progress.setVisible(True)
        self.scan_status.setText("Сканирование GGUF...")
        self.update_action_buttons()
        if not silent:
            self.log("🔍 Сканирование моделей...")
        self.scanner = ModelScanner(base_path)
        self.scanner.progress.connect(self.on_scan_progress)
        self.scanner.models_found.connect(self.on_models_found)
        self.scanner.finished.connect(self.on_scan_finished)
        self.scanner.start()

    def cancel_scan(self):
        """Отмена сканирования."""
        if self.scanner and self.scanner.isRunning():
            self.scan_cancel_requested = True
            self.scan_status.setText("Отмена сканирования...")
            self.log("⏹ Отмена сканирования моделей...")
            self.scanner.requestInterruption()
            self.scan_btn.setEnabled(False)

    def on_scan_progress(self, text):
        """Обработка прогресса сканирования."""
        self.scan_status.setText(text)

    def on_models_found(self, models):
        """Обработка найденных моделей."""
        if self.scan_cancel_requested:
            return
        current_path = self.model_combo.currentData() or self.settings.get(
            "last_model_path", ""
        )
        self.models = models
        self.models_by_path = {i["path"]: i for i in models}
        self.model_combo.clear()
        for i in models:
            self.model_combo.addItem(i["display"], i["path"])
        if current_path:
            idx = self.model_combo.findData(current_path)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        elif models:
            self.model_combo.setCurrentIndex(0)
        self.save_settings()
        self.log(f"✅ Найдено моделей: {len(models)}")
        self.scan_status.setText(f"Найдено моделей: {len(models)}")

    def on_scan_finished(self):
        """Обработка завершения сканирования."""
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("🔍 Сканировать")
        self.scan_progress.setVisible(False)
        if self.scan_cancel_requested:
            self.scan_status.setText("Сканирование отменено")
            self.scan_cancel_requested = False
        self.update_action_buttons()

    def on_model_selected(self, *_):
        """Обработка выбора модели."""
        model_path = self.model_combo.currentData()
        if not model_path:
            # Проверяем, введен ли путь вручную
            text = self.model_combo.currentText().strip()
            if text and os.path.exists(text) and text.lower().endswith(".gguf"):
                model_path = text
                self.model_combo.setItemData(
                    self.model_combo.currentIndex(), model_path
                )
            else:
                self.model_info.setText("Выберите модель")
                return
        info = self.models_by_path.get(model_path)
        if not info:
            info = extract_model_info(model_path)
            info["display"] = self.model_combo.currentText()
            self.models_by_path[model_path] = info
        max_ctx = info.get("context_length") or "не указан"
        quant = info.get("quant") or "не определено"
        arch = info.get("architecture") or "не определено"
        size_gib = info.get("size_gib", 0)
        recommended_ctx = info.get("recommended_ctx", 4096)
        mmproj = info.get("mmproj_path") or "не найден"
        layers = info.get("block_count") if info.get("block_count") is not None else "не определено"
        error = (
            f"\nMetadata: {info['metadata_error']}"
            if info.get("metadata_error")
            else ""
        )
        self.model_info.setText(
            f"Архитектура: {arch}; квант: {quant}; размер: {size_gib} GiB; "
            f"слои: {layers}; ctx модели: {max_ctx}; рекомендовано: {recommended_ctx}; mmproj: {mmproj}{error}"
        )
        self.settings["last_model_path"] = model_path
        self.update_integration_model_label()
        if self.auto_params.isChecked() and not self.loading_profile:
            self.apply_recommended_params(info)
        # Обновляем preview CLI
        self.update_cli_preview()

    def apply_recommended_params(self, info):
        """Применение рекомендуемых параметров."""
        rec = info.get("recommended_ctx")
        if rec:
            self.ctx_size.setValue(rec)
        quant = (info.get("quant") or "").upper()
        if (
            quant.startswith(("Q2", "Q3", "IQ1", "IQ2", "IQ3"))
            or info.get("recommended_ctx", 0) >= 16384
        ):
            self.cache_type_k.setCurrentText("q8_0")
            self.cache_type_v.setCurrentText("q8_0")
        else:
            self.cache_type_k.setCurrentText("f16")
            self.cache_type_v.setCurrentText("f16")
        if self.gpu_auto.isChecked():
            self.gpu_layers.setDisabled(True)
        self.batch_size.setValue(2048)
        self.ubatch_size.setValue(2048)

    # === Методы сборки аргументов ===

    def build_args(self, for_benchmark=False):
        """Сборка аргументов командной строки.

        Args:
            for_benchmark: Если True, собирает аргументы для бенчмарка.

        Returns:
            Список аргументов или None при ошибке.
        """
        args = []
        model_path = self.model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, "Ошибка", "Выберите модель")
            return None
        if for_benchmark:
            args.extend(
                [
                    "-m",
                    model_path,
                    "-p",
                    str(self.bench_prompt.value()),
                    "-n",
                    str(self.bench_gen.value()),
                ]
            )
            args.extend(["-ngl", self.gpu_layers_arg(for_benchmark=True)])
            if self.flash_attn.isChecked():
                args.extend(["-fa", "1"])
            args.extend(
                [
                    "-ctk",
                    self.cache_type_k.currentText(),
                    "-ctv",
                    self.cache_type_v.currentText(),
                ]
            )
            args.extend(
                [
                    "-b",
                    str(self.batch_size.value()),
                    "-ub",
                    str(min(self.ubatch_size.value(), self.batch_size.value())),
                ]
            )
        else:
            args.extend(
                [
                    "-m",
                    model_path,
                    "--port",
                    str(self.port.value()),
                    "-ngl",
                    self.gpu_layers_arg(),
                ]
            )
            args.extend(
                ["-c", str(self.ctx_size.value()), "-t", str(self.threads.value())]
            )
            if self.threads_batch.value() > 0:
                args.extend(["-tb", str(self.threads_batch.value())])
            args.extend(
                [
                    "-b",
                    str(self.batch_size.value()),
                    "-ub",
                    str(min(self.ubatch_size.value(), self.batch_size.value())),
                ]
            )
            args.extend(
                [
                    "-ctk",
                    self.cache_type_k.currentText(),
                    "-ctv",
                    self.cache_type_v.currentText(),
                ]
            )
            args.extend(["-np", str(self.parallel_slots.value())])
            if self.cpu_moe_layers.value() > 0:
                args.extend(["-ncmoe", str(self.cpu_moe_layers.value())])
            if self.fit_off.isChecked():
                args.extend(["--fit", "off"])
            rea = self.reasoning_mode.currentText()
            if rea != "auto":
                args.extend(["-rea", rea])
            if self.ctx_checkpoints.value() >= 0:
                args.extend(["--ctx-checkpoints", str(self.ctx_checkpoints.value())])
            if self.cache_ram.value() >= -1:
                args.extend(["--cache-ram", str(self.cache_ram.value())])
            args.extend(
                [
                    "--temp",
                    str(self.temperature.value()),
                    "--repeat-penalty",
                    str(self.repeat_penalty.value()),
                ]
            )
            if self.flash_attn.isChecked():
                args.extend(["--flash-attn", "on"])
            model_info = self.models_by_path.get(model_path) or {}
            mmproj_path = model_info.get("mmproj_path", "")
            if self.use_mmproj.isChecked():
                if mmproj_path:
                    args.extend(["-mm", mmproj_path])
                if not self.mmproj_offload.isChecked():
                    args.append("--no-mmproj-offload")
            else:
                args.append("--no-mmproj")
            if self.use_mmap.isChecked():
                args.append("--mmap")
            else:
                args.append("--no-mmap")
            if self.use_mlock.isChecked():
                args.append("--mlock")
            if self.verbose.isChecked():
                args.append("--verbose")
            if self.log_timestamps.isChecked():
                args.append("--log-timestamps")
            if not self.cont_batching.isChecked():
                args.append("--no-cont-batching")
            if not self.cache_prompt.isChecked():
                args.append("--no-cache-prompt")
            if self.context_shift.isChecked():
                args.append("--context-shift")
            if self.no_webui.isChecked():
                args.append("--no-webui")
            if self.extra_args.text():
                try:
                    extra = shlex.split(self.extra_args.text())
                    invalid = self._validate_extra_args(extra)
                    if invalid:
                        QMessageBox.warning(
                            self,
                            "Ошибка",
                            f"Недопустимые параметры: {', '.join(invalid)}\n"
                            f"Разрешены только флаги из списка: {', '.join(sorted(LLAMA_ALLOWED_FLAGS))}",
                        )
                        return None
                    args.extend(extra)
                except ValueError as exc:
                    QMessageBox.warning(
                        self, "Ошибка", f"Не удалось разобрать доп. параметры: {exc}"
                    )
                    return None
        return args

    def _validate_extra_args(self, args: List[str]) -> List[str]:
        """Валидация дополнительных аргументов.

        Проверяет:
        1. Флаг в whitelist
        2. Значения для флагов, принимающих пути
        3. Значения для флагов, принимающих IP/хосты

        Args:
            args: Список аргументов.

        Returns:
            Список недопустимых аргументов с описанием ошибки.
        """
        invalid = []
        i = 0
        while i < len(args):
            arg = args[i]
            if not arg.startswith("-"):
                i += 1
                continue

            # Проверяем флаг
            base_arg = arg.split("=")[0] if "=" in arg else arg
            if base_arg not in LLAMA_ALLOWED_FLAGS:
                invalid.append(f"{arg} (неизвестный флаг)")
                i += 1
                continue

            # Проверяем значение для path-флагов
            path_flags = {
                "--grammar-file",
                "--api-key-file",
                "--lora",
                "--lora-scaled",
                "--mmproj",
                "--chat-template-file",
            }
            if base_arg in path_flags:
                if "=" in arg:
                    value = arg.split("=", 1)[1]
                elif i + 1 < len(args):
                    value = args[i + 1]
                    i += 1
                else:
                    invalid.append(f"{arg} (требуется значение)")
                    i += 1
                    continue

                try:
                    # Проверяем, что путь не выходит за пределы моделей
                    model_dir = Path(self.model_dir.text() or ".").resolve()
                    validate_path(value, base_dir=model_dir)
                except ValueError as e:
                    invalid.append(f"{arg} {value} (недопустимый путь: {e})")

            # Проверяем значение для host
            if base_arg == "--host":
                if "=" in arg:
                    value = arg.split("=", 1)[1]
                elif i + 1 < len(args):
                    value = args[i + 1]
                    i += 1
                else:
                    invalid.append(f"{arg} (требуется значение)")
                    i += 1
                    continue

                if value in ("0.0.0.0", "::", ""):
                    invalid.append(
                        f"{arg} {value} (запрещено: открывает сервер всем интерфейсам)"
                    )

            i += 1

        return invalid

    def gpu_layers_arg(self, for_benchmark=False):
        """Получение аргумента GPU layers."""
        if not self.gpu_auto.isChecked():
            return str(self.gpu_layers.value())
        return "99" if for_benchmark else "auto"

    # === Методы обновления llama.cpp ===

    def update_llamacpp(self):
        """Обновление llama.cpp с подтверждением и бэкапом."""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(
                self,
                "llama.cpp updater",
                "Stop llama-server before updating llama.cpp.",
            )
            return
        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(
                self, "llama.cpp updater", "Stop benchmark before updating llama.cpp."
            )
            return
        if self.updater and self.updater.isRunning():
            return

        reply = QMessageBox.question(
            self,
            "Обновление llama.cpp",
            "Будет скачана и установлена последняя версия llama.cpp.\n"
            "Текущие бинарники будут сохранены в папке backup.\n"
            "Продолжить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        exe = self.exe_path.text().strip()
        if not exe or not os.path.exists(exe):
            QMessageBox.critical(
                self, "llama.cpp updater", "Select an existing llama-server.exe first."
            )
            return

        self.update_progress.setValue(0)
        self.update_progress.setVisible(True)
        self.update_status.setText("Checking release...")
        self.log("llama.cpp update: checking latest release\n")
        self.updater = LlamaCppUpdater(exe)
        self.updater.progress.connect(self.on_update_progress)
        self.updater.percent.connect(self.update_progress.setValue)
        self.updater.completed.connect(self.on_update_completed)
        self.updater.finished.connect(self.on_update_thread_finished)
        self.updater.start()
        self.update_action_buttons()

    def on_update_progress(self, text):
        """Обработка прогресса обновления."""
        self.update_status.setText(text)
        self.log(f"llama.cpp update: {text}\n")

    def on_update_completed(self, changed, message):
        """Обработка завершения обновления."""
        self.update_status.setText(message)
        self.log(
            f"llama.cpp update: {message}\n",
            "error" if "failed" in message.lower() else "info",
        )
        if changed:
            self.auto_detect_bench()
            self.save_settings()

    def on_update_thread_finished(self):
        """Обработка завершения потока обновления."""
        self.update_progress.setVisible(False)
        self.update_action_buttons()
        self.updater = None

    def _check_updater_started(self):
        """Проверка, что updater запустился."""
        if self.updater:
            is_running = self.updater.isRunning()
            if not is_running:
                self.log("⚠️ Updater failed to start or finished too quickly", "error")
                self.update_status.setText("Failed to start updater")
                self.update_progress.setVisible(False)
                self.update_action_buttons()

    def auto_detect_bench(self):
        """Автоопределение пути к бенчмарку."""
        server_path = self.exe_path.text()
        if server_path and os.path.exists(server_path):
            dir_path = os.path.dirname(server_path)
            bench_path = os.path.join(
                dir_path, os.path.basename(server_path).replace("server", "bench")
            )
            if os.path.exists(bench_path):
                self.bench_path.setText(bench_path)

    # === Методы работы с файлами ===

    def browse_exe(self):
        """Выбор пути к llama-server.exe."""
        f, _ = QFileDialog.getOpenFileName(
            self, "Выберите llama-server", "", "Executable (*.exe)"
        )
        if f:
            self.exe_path.setText(f)
            self.save_settings()

    def browse_bench(self):
        """Выбор пути к llama-bench.exe."""
        f, _ = QFileDialog.getOpenFileName(
            self, "Выберите llama-bench", "", "Executable (*.exe)"
        )
        if f:
            self.bench_path.setText(f)
            self.save_settings()

    def browse_model_dir(self):
        """Выбор папки с моделями."""
        d = QFileDialog.getExistingDirectory(self, "Выберите папку с моделями")
        if d:
            self.model_dir.setText(d)
            self.save_settings()
            self.scan_models()

    def browse_opencode_config(self):
        """Выбор пути к opencode.json."""
        f, _ = QFileDialog.getOpenFileName(
            self, "Выберите opencode.json", "", "JSON (*.json);;All files (*.*)"
        )
        if f:
            self.opencode_config_path.setText(f)
            self.save_settings()
            self.check_integration_models(silent=True)

    def browse_pi_config(self):
        """Выбор пути к PI config.json."""
        f, _ = QFileDialog.getOpenFileName(
            self, "Выберите PI config.json", "", "JSON (*.json);;All files (*.*)"
        )
        if f:
            self.pi_config_path.setText(f)
            self.save_settings()
            self.check_integration_models(silent=True)

    # === Методы интеграции ===

    def current_config_target(self):
        """Получение текущей цели интеграции."""
        return self.integration_target.currentData() or "opencode"

    def current_config_path(self):
        """Получение пути к текущему конфигу."""
        return (
            self.pi_config_path.text().strip()
            if self.current_config_target() == "pi"
            else self.opencode_config_path.text().strip()
        )

    def current_base_url(self):
        """Получение базового URL."""
        return f"http://127.0.0.1:{self.port.value()}/v1"

    def current_model_id(self):
        """Получение ID текущей модели."""
        model_path = self.model_combo.currentData()
        if model_path:
            return Path(model_path).stem
        text = self.model_combo.currentText().strip()
        if text.lower().endswith(".gguf") and os.path.exists(text):
            return Path(text).stem
        return text

    def update_integration_model_label(self):
        """Обновление метки модели для интеграции."""
        mid = self.current_model_id()
        self.integration_model_label.setText(
            f"Модель для добавления: {mid}"
            if mid
            else "Модель для добавления: не выбрана"
        )

    def read_integration_config(self):
        """Чтение конфигурации интеграции."""
        path = self.current_config_path()
        if not path:
            raise ValueError("Укажите путь к JSON-конфигу")
        return path, load_or_create_json(path)

    def integration_model_ids(self, data):
        """Получение списка моделей из конфига."""
        return get_model_ids(data, self.current_config_target())

    def check_integration_models(self, silent=False):
        """Проверка моделей в конфиге интеграции."""
        self.update_integration_model_label()
        self.integration_models_list.clear()
        try:
            path, data = self.read_integration_config()
            model_ids = self.integration_model_ids(data)
        except Exception as exc:
            self.integration_status.setText(f"Не удалось прочитать конфиг: {exc}")
            if not silent:
                QMessageBox.warning(self, "Конфиг моделей", str(exc))
            return
        for mid in model_ids:
            self.integration_models_list.addItem(mid)
        current_id = self.current_model_id()
        target_name = self.integration_target.currentText()
        exists_text = (
            "есть в конфиге"
            if current_id and current_id in model_ids
            else "нет в конфиге"
        )
        self.integration_status.setText(
            f"{target_name}: {len(model_ids)} моделей. "
            f"Выбранная: {current_id or 'не выбрана'} ({exists_text})."
        )
        self.log(f"{target_name}: проверено моделей: {len(model_ids)}")

    def add_model_to_integration(self):
        """Добавление модели в конфиг интеграции."""
        model_id = self.current_model_id()
        if not model_id:
            QMessageBox.warning(self, "Конфиг моделей", "Выберите модель GGUF")
            return
        try:
            path, data = self.read_integration_config()
            if self.current_config_target() == "pi":
                _, models = ensure_pi_llamacpp_provider(data, self.current_base_url())
                existing = {
                    str(m.get("id") or m.get("name")): m
                    for m in models
                    if isinstance(m, dict) and (m.get("id") or m.get("name"))
                }
                if model_id not in existing:
                    ctx = int(self.ctx_size.value())
                    models.append(
                        {
                            "contextWindow": ctx,
                            "id": model_id,
                            "input": ["text", "image"],
                            "maxTokens": min(16000, ctx),
                            "name": model_id,
                        }
                    )
            else:
                _, models = ensure_opencode_llamacpp_provider(
                    data, self.current_base_url()
                )
                models.setdefault(model_id, {"name": model_id})
            write_json_file_safely(path, data)
            self.save_settings()
            self.check_integration_models(silent=True)
            self.integration_status.setText(f"Добавлено/обновлено: {model_id}")
            self.log(f"Конфиг моделей: добавлено/обновлено {model_id}")
        except Exception as exc:
            self.integration_status.setText(f"Не удалось добавить модель: {exc}")
            QMessageBox.critical(self, "Конфиг моделей", str(exc))

    def remove_model_from_integration(self):
        """Удаление модели из конфига интеграции."""
        sel = self.integration_models_list.currentItem()
        model_id = sel.text() if sel else self.current_model_id()
        if not model_id:
            QMessageBox.warning(
                self, "Конфиг моделей", "Выберите модель в списке или GGUF-модель"
            )
            return
        try:
            path, data = self.read_integration_config()
            removed = False
            if self.current_config_target() == "pi":
                providers = data.get("providers") or data.get("provider") or data
                provider = (
                    providers.get(LLAMACPP_PROVIDER_ID, {})
                    if isinstance(providers, dict)
                    else {}
                )
                models = (
                    provider.get("models", []) if isinstance(provider, dict) else []
                )
                if isinstance(models, list):
                    before = len(models)
                    models[:] = [
                        m
                        for m in models
                        if not (
                            isinstance(m, dict)
                            and str(m.get("id") or m.get("name")) == model_id
                        )
                    ]
                    removed = len(models) != before
            else:
                providers = data.get("provider") or data.get("providers") or data
                provider = (
                    providers.get(LLAMACPP_PROVIDER_ID, {})
                    if isinstance(providers, dict)
                    else {}
                )
                models = (
                    provider.get("models", {}) if isinstance(provider, dict) else {}
                )
                if isinstance(models, dict):
                    removed = models.pop(model_id, None) is not None
            if removed:
                write_json_file_safely(path, data)
            self.check_integration_models(silent=True)
            status = "Удалено" if removed else "Модель не найдена"
            self.integration_status.setText(f"{status}: {model_id}")
            self.log(f"Конфиг моделей: {status.lower()} {model_id}")
        except Exception as exc:
            self.integration_status.setText(f"Не удалось удалить модель: {exc}")
            QMessageBox.critical(self, "Конфиг моделей", str(exc))

    # === Методы работы с профилями ===

    def save_profile(self):
        """Сохранение текущего профиля."""
        name = self.profile_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Ошибка", "Введите имя профиля")
            return
        self.profiles[name] = {
            "model": self.model_combo.currentText(),
            "model_path": self.model_combo.currentData(),
            "temperature": self.temperature.value(),
            "repeat_penalty": self.repeat_penalty.value(),
            "gpu_auto": self.gpu_auto.isChecked(),
            "gpu_layers": self.gpu_layers.value(),
            "ctx_size": self.ctx_size.value(),
            "threads": self.threads.value(),
            "port": self.port.value(),
            "flash_attn": self.flash_attn.isChecked(),
            "use_mmap": self.use_mmap.isChecked(),
            "use_mlock": self.use_mlock.isChecked(),
            "verbose": self.verbose.isChecked(),
            "log_timestamps": self.log_timestamps.isChecked(),
            "cache_type_k": self.cache_type_k.currentText(),
            "cache_type_v": self.cache_type_v.currentText(),
            "batch_size": self.batch_size.value(),
            "ubatch_size": self.ubatch_size.value(),
            "parallel_slots": self.parallel_slots.value(),
            "cont_batching": self.cont_batching.isChecked(),
            "cache_prompt": self.cache_prompt.isChecked(),
            "context_shift": self.context_shift.isChecked(),
            "no_webui": self.no_webui.isChecked(),
            "extra_args": self.extra_args.text(),
            "bench_prompt": self.bench_prompt.value(),
            "bench_gen": self.bench_gen.value(),
        }
        self.save_profiles()
        self.refresh_profile_list()
        self.log(f"💾 Профиль сохранен: {name}")

    def load_profile(self, item):
        """Загрузка профиля."""
        name = item.text()
        if name not in self.profiles:
            return
        p = self.profiles[name]
        self.profile_name.setText(name)
        self.loading_profile = True
        idx = self.model_combo.findData(p.get("model_path", ""))
        if idx < 0:
            idx = self.model_combo.findText(p.get("model", ""))
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.temperature.setValue(p.get("temperature", 0.7))
        self.repeat_penalty.setValue(p.get("repeat_penalty", 1.1))
        self.use_mmproj.setChecked(p.get("use_mmproj", True))
        self.mmproj_offload.setChecked(p.get("mmproj_offload", True))
        self.gpu_auto.setChecked(p.get("gpu_auto", True))
        self.gpu_layers.setValue(p.get("gpu_layers", 33))
        self.ctx_size.setValue(p.get("ctx_size", 4096))
        self.threads.setValue(p.get("threads", 4))
        self.port.setValue(p.get("port", 8080))
        self.flash_attn.setChecked(p.get("flash_attn", True))
        self.use_mmap.setChecked(p.get("use_mmap", True))
        self.use_mlock.setChecked(p.get("use_mlock", False))
        self.verbose.setChecked(p.get("verbose", False))
        self.log_timestamps.setChecked(p.get("log_timestamps", False))
        self.cache_type_k.setCurrentText(p.get("cache_type_k", "f16"))
        self.cache_type_v.setCurrentText(p.get("cache_type_v", "f16"))
        self.batch_size.setValue(p.get("batch_size", 2048))
        self.ubatch_size.setValue(p.get("ubatch_size", 512))
        self.parallel_slots.setValue(p.get("parallel_slots", 1))
        self.cont_batching.setChecked(p.get("cont_batching", True))
        self.cache_prompt.setChecked(p.get("cache_prompt", True))
        self.context_shift.setChecked(p.get("context_shift", False))
        self.no_webui.setChecked(p.get("no_webui", False))
        self.extra_args.setText(p.get("extra_args", ""))
        self.bench_prompt.setValue(p.get("bench_prompt", 128))
        self.bench_gen.setValue(p.get("bench_gen", 256))
        self.loading_profile = False
        self.on_model_selected()
        self.log(f"📂 Загружен профиль: {name}")

    def delete_profile(self):
        """Удаление профиля."""
        name = self.profile_name.text().strip()
        if name in self.profiles:
            del self.profiles[name]
            self.save_profiles()
            self.refresh_profile_list()
            self.profile_name.clear()
            self.log(f"🗑 Профиль удален: {name}")

    def refresh_profile_list(self):
        """Обновление списка профилей."""
        self.profile_list.clear()
        for name in self.profiles.keys():
            self.profile_list.addItem(name)

    def save_profiles(self):
        """Сохранение профилей в файл."""
        try:
            write_json_file_safely(self.profiles_file, self.profiles)
        except Exception as e:
            self.log(f"Ошибка сохранения профилей: {e}", "error")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить профиль: {e}")

    # === Методы работы с настройками ===

    def save_settings(self) -> None:
        """Сохранение настроек."""
        self.settings = {
            "exe": self.exe_path.text(),
            "bench": self.bench_path.text(),
            "model_dir": self.model_dir.text(),
            "opencode_config": self.opencode_config_path.text(),
            "pi_config": self.pi_config_path.text(),
            "integration_target": self.current_config_target(),
            "bench_prompt": self.bench_prompt.value(),
            "bench_gen": self.bench_gen.value(),
            "auto_params": self.auto_params.isChecked(),
            "use_mmproj": self.use_mmproj.isChecked(),
            "mmproj_offload": self.mmproj_offload.isChecked(),
            "last_model_path": self.model_combo.currentData()
            or self.settings.get("last_model_path", ""),
            "model_cache": [],
            "temperature": self.temperature.value(),
            "repeat_penalty": self.repeat_penalty.value(),
            "gpu_auto": self.gpu_auto.isChecked(),
            "gpu_layers": self.gpu_layers.value(),
            "cpu_moe_layers": self.cpu_moe_layers.value(),
            "ctx_size": self.ctx_size.value(),
            "threads": self.threads.value(),
            "threads_batch": self.threads_batch.value(),
            "port": self.port.value(),
            "flash_attn": self.flash_attn.isChecked(),
            "fit_off": self.fit_off.isChecked(),
            "reasoning_mode": self.reasoning_mode.currentText(),
            "use_mmap": self.use_mmap.isChecked(),
            "use_mlock": self.use_mlock.isChecked(),
            "verbose": self.verbose.isChecked(),
            "log_timestamps": self.log_timestamps.isChecked(),
            "cache_type_k": self.cache_type_k.currentText(),
            "cache_type_v": self.cache_type_v.currentText(),
            "batch_size": self.batch_size.value(),
            "ubatch_size": self.ubatch_size.value(),
            "parallel_slots": self.parallel_slots.value(),
            "ctx_checkpoints": self.ctx_checkpoints.value(),
            "cache_ram": self.cache_ram.value(),
            "cont_batching": self.cont_batching.isChecked(),
            "cache_prompt": self.cache_prompt.isChecked(),
            "context_shift": self.context_shift.isChecked(),
            "no_webui": self.no_webui.isChecked(),
            "extra_args": self.extra_args.text(),
        }
        try:
            write_json_file_safely(self.settings_file, self.settings)
        except Exception as e:
            self.log(f"Ошибка сохранения настроек: {e}", "error")

    def load_data(self):
        """Загрузка настроек и профилей."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
                    self.exe_path.setText(self.settings.get("exe", ""))
                    self.bench_path.setText(self.settings.get("bench", ""))
                    self.model_dir.setText(self.settings.get("model_dir", ""))
                    self.opencode_config_path.setText(
                        self.settings.get("opencode_config", "")
                    )
                    self.pi_config_path.setText(self.settings.get("pi_config", ""))
                    idx = self.integration_target.findData(
                        self.settings.get("integration_target", "opencode")
                    )
                    if idx >= 0:
                        self.integration_target.setCurrentIndex(idx)
                    self.bench_prompt.setValue(self.settings.get("bench_prompt", 128))
                    self.bench_gen.setValue(self.settings.get("bench_gen", 256))
                    self.auto_params.setChecked(self.settings.get("auto_params", True))
                    self.use_mmproj.setChecked(self.settings.get("use_mmproj", True))
                    self.mmproj_offload.setChecked(
                        self.settings.get("mmproj_offload", True)
                    )
                    self.gpu_auto.setChecked(self.settings.get("gpu_auto", True))
                    self.gpu_layers.setValue(self.settings.get("gpu_layers", 33))
                    self.cpu_moe_layers.setValue(self.settings.get("cpu_moe_layers", 0))
                    self.ctx_size.setValue(self.settings.get("ctx_size", 4096))
                    self.threads.setValue(
                        self.settings.get("threads", os.cpu_count() or 4)
                    )
                    self.threads_batch.setValue(self.settings.get("threads_batch", 0))
                    self.port.setValue(self.settings.get("port", 8080))
                    self.temperature.setValue(self.settings.get("temperature", 0.7))
                    self.repeat_penalty.setValue(
                        self.settings.get("repeat_penalty", 1.1)
                    )
                    self.flash_attn.setChecked(self.settings.get("flash_attn", True))
                    self.fit_off.setChecked(self.settings.get("fit_off", True))
                    self.reasoning_mode.setCurrentText(
                        self.settings.get("reasoning_mode", "off")
                    )
                    self.use_mmap.setChecked(self.settings.get("use_mmap", True))
                    self.use_mlock.setChecked(self.settings.get("use_mlock", False))
                    self.verbose.setChecked(self.settings.get("verbose", False))
                    self.log_timestamps.setChecked(
                        self.settings.get("log_timestamps", False)
                    )
                    self.cache_type_k.setCurrentText(
                        self.settings.get("cache_type_k", "f16")
                    )
                    self.cache_type_v.setCurrentText(
                        self.settings.get("cache_type_v", "f16")
                    )
                    self.batch_size.setValue(self.settings.get("batch_size", 2048))
                    self.ubatch_size.setValue(self.settings.get("ubatch_size", 2048))
                    self.parallel_slots.setValue(self.settings.get("parallel_slots", 1))
                    self.ctx_checkpoints.setValue(
                        self.settings.get("ctx_checkpoints", -1)
                    )
                    self.cache_ram.setValue(self.settings.get("cache_ram", -2))
                    self.cont_batching.setChecked(
                        self.settings.get("cont_batching", True)
                    )
                    self.cache_prompt.setChecked(
                        self.settings.get("cache_prompt", True)
                    )
                    self.context_shift.setChecked(
                        self.settings.get("context_shift", False)
                    )
                    self.no_webui.setChecked(self.settings.get("no_webui", False))
                    self.extra_args.setText(self.settings.get("extra_args", ""))
                    cached = self.settings.get("model_cache", [])
                    if cached:
                        self.on_models_found(cached)
                    self.update_integration_model_label()
                    self.check_integration_models(silent=True)
            except Exception as e:
                self.log(f"Ошибка загрузки настроек: {e}", "error")
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
                    self.refresh_profile_list()
            except Exception as e:
                self.log(f"Ошибка загрузки профилей: {e}", "error")

    # === Обработка закрытия ===

    def closeEvent(self, event):
        """Обработка закрытия окна.

        При запущенном сервере сворачивает в трей вместо закрытия.
        При повторном закрытии — подтверждение и graceful shutdown.

        Args:
            event: Событие закрытия.
        """
        has_running = (
            self.process.state() != QProcess.ProcessState.NotRunning
            or self.bench_process.state() != QProcess.ProcessState.NotRunning
        )

        # Если сервер запущен и не было подтверждения — сворачиваем в трей
        if has_running and not hasattr(self, "_close_confirmed"):
            self.hide()
            if self.tray_icon:
                self.tray_icon.showMessage(
                    "LlamaServer GUI",
                    "Сервер работает в фоне. Двойной клик по иконке для отображения.",
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
            event.ignore()
            return

        self.save_settings()

        # Останавливаем batch log timer
        if self.log_timer.isActive():
            self.log_timer.stop()
            self.flush_log_buffer()

        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()

        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            self.bench_process.terminate()
            if not self.bench_process.waitForFinished(2000):
                self.bench_process.kill()

        if self.scanner and self.scanner.isRunning():
            self.scanner.requestInterruption()
            self.scanner.wait(1000)

        # Скрываем tray icon
        if self.tray_icon:
            self.tray_icon.hide()

        event.accept()


def main():
    """Точка входа в приложение."""
    app = QApplication(sys.argv)
    window = LlamaGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
