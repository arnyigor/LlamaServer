"""Главное окно и панели интерфейса."""

import os
import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QComboBox,
    QLabel,
    QSpinBox,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QFrame,
    QGroupBox,
    QMessageBox,
    QListWidget,
    QTableWidget,
    QTableWidgetItem,
    QDoubleSpinBox,
    QCheckBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QGridLayout,
    QAbstractItemView,
    QHeaderView,
    QTabWidget,
    QMenu,
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QFont, QIcon

from src.core.constants import (
    AUTO_SENTINEL,
    STATUS_COLOR_ERROR,
    STATUS_COLOR_MUTED,
    STATUS_COLOR_MUTED_DARK,
    STATUS_COLOR_PENDING,
    STATUS_COLOR_READY,
    STATUS_COLOR_RUNNING,
    STATUS_COLOR_WARNING,
    SAMPLING_AUTO_FLOAT,
    SAMPLING_AUTO_INT,
    SAMPLING_LAST_N_AUTO,
    SAMPLING_PENALTY_AUTO,
    SAMPLING_SEED_AUTO,
    SERVER_DEFAULT_SENTINEL,
)
from src.ui.widgets import CollapsiblePanel, NoWheelValueChangeFilter
from src.ui.panels.paths_panel import PathsPanel
from src.ui.mem_viz_widget import MemoryVisualizationWidget
from src.ui.autotune_widget import AutoTuneWidget
from src.ui.header_bar import HeaderBar
from src.ui.log_dock import LogDock
from src.ui.nav_rail import NavRail


class MainWindowUI(QMainWindow):
    # Порядок страниц навигации (label, key). NavRail и QStackedWidget в
    # _build_pages собираются в этом же порядке — единый источник истины.
    NAV_PAGES = [
        ("Dashboard", "dashboard"),
        ("Paths", "paths"),
        ("Launch", "performance"),
        ("Sampling", "sampling"),
        ("Server", "server"),
        ("Models", "library"),
        ("Integration", "integration"),
        ("Benchmark", "benchmark"),
        ("AutoTune", "autotune"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Llama Server Studio")
        self.setMinimumSize(1050, 560)
        self._apply_initial_geometry()
        self._apply_app_icon()

        self.models = []
        self.models_by_path = {}
        self.loading_profile = False

        self.ui_settings = QSettings("LlamaServerGUI", "UIState")

        self._setup_ui()
        self._hide_extra_widgets()
        self._no_wheel_value_filter = NoWheelValueChangeFilter(self)
        QApplication.instance().installEventFilter(self._no_wheel_value_filter)
        self._setup_tooltips()
        self._load_ui_state()

    def _apply_app_icon(self):
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
        icon_path = root / "assets" / "llama_server_icon.svg"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _apply_initial_geometry(self):
        preferred_w = 1570
        preferred_h = 820
        screen = QApplication.primaryScreen()
        if screen:
            available = screen.availableGeometry()
            width = min(preferred_w, max(1050, available.width() - 80))
            height = min(preferred_h, max(560, available.height() - 80))
            x = available.x() + max(0, (available.width() - width) // 2)
            y = available.y() + max(0, (available.height() - height) // 2)
            self.setGeometry(x, y, width, height)
        else:
            self.setGeometry(100, 100, preferred_w, preferred_h)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # === Шапка: профиль + Save flyout + язык ===
        self.header = HeaderBar()
        # Реэкспорт для совместимости (config/main.py могут ссылаться на language_combo)
        self.language_combo = self.header.language_combo
        root.addWidget(self.header)

        # === Постоянная тонкая полоса статуса (всегда видна) ===
        self.status_bar_widget = self._build_status_bar()
        root.addWidget(self.status_bar_widget)

        # === Контент: nav-рейл + страницы ===
        # _build_all_widgets() внутри вызывает _build_launch_controls_section()
        # (первый вызов) и создаёт self.launch_controls_widget — вызываем ДО
        # добавления панели запуска в root, чтобы не дублировать кнопки.
        self._build_all_widgets()
        root.addWidget(self.launch_controls_widget)
        self.pages = self._build_pages()

        content = QSplitter(Qt.Orientation.Horizontal)
        self.nav_rail = NavRail(self.NAV_PAGES)
        self.nav_rail.page_selected.connect(self._on_nav_selected)
        content.addWidget(self.nav_rail)
        content.addWidget(self.pages)
        content.setStretchFactor(0, 0)
        content.setStretchFactor(1, 1)
        content.setSizes([190, 1000])
        self.content_splitter = content

        # === Нижний док логов (Phase 2: вынесен в LogDock) ===
        self.log_dock = LogDock()
        # Реэкспорт атрибутов для совместимости с main.py
        self.logs = self.log_dock.logs
        self.autoscroll_logs = self.log_dock.autoscroll_logs
        self.copy_last_error_btn = self.log_dock.copy_last_error_btn
        self.open_diagnostics_btn = self.log_dock.open_diagnostics_btn
        self.log_dock.toggle_maximize.connect(self._toggle_log_maximize)

        # Вертикальный сплиттер: контент (nav|pages) + лог-док (ресайз + maximize)
        main_vsplit = QSplitter(Qt.Orientation.Vertical)
        main_vsplit.addWidget(self.content_splitter)
        main_vsplit.addWidget(self.log_dock)
        main_vsplit.setStretchFactor(0, 1)
        main_vsplit.setStretchFactor(1, 0)
        main_vsplit.setSizes([700, 180])
        self.main_vsplit = main_vsplit
        self._main_vsplit_docked = [700, 180]
        self._log_maximized = False
        root.addWidget(self.main_vsplit, 1)

        # Стартовая страница
        self.nav_rail.setCurrentRow(0)

    def _direct_layout_of(self, widget):
        """Возвращает layout, который напрямую содержит виджет (с учётом вложенности).

        ``widget.parentWidget().layout()`` отдаёт верхнеуровневый layout панели,
        тогда как сам виджет лежит во вложенном под-layout'е (напр. r8c внутри
        content_layout). Простой ``indexOf`` по верхнеуровневому layout не найдёт
        вложенный виджет, поэтому ищем layout, содержащий виджет, обходом по
        вложенности.
        """
        parent = widget.parentWidget()
        if parent is None:
            return None
        top = parent.layout()
        if top is None:
            return None
        stack = [top]
        while stack:
            lay = stack.pop()
            if lay.indexOf(widget) >= 0:
                return lay
            for i in range(lay.count()):
                sub = lay.itemAt(i).layout()
                if sub is not None:
                    stack.append(sub)
        return None

    def _label_before_widget(self, widget, layout):
        """QLabel-подпись непосредственно перед виджетом в layout (пропуская spacer'ы).

        Для самоподписанных чекбоксов (текст внутри самого виджета) возвращает
        None — отдельной подписи нет.
        """
        idx = layout.indexOf(widget)
        if idx <= 0:
            return None
        for i in range(idx - 1, -1, -1):
            item = layout.itemAt(i)
            if item is None:
                continue
            if item.spacerItem() is not None:
                continue
            w = item.widget()
            if w is None:
                continue
            if isinstance(w, QLabel):
                return w
            return None  # другой виджет перед нашим — подписи нет
        return None

    def _hide_extra_widgets(self) -> None:
        """EXTRA-параметры (managed=False) больше не управляются UI при сборке команды.

        Их виджеты создаём (для совместимости с FIELD_WIDGET_MAP и синхронизацией
        settings↔виджеты в config.py), но не показываем — они живут только в
        текстовом поле extra_args. Вынимаем их и сопутствующие подписи (QLabel),
        стоящие перед ними в том же layout, чтобы не оставлять «осиротевших»
        подписей рядом с пустым местом.
        """
        extra_attrs = [
            "ctx_checkpoints",
            "cache_ram",
            "split_mode",
            "main_gpu",
            "cuda_device",
            "use_mlock",
            "verbose",
            "log_timestamps",
            "context_shift",
            "no_webui",
            "use_chat_template",
            "chat_template_file",
            "chat_template_btn",
            "cuda_visible_devices",
            "cuda_module_loading",
            "kv_unified",
            "cont_batching",
            "cache_prompt",
            "use_mmap",
        ]
        for attr in extra_attrs:
            widget = getattr(self, attr, None)
            if widget is None or not hasattr(widget, "hide"):
                continue
            layout = self._direct_layout_of(widget)
            if layout is not None:
                label = self._label_before_widget(widget, layout)
                if label is not None:
                    layout.removeWidget(label)
                    label.hide()
                layout.removeWidget(widget)
            widget.hide()

    def _build_all_widgets(self):
        """Создаёт все виджеты (self.*) без привязки к старым контейнерам.

        Сборка в страницы навигации выполняется в ``_build_pages``. Порядок
        важен: performance создаёт панели, которые sampling наполняет контентом.
        """
        self._build_launch_controls_section()
        self._build_hf_models_section()
        self._build_paths_section()
        self._build_model_section()
        self._build_performance_section()
        self._build_sampling_section()
        self._build_integration_section()
        self._build_benchmark_section()
        self._build_cli_section()
        self.overview_content_widget = self._build_overview_content()

        self.autotune = AutoTuneWidget()

        self._apply_advanced_mode(self.advanced_mode_chk.isChecked())
        self._collect_runtime_lockable()

    def _collect_runtime_lockable(self):
        """Виджеты, блокируемые во время работы сервера/bench/autotune."""
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
            self.top_k,
            self.top_p,
            self.min_p,
            self.typical_p,
            self.repeat_penalty,
            self.repeat_last_n,
            self.presence_penalty,
            self.frequency_penalty,
            self.seed,
            self.use_mlock,
            self.verbose,
            self.log_timestamps,
            self.parallel_slots,
            self.kv_unified,
            self.speculative_mtp,
            self.spec_draft_n_max,
            self.spec_draft_p_min,
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
            self.context_shift,
            self.no_webui,
            self.jinja,
            self.reasoning_mode,
            self.enable_thinking,
            self.reasoning_effort,
            self.reasoning_preserve,
            self.reasoning_budget,
            self.reasoning_budget_message,
            self.extra_args,
            self.cuda_visible_devices,
            self.cuda_module_loading,
            self.bench_prompt,
            self.bench_gen,
            self.preset_name_combo,
            self.add_preset_btn,
            self.delete_preset_btn,
            self.save_preset_btn,
            self.cli_manual_mode,
            self.cli_apply_btn,
            self.cli_import_btn,
        ]
        self._runtime_lockable.extend(getattr(self, "ctx_quick_buttons", []))

    def _build_launch_controls_section(self):
        # === Постоянная панель управления (всегда видна) ===
        container = QWidget()
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(6)

        # Row 1: запуск + basic/advanced + preset
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self.start_btn = QPushButton(self.tr("Start Server"))
        self.start_btn.setStyleSheet(
            "background-color: "
            + STATUS_COLOR_RUNNING
            + "; color: white; font-weight: bold; padding: 8px;"
        )
        self.reload_btn = QPushButton(self.tr("Restart"), enabled=False)
        self.reload_btn.setVisible(False)
        self.reload_btn.setToolTip(
            "Restart the running server and apply the current model parameters"
        )
        self.reload_btn.setStyleSheet(
            "background-color: "
            + STATUS_COLOR_WARNING
            + "; color: white; font-weight: bold; padding: 8px;"
        )
        self.stop_btn = QPushButton(self.tr("Stop"), enabled=False)
        self.stop_btn.setStyleSheet(
            "background-color: "
            + STATUS_COLOR_ERROR
            + "; color: white; font-weight: bold; padding: 8px;"
        )
        self.force_stop_btn = QPushButton(self.tr("Force Stop"), enabled=True)
        self.force_stop_btn.setToolTip(
            "Immediately kills llama-server process tree if normal stop is stuck"
        )
        self.force_stop_btn.setStyleSheet(
            "background-color: #8B0000; color: white; font-weight: bold; padding: 8px;"
        )
        # Скрыта по умолчанию; появляется автоматически через ~5 сек после
        # нажатия Stop, если сервер не остановился (см. _reveal_force_stop).
        self.force_stop_btn.setVisible(False)

        # Basic/Advanced (Этап 3.2): скрывает продвинутые панели разом.
        self.advanced_mode_chk = QCheckBox(self.tr("Advanced settings"))
        self.advanced_mode_chk.setChecked(
            self.ui_settings.value("advancedMode", True, type=bool)
        )
        self.advanced_mode_chk.setToolTip(
            "Show the advanced Memory (KV-cache) panel on the Launch page. "
            "Basic mode keeps model selection, launch controls, Runtime stats "
            "and all other pages."
        )
        self.advanced_mode_chk.toggled.connect(self._apply_advanced_mode)
        row1.addWidget(self.start_btn)
        row1.addWidget(self.reload_btn)
        row1.addWidget(self.stop_btn)
        row1.addWidget(self.force_stop_btn)
        row1.addWidget(self.advanced_mode_chk)
        row1.addStretch(1)

        # Preset (per-model performance preset) — единственный механизм
        # сохранения настроек; перенесён сюда рядом с запуском.
        self.preset_name_combo = QComboBox()
        self.preset_name_combo.setEditable(False)
        self.preset_name_combo.addItem("default")
        self.preset_name_combo.setCurrentText("default")
        self.preset_name_combo.setMinimumWidth(130)
        self.preset_name_combo.setMaximumWidth(220)
        self.preset_name_combo.setToolTip(
            "Preset name for current model; use task names like coding or rag"
        )
        self.save_preset_btn = QPushButton(self.tr("Save Preset"))
        self.save_preset_btn.setToolTip(
            "Save parameters (ngl, ncmoe, etc.) under selected preset name"
        )
        self.add_preset_btn = QPushButton(self.tr("Add"))
        self.add_preset_btn.setToolTip("Add a named preset for the selected model")
        self.delete_preset_btn = QPushButton(self.tr("Delete"))
        self.delete_preset_btn.setEnabled(False)
        self.delete_preset_btn.setToolTip(
            "Delete the selected named preset. The default preset cannot be deleted."
        )
        self.preset_status = QLabel(self.tr("Preset: none"))
        self.preset_status.setStyleSheet("color: " + STATUS_COLOR_MUTED + ";")
        row1.addWidget(QLabel(self.tr("Preset:")))
        row1.addWidget(self.preset_name_combo)
        row1.addWidget(self.add_preset_btn)
        row1.addWidget(self.delete_preset_btn)
        row1.addWidget(self.save_preset_btn)
        vlay.addLayout(row1)

        # Row 2: компактный readout выбранной модели + ключевых параметров
        # (виден на любой странице навигации).
        self.launch_readout = QLabel(self.tr("Model: -"))
        self.launch_readout.setStyleSheet("color: #888; padding: 2px 0;")
        self.launch_readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        vlay.addWidget(self.launch_readout)

        self.launch_controls_widget = container

    def _apply_advanced_mode(self, advanced: bool):
        """Basic mode: спрятать продвинутые панели, оставить Model + Launch."""
        for panel in self._advanced_panels():
            panel.setVisible(bool(advanced))
        self.ui_settings.setValue("advancedMode", bool(advanced))

    def _advanced_panels(self):
        # Basic mode hides only the Memory (KV-cache) panel on the Launch page.
        # Sampling/Server and all other nav pages stay fully visible.
        return [
            panel for panel in (getattr(self, "adv_panel", None),) if panel is not None
        ]

    def _build_paths_section(self):
        # === 1. Пути === отдельный виджет (src/ui/panels/paths_panel.py);
        # атрибуты реэкспортируются, чтобы config/main.py работали как раньше.
        self.paths_panel = PathsPanel()
        self.paths_panel.browse_exe_requested.connect(self._browse_exe_clicked)
        self.paths_panel.browse_models_requested.connect(self._browse_model_dir_clicked)
        self.exe_path = self.paths_panel.exe_path
        self.bench_path = self.paths_panel.bench_path
        self.model_dir = self.paths_panel.model_dir
        self.cuda_version_combo = self.paths_panel.cuda_version_combo
        self.launch_cuda_version_combo = self.paths_panel.launch_cuda_version_combo
        self.update_llama_btn = self.paths_panel.update_llama_btn
        self.update_status = self.paths_panel.update_status
        self.update_progress = self.paths_panel.update_progress

    def _build_model_section(self):
        # === 2. Модель ===
        g_model = QGroupBox(self.tr("Model"))
        lm = QVBoxLayout(g_model)
        lm.setContentsMargins(12, 18, 12, 12)
        lm.setSpacing(8)

        scan_row = QHBoxLayout()
        self.scan_btn = QPushButton(self.tr("Scan"))
        scan_row.addWidget(self.scan_btn)
        lm.addLayout(scan_row)

        self.scan_status = QLabel(self.tr("Models not scanned"))
        self.scan_progress = QProgressBar(visible=False, minimum=0, maximum=0)
        lm.addWidget(self.scan_status)
        lm.addWidget(self.scan_progress)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        self.model_combo.setMinimumHeight(30)
        self.model_combo.setMaxVisibleItems(25)
        self.model_combo.setMinimumContentsLength(80)
        self.model_combo.setStyleSheet(
            "QComboBox { padding-left: 6px; padding-right: 34px; } "
            "QComboBox::drop-down { width: 30px; }"
        )

        lm.addWidget(QLabel(self.tr("Found GGUF:")))
        lm.addWidget(self.model_combo)

        self.auto_params = QCheckBox(self.tr("Auto setup ctx/GPU/cache by GGUF"))
        self.auto_params.setChecked(True)
        lm.addWidget(self.auto_params)

        self.model_info = QLabel(self.tr("Select model"))
        self.model_info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.model_id_label = QLabel(self.tr(""))
        self.model_id_label.setStyleSheet("color: #888; font-size: 10px;")
        self.model_id_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.model_id_label.setWordWrap(True)
        self.copy_model_btn = QPushButton(self.tr("Copy model path"))
        self.copy_model_btn.setFixedHeight(22)
        self.copy_model_btn.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        mrow = QHBoxLayout()
        mrow.addWidget(self.copy_model_btn)
        mrow.addWidget(self.model_id_label, 1)
        lm.addWidget(self.model_info)
        lm.addLayout(mrow)
        self.model_group = g_model

    def _build_hf_models_section(self):
        # === 2a. Локальные модели + загрузка с Hugging Face ===
        self.models_panel = CollapsiblePanel(
            self.tr("Local model manager and download"),
            settings_key="panel_models",
            collapsible=False,
        )
        local = self.models_panel.content_layout

        local_row = QHBoxLayout()
        self.local_models_refresh_btn = QPushButton(self.tr("Refresh local models"))
        self.local_models_delete_btn = QPushButton(self.tr("Delete selected"))
        self.local_models_delete_btn.setEnabled(False)
        self.local_models_delete_btn.setToolTip(
            "Safely delete the selected model folder/file from the Models base folder."
        )
        local_row.addWidget(QLabel(self.tr("All models under Models:")))
        local_row.addStretch(1)
        local_row.addWidget(self.local_models_refresh_btn)
        local_row.addWidget(self.local_models_delete_btn)
        local.addLayout(local_row)

        self.local_models_list = QTableWidget(0, 5)
        self.local_models_list.setHorizontalHeaderLabels(
            ["Name", "Type", "GGUF", "Size", "Examples"]
        )
        self.local_models_list.setMaximumHeight(130)
        self.local_models_list.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.local_models_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.local_models_list.horizontalHeader().setStretchLastSection(True)
        self.local_models_list.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.local_models_list.setToolTip(
            "Shows every local model folder/file found under the Models path, not only HF downloads. "
            "Projectors/MTP drafts are included in folder deletion but not shown as standalone models."
        )
        local.addWidget(self.local_models_list)

        self.local_models_status = QLabel(self.tr("Refresh to list local models"))
        self.local_models_status.setWordWrap(True)
        local.addWidget(self.local_models_status)

        # --- Подблок загрузки с Hugging Face (можно скрыть) ---
        self.show_hf_download = QCheckBox(self.tr("Show Hugging Face download"))
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
        self.hf_scan_btn = QPushButton(self.tr("Scan HF"))
        hf_filter_row.addWidget(self.hf_quant_filter, 1)
        hf_filter_row.addWidget(self.hf_scan_btn)
        hf.addLayout(hf_filter_row)

        hf_opts_row = QHBoxLayout()
        self.hf_include_mmproj = QCheckBox(self.tr("also vision/mmproj"))
        self.hf_include_mmproj.setChecked(True)
        self.hf_download_btn = QPushButton(self.tr("Download selected models"))
        self.hf_download_btn.setEnabled(False)
        self.hf_pause_btn = QPushButton(self.tr("Pause selected"))
        self.hf_pause_btn.setEnabled(False)
        self.hf_cancel_btn = QPushButton(self.tr("Cancel selected"))
        self.hf_cancel_btn.setEnabled(False)
        hf_opts_row.addWidget(self.hf_include_mmproj)
        hf_opts_row.addWidget(self.hf_download_btn)
        hf_opts_row.addWidget(self.hf_pause_btn)
        hf_opts_row.addWidget(self.hf_cancel_btn)
        hf_opts_row.addStretch(1)
        hf.addLayout(hf_opts_row)

        self.hf_files = QTableWidget(0, 4)
        self.hf_files.setHorizontalHeaderLabels(["Name", "Quant", "Size", "Progress"])
        self.hf_files.setMaximumHeight(120)
        self.hf_files.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.hf_files.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.hf_files.horizontalHeader().setStretchLastSection(True)
        self.hf_files.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.hf_files.setToolTip(
            "Select one or several GGUF files (Ctrl/Shift). Each file is downloaded "
            "as an independent concurrent task. Vision projector is added once."
        )
        hf.addWidget(self.hf_files)

        self.hf_status = QLabel(self.tr("Paste repo and scan"))
        self.hf_status.setWordWrap(True)
        self.hf_progress = QProgressBar(visible=False, minimum=0, maximum=100)
        hf.addWidget(self.hf_status)
        hf.addWidget(self.hf_progress)

        hf.addWidget(QLabel(self.tr("Downloads (select tasks to pause/cancel):")))
        self.hf_downloads = QTableWidget(0, 6)
        self.hf_downloads.setHorizontalHeaderLabels(
            ["Name", "Status", "Progress", "Size", "Speed", "ETA"]
        )
        self.hf_downloads.setMaximumHeight(220)
        self.hf_downloads.setWordWrap(False)
        self.hf_downloads.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.hf_downloads.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.hf_downloads.horizontalHeader().setStretchLastSection(True)
        self.hf_downloads.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.hf_downloads.setToolTip(
            "Independent parallel downloads with downloaded/total size, remaining size, "
            "speed and ETA. Select one or several tasks before Pause/Cancel."
        )
        hf.addWidget(self.hf_downloads)

        hf_local_row = QHBoxLayout()
        self.hf_refresh_local_btn = QPushButton(self.tr("Refresh local"))
        self.hf_delete_local_folder_btn = QPushButton(self.tr("Delete local folder"))
        self.hf_delete_local_folder_btn.setEnabled(False)
        hf_local_row.addWidget(QLabel(self.tr("Local files:")))
        hf_local_row.addStretch(1)
        hf_local_row.addWidget(self.hf_refresh_local_btn)
        hf_local_row.addWidget(self.hf_delete_local_folder_btn)
        hf.addLayout(hf_local_row)

        self.hf_local_files = QTableWidget(0, 3)
        self.hf_local_files.setHorizontalHeaderLabels(["Name", "Status", "Size"])
        self.hf_local_files.setMaximumHeight(90)
        self.hf_local_files.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.hf_local_files.horizontalHeader().setStretchLastSection(True)
        self.hf_local_files.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.hf_local_files.setToolTip(
            "Files already present in <Models>/<author>/<model>. Delete local folder removes the whole repo folder including mmproj/vision files."
        )
        hf.addWidget(self.hf_local_files)
        local.addWidget(self.hf_section)
        self.show_hf_download.toggled.connect(self.hf_section.setVisible)

        self.runtime_stats_group = QGroupBox(self.tr("Runtime stats"))
        stats = QGridLayout(self.runtime_stats_group)
        stats.setContentsMargins(12, 18, 12, 12)
        stats.setHorizontalSpacing(10)
        stats.setVerticalSpacing(6)
        self.speed_label = QLabel(self.tr("Speed: -"))
        self.speed_label.setTextFormat(Qt.RichText)
        self.speed_label.setProperty("class", "mono")
        self.tokens_label = QLabel(self.tr("Tokens: total 0 | task 0"))
        self.tokens_label.setTextFormat(Qt.RichText)
        self.tokens_label.setProperty("class", "mono")
        self.request_tokens_label = QLabel(self.tr("Request: -"))
        self.request_tokens_label.setTextFormat(Qt.RichText)
        self.request_tokens_label.setProperty("class", "mono")
        self.tokens_saved_label = QLabel(self.tr("Saved: 0"))
        self.tokens_saved_label.setTextFormat(Qt.RichText)
        self.tokens_saved_label.setProperty("class", "mono")
        self.active_time_label = QLabel(
            self.tr("Work time: 0:00 (Prompt 0:00 | Gen 0:00)")
        )
        self.active_time_label.setTextFormat(Qt.RichText)
        self.active_time_label.setProperty("class", "mono")
        self.current_time_label = QLabel(
            self.tr("Last request: 0:00 (Prompt 0:00 | Gen 0:00)")
        )
        self.current_time_label.setTextFormat(Qt.RichText)
        self.current_time_label.setProperty("class", "mono")
        self.tokens_reset_btn = QPushButton(self.tr("Save task & reset"))
        self.tokens_reset_btn.setToolTip(
            "Save current task token count to Saved and start the next task from zero. "
            "Resets the task counter, Current time and Request label."
        )
        self.export_stats_btn = QPushButton(self.tr("Export stats"))
        self.export_stats_btn.setToolTip(
            "Export current runtime counters to a JSON file."
        )
        self.copy_stats_md_btn = QPushButton(self.tr("Copy stats MD"))
        self.copy_stats_md_btn.setToolTip(
            "Copy current runtime counters to the clipboard as Markdown."
        )
        self.reset_session_btn = QPushButton(self.tr("Reset session"))
        self.reset_session_btn.setToolTip(
            "Zero all live runtime stats: total/task tokens, prompt/generated, "
            "Active and Current time, Request label. Saved history is kept."
        )
        self.reset_saved_btn = QPushButton(self.tr("Reset saved"))
        self.reset_saved_btn.setToolTip(
            "Zero the accumulated Saved history (last and total)."
        )
        for btn in [
            self.tokens_reset_btn,
            self.export_stats_btn,
            self.copy_stats_md_btn,
            self.reset_session_btn,
            self.reset_saved_btn,
        ]:
            btn.setMinimumWidth(120)
            btn.setMaximumWidth(150)

        stats.addWidget(self.speed_label, 0, 0, 1, 3)
        stats.addWidget(self.tokens_label, 1, 0, 1, 2)
        stats.addWidget(self.tokens_reset_btn, 1, 2)
        stats.addWidget(self.request_tokens_label, 2, 0)
        stats.addWidget(self.tokens_saved_label, 2, 1)
        stats.addWidget(self.reset_saved_btn, 2, 2)
        stats.addWidget(self.active_time_label, 3, 0, 1, 2)
        stats.addWidget(self.reset_session_btn, 3, 2)
        stats.addWidget(self.current_time_label, 4, 0, 1, 2)
        stats.addWidget(self.export_stats_btn, 4, 2)
        stats.addWidget(self.copy_stats_md_btn, 5, 2)
        stats.setColumnStretch(1, 1)

    def _build_performance_section(self):
        # === 3. Производительность ===
        self.g_launch = QGroupBox(self.tr("Launch settings"))
        launch = QVBoxLayout(self.g_launch)
        launch.setContentsMargins(12, 18, 12, 12)
        launch.setSpacing(8)

        self.adv_panel = CollapsiblePanel(
            self.tr("Память (KV-кэш)"),
            settings_key="panel_adv",
            collapsible=False,
        )
        lperf = self.adv_panel.content_layout
        lperf.setContentsMargins(8, 6, 8, 6)
        lperf.setSpacing(8)
        self.sampling_panel = CollapsiblePanel(
            self.tr("Generation: Sampling and Penalties"),
            settings_key="panel_sampling",
            collapsible=False,
        )
        sampling = self.sampling_panel.content_layout
        sampling.setContentsMargins(8, 6, 8, 6)
        sampling.setSpacing(8)
        self.server_panel = CollapsiblePanel(
            self.tr("Server, Templates and Diagnostics"),
            settings_key="panel_server",
            collapsible=False,
        )
        server_opts = self.server_panel.content_layout
        server_opts.setContentsMargins(8, 6, 8, 6)
        server_opts.setSpacing(8)
        # Launch settings (g_launch) keeps only context, vision and CUDA.
        # GPU offload, KV cache type, attention and the Memory (KV-cache)
        # panel are moved to the Sampling page as separate blocks.

        self.launch_summary_group = QGroupBox(self.tr("Launch preflight"))
        launch_summary = QVBoxLayout(self.launch_summary_group)
        launch_summary.setContentsMargins(12, 18, 12, 12)
        launch_summary.setSpacing(6)
        self.preflight_status = QLabel(
            self.tr("Select a model to estimate launch readiness")
        )
        self.preflight_status.setWordWrap(True)
        self.preflight_status.setStyleSheet(
            "font-weight: bold; color: " + STATUS_COLOR_MUTED_DARK + ";"
        )
        self.preflight_model = QLabel(self.tr("Model: -"))
        self.preflight_context = QLabel(self.tr("Context: -"))
        self.preflight_kv = QLabel(self.tr("KV: -"))
        self.preflight_gpu = QLabel(self.tr("GPU offload: -"))
        self.preflight_mtp = QLabel(self.tr("MTP: -"))
        self.preflight_endpoint = QLabel(self.tr("Endpoint: -"))
        self.preflight_warning = QLabel(self.tr(""))
        self.preflight_warning.setWordWrap(True)
        self.preflight_warning.setStyleSheet("color: " + STATUS_COLOR_PENDING + ";")
        for label in [
            self.preflight_model,
            self.preflight_context,
            self.preflight_kv,
            self.preflight_gpu,
            self.preflight_mtp,
            self.preflight_endpoint,
        ]:
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            launch_summary.addWidget(label)
        launch_summary.addWidget(self.preflight_warning)

        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(0, 999)
        self.gpu_layers.setValue(33)
        self.gpu_auto = QCheckBox(self.tr("auto"))
        self.gpu_auto.setChecked(True)
        self.gpu_layers_all = QCheckBox(self.tr("all"))

        def sync_gpu_layer_controls():
            self.gpu_layers.setDisabled(
                self.gpu_auto.isChecked() or self.gpu_layers_all.isChecked()
            )
            self.gpu_auto.setDisabled(self.gpu_layers_all.isChecked())

        self.gpu_auto.toggled.connect(lambda _checked: sync_gpu_layer_controls())
        self.gpu_layers_all.toggled.connect(lambda _checked: sync_gpu_layer_controls())
        sync_gpu_layer_controls()
        self.cpu_moe_layers = QSpinBox()
        self.cpu_moe_layers.setRange(AUTO_SENTINEL, 200)
        self.cpu_moe_layers.setValue(AUTO_SENTINEL)
        self.cpu_moe_layers.setSpecialValueText("auto")

        self.cuda_status_label = QLabel(self.tr("CUDA build: not checked"))
        self.cuda_status_label.setWordWrap(True)
        self.cuda_status_label.setStyleSheet("color: " + STATUS_COLOR_MUTED_DARK + ";")
        self.cuda_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        r_cuda = QHBoxLayout()
        r_cuda.addWidget(QLabel(self.tr("CUDA build:")))
        r_cuda.addWidget(self.launch_cuda_version_combo)
        r_cuda.addWidget(self.cuda_status_label, 1)
        launch.addLayout(r_cuda)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel(self.tr("GPU offload (-ngl):")))
        r1.addWidget(self.gpu_layers)
        r1.addWidget(self.gpu_auto)
        r1.addWidget(self.gpu_layers_all)
        r1.addStretch(1)

        self.ctx_size = QSpinBox()
        self.ctx_size.setRange(AUTO_SENTINEL, 1048576)
        self.ctx_size.setSingleStep(512)
        self.ctx_size.setValue(AUTO_SENTINEL)
        self.ctx_size.setSpecialValueText("auto")

        r2a = QHBoxLayout()
        r2a.addWidget(QLabel(self.tr("Context Size (-c):")))
        r2a.addWidget(self.ctx_size)
        self.ctx_help_btn = QToolButton()
        self.ctx_help_btn.setText("?")
        self.ctx_help_btn.setToolTip("Open detailed context/VRAM guidance")
        r2a.addWidget(self.ctx_help_btn)
        r2a.addSpacing(10)
        r2a.addWidget(QLabel(self.tr("CPU MoE (-ncmoe):")))
        r2a.addWidget(self.cpu_moe_layers)
        self.ncmoe_help_btn = QToolButton()
        self.ncmoe_help_btn.setText("?")
        self.ncmoe_help_btn.setToolTip("Open detailed CPU MoE/VRAM guidance")
        r2a.addWidget(self.ncmoe_help_btn)
        r2a.addStretch(1)
        launch.addLayout(r2a)

        # Быстрые кнопки контекста — на отдельном ряду, чтобы не перегружать
        # основной ряд spinbox'ов. Стандартные степени 2 (8K..256K); нестандартные
        # 24K/41K/65K убраны (вводят в заблуждение: 40960 = 40K, 65536 = 64K).
        r2b = QHBoxLayout()
        r2b.addWidget(QLabel(self.tr("Quick:")))
        self.ctx_quick_buttons = []
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
            self.ctx_quick_buttons.append(btn)
            r2b.addWidget(btn)
            if label == "32K":
                r2b.addSpacing(10)
        r2b.addStretch(1)
        launch.addLayout(r2b)

        # Vision (mmproj) — перенесено из секции модели на страницу Запуск.
        mmproj_row = QHBoxLayout()
        self.use_mmproj = QCheckBox(self.tr("Use mmproj"))
        self.use_mmproj.setChecked(True)
        self.mmproj_offload = QCheckBox(self.tr("mmproj offload"))
        self.mmproj_offload.setChecked(True)
        mmproj_row.addWidget(self.use_mmproj)
        mmproj_row.addWidget(self.mmproj_offload)
        launch.addLayout(mmproj_row)

        # GPU offload (-ngl) — блок на странице Сэмплинг.
        gpu_offload_box = QGroupBox(self.tr("GPU offload (-ngl)"))
        gpu_offload_box.setLayout(r1)
        sampling.addWidget(gpu_offload_box)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(AUTO_SENTINEL, 32768)
        self.batch_size.setSingleStep(128)
        self.batch_size.setValue(AUTO_SENTINEL)
        self.batch_size.setSpecialValueText("auto")
        self.ubatch_size = QSpinBox()
        self.ubatch_size.setRange(AUTO_SENTINEL, 8192)
        self.ubatch_size.setSingleStep(64)
        self.ubatch_size.setValue(AUTO_SENTINEL)
        self.ubatch_size.setSpecialValueText("auto")

        self.cache_type_k = QComboBox()
        self.cache_type_v = QComboBox()
        for ct in ["f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "f32"]:
            self.cache_type_k.addItem(ct)
            self.cache_type_v.addItem(ct)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel(self.tr("KV K / V:")))
        r3.addWidget(self.cache_type_k)
        r3.addWidget(self.cache_type_v)
        r3.addStretch(1)
        kv_cache_box = QGroupBox(self.tr("KV cache type"))
        kv_cache_box.setLayout(r3)
        sampling.addWidget(kv_cache_box)

        # Batch / UBatch — генеративные параметры, на страницу Сэмплинг.
        r_batch = QHBoxLayout()
        r_batch.addWidget(QLabel(self.tr("Batch / UBatch (-b / -ub):")))
        r_batch.addWidget(self.batch_size)
        r_batch.addWidget(self.ubatch_size)
        r_batch.addStretch(1)
        sampling.addLayout(r_batch)

        self.threads = QSpinBox()
        self.threads.setRange(1, 64)
        self.threads.setValue(os.cpu_count() or 4)
        self.threads_batch = QSpinBox()
        self.threads_batch.setRange(0, 64)
        self.threads_batch.setSpecialValueText("same")
        self.threads_batch.setValue(0)
        r4 = QHBoxLayout()
        r4.addWidget(QLabel(self.tr("Threads gen / batch (-t / -tb):")))
        r4.addWidget(self.threads)
        r4.addWidget(self.threads_batch)
        sampling.addLayout(r4)

        self.flash_attn = QCheckBox(self.tr("Flash Attention (-fa)"))
        self.flash_attn.setChecked(True)
        self.fit_off = QCheckBox(self.tr("Fit off (--fit off)"))
        self.fit_off.setChecked(True)
        r6 = QHBoxLayout()
        r6.addWidget(self.flash_attn)
        r6.addWidget(self.fit_off)
        r6.addStretch(1)
        attention_box = QGroupBox(self.tr("Attention / Fit"))
        attention_box.setLayout(r6)
        sampling.addWidget(attention_box)

        self.reasoning_mode = QComboBox()
        self.reasoning_mode.addItems(["off", "auto", "on"])
        self.reasoning_mode.setCurrentText("off")
        self.enable_thinking = QComboBox()
        self.enable_thinking.addItems(["off", "false", "true"])
        self.enable_thinking.setCurrentText("off")
        self.reasoning_effort = QComboBox()
        self.reasoning_effort.addItems(["", "low", "medium", "xhigh"])
        self.reasoning_effort.setCurrentText("")
        self.reasoning_preserve = QComboBox()
        self.reasoning_preserve.addItems(["off", "preserve", "no-preserve"])
        self.reasoning_preserve.setCurrentText("off")
        self.reasoning_budget = QSpinBox()
        self.reasoning_budget.setRange(0, 32767)
        self.reasoning_budget.setValue(0)
        self.reasoning_budget_message = QLineEdit(placeholderText="optional")
        r7 = QHBoxLayout()
        r7.addWidget(QLabel(self.tr("Reasoning (--reasoning):")))
        r7.addWidget(self.reasoning_mode)
        r7.addSpacing(10)
        r7.addWidget(QLabel(self.tr("Thinking:")))
        r7.addWidget(self.enable_thinking)
        r7.addSpacing(10)
        r7.addWidget(QLabel(self.tr("Effort (--reasoning-effort):")))
        r7.addWidget(self.reasoning_effort)
        r7.addSpacing(10)
        r7.addWidget(QLabel(self.tr("Preserve (--reasoning-preserve):")))
        r7.addWidget(self.reasoning_preserve)
        sampling.addLayout(r7)

        # Budget и Budget msg — вертикально ("друг под другом"): текстовое
        # поле рядом со спинбоксом неудобно, а сообщение удобнее на всю ширину.
        r7b = QVBoxLayout()
        r7b.setSpacing(4)
        row_budget = QHBoxLayout()
        row_budget.addWidget(QLabel(self.tr("Budget (--reasoning-budget):")))
        row_budget.addWidget(self.reasoning_budget)
        row_budget.addStretch(1)
        r7b.addLayout(row_budget)
        row_budget_msg = QHBoxLayout()
        row_budget_msg.addWidget(
            QLabel(self.tr("Budget msg (--reasoning-budget-message):"))
        )
        row_budget_msg.addWidget(self.reasoning_budget_message, 1)
        r7b.addLayout(row_budget_msg)
        sampling.addLayout(r7b)

        self.host = QLineEdit(placeholderText="127.0.0.1")
        self.host.setText("127.0.0.1")
        self.host.setMaximumWidth(120)
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(8080)
        self.parallel_slots = QSpinBox()
        self.parallel_slots.setRange(AUTO_SENTINEL, 16)
        self.parallel_slots.setValue(AUTO_SENTINEL)
        self.parallel_slots.setSpecialValueText("auto")
        r8 = QHBoxLayout()
        r8.addWidget(QLabel(self.tr("Host:")))
        r8.addWidget(self.host)
        r8.addWidget(QLabel(self.tr("Port:")))
        r8.addWidget(self.port)
        r8.addSpacing(10)
        r8.addWidget(QLabel(self.tr("Slots (-np):")))
        r8.addWidget(self.parallel_slots)
        server_opts.addLayout(r8)

        self.kv_unified = QCheckBox(self.tr("KV unified (-kvu)"))
        self.speculative_mtp = QCheckBox(self.tr("MTP speculative"))
        self.spec_draft_n_max = QSpinBox()
        self.spec_draft_n_max.setRange(1, 32)
        self.spec_draft_n_max.setValue(8)
        self.spec_draft_n_max.setToolTip(
            "Maximum speculative MTP tokens. Coding default: 8; conservative: 2-4; aggressive: 16."
        )
        self.spec_draft_p_min = QDoubleSpinBox()
        self.spec_draft_p_min.setRange(0.0, 1.0)
        self.spec_draft_p_min.setDecimals(2)
        self.spec_draft_p_min.setSingleStep(0.05)
        self.spec_draft_p_min.setValue(0.8)
        self.spec_draft_p_min.setToolTip(
            "Minimum MTP confidence. 0.8 avoids expensive long speculation when the draft head is uncertain."
        )
        self.spec_draft_gpu_layers = QLineEdit(placeholderText="all")
        self.spec_draft_gpu_layers.setText("all")
        self.spec_draft_gpu_layers.setMaximumWidth(60)
        r8b = QHBoxLayout()
        r8b.addWidget(self.kv_unified)
        r8b.addWidget(self.speculative_mtp)
        r8b.addSpacing(10)
        r8b.addWidget(QLabel(self.tr("MTP n-max / p-min / draft ngl:")))
        r8b.addWidget(self.spec_draft_n_max)
        r8b.addWidget(self.spec_draft_p_min)
        r8b.addWidget(self.spec_draft_gpu_layers)
        sampling.addLayout(r8b)

        r8b2 = QHBoxLayout()
        r8b2.addWidget(QLabel(self.tr("MTP draft GGUF:")))
        self.spec_draft_model_path = QLineEdit(
            placeholderText="Auto-detected, or browse for separate MTP GGUF"
        )
        self.spec_draft_model_path.setToolTip(
            "Optional separate MTP/draft GGUF. Gemma 4 packages often include it in an MTP folder; Qwen3.6 may require a separate file."
        )
        self.spec_draft_model_btn = QPushButton(self.tr("..."))
        self.spec_draft_model_btn.setFixedWidth(32)
        self.spec_draft_model_btn.clicked.connect(
            lambda _checked=False: self._browse_mtp_draft_clicked()
        )
        r8b2.addWidget(self.spec_draft_model_path, 1)
        r8b2.addWidget(self.spec_draft_model_btn)
        sampling.addLayout(r8b2)

        self.cuda_device = QLineEdit(placeholderText="CUDA0")
        self.cuda_device.setMaximumWidth(80)
        self.spec_draft_device = QLineEdit(placeholderText="CUDA0")
        self.spec_draft_device.setMaximumWidth(80)
        self.split_mode = QComboBox()
        self.split_mode.addItems(["", "none", "layer", "row"])
        self.main_gpu = QSpinBox()
        self.main_gpu.setRange(AUTO_SENTINEL, 16)
        self.main_gpu.setSpecialValueText("auto")
        self.main_gpu.setValue(AUTO_SENTINEL)
        r8c = QHBoxLayout()
        r8c.addWidget(QLabel(self.tr("Device:")))
        r8c.addWidget(self.cuda_device)
        r8c.addWidget(QLabel(self.tr("Draft device:")))
        r8c.addWidget(self.spec_draft_device)
        r8c.addWidget(QLabel(self.tr("Split:")))
        r8c.addWidget(self.split_mode)
        r8c.addWidget(QLabel(self.tr("Main GPU:")))
        r8c.addWidget(self.main_gpu)
        sampling.addLayout(r8c)

        self.ctx_checkpoints = QSpinBox()
        self.ctx_checkpoints.setRange(AUTO_SENTINEL, 128)
        self.ctx_checkpoints.setSpecialValueText("default")
        self.ctx_checkpoints.setValue(AUTO_SENTINEL)
        self.cache_ram = QSpinBox()
        self.cache_ram.setRange(SERVER_DEFAULT_SENTINEL, 262144)
        self.cache_ram.setSpecialValueText("default")
        self.cache_ram.setValue(SERVER_DEFAULT_SENTINEL)
        r9 = QHBoxLayout()
        r9.addWidget(QLabel(self.tr("Ctx Checkpoints:")))
        r9.addWidget(self.ctx_checkpoints)
        r9.addSpacing(10)
        r9.addWidget(QLabel(self.tr("Cache RAM (MiB):")))
        r9.addWidget(self.cache_ram)
        lperf.addLayout(r9)

    def _build_sampling_section(self):
        # === 4. Generation / Sampling ===
        # Layout-объекты панелей создаются в performance-секции; здесь и
        # ниже (server/diagnostics-часть) используем те же самые объекты.
        lperf = self.adv_panel.content_layout
        sampling = self.sampling_panel.content_layout
        server_opts = self.server_panel.content_layout
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(SAMPLING_AUTO_FLOAT, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(SAMPLING_AUTO_FLOAT)
        self.temperature.setDecimals(2)
        self.temperature.setSpecialValueText("auto")
        self.repeat_penalty = QDoubleSpinBox()
        self.repeat_penalty.setRange(SAMPLING_AUTO_FLOAT, 2.0)
        self.repeat_penalty.setSingleStep(0.01)
        self.repeat_penalty.setValue(SAMPLING_AUTO_FLOAT)
        self.repeat_penalty.setDecimals(2)
        self.repeat_penalty.setSpecialValueText("auto")
        self.top_k = QSpinBox()
        self.top_k.setRange(SAMPLING_AUTO_INT, 10000)
        self.top_k.setValue(SAMPLING_AUTO_INT)
        self.top_k.setSpecialValueText("auto")
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(SAMPLING_AUTO_FLOAT, 1.0)
        self.top_p.setSingleStep(0.01)
        self.top_p.setDecimals(3)
        self.top_p.setValue(SAMPLING_AUTO_FLOAT)
        self.top_p.setSpecialValueText("auto")
        self.min_p = QDoubleSpinBox()
        self.min_p.setRange(SAMPLING_AUTO_FLOAT, 1.0)
        self.min_p.setSingleStep(0.01)
        self.min_p.setDecimals(3)
        self.min_p.setValue(SAMPLING_AUTO_FLOAT)
        self.min_p.setSpecialValueText("auto")
        self.typical_p = QDoubleSpinBox()
        self.typical_p.setRange(SAMPLING_AUTO_FLOAT, 1.0)
        self.typical_p.setSingleStep(0.01)
        self.typical_p.setDecimals(3)
        self.typical_p.setValue(SAMPLING_AUTO_FLOAT)
        self.typical_p.setSpecialValueText("auto")
        self.repeat_last_n = QSpinBox()
        self.repeat_last_n.setRange(SAMPLING_LAST_N_AUTO, 1048576)
        self.repeat_last_n.setValue(SAMPLING_LAST_N_AUTO)
        self.repeat_last_n.setSpecialValueText("auto")
        self.presence_penalty = QDoubleSpinBox()
        self.presence_penalty.setRange(SAMPLING_PENALTY_AUTO, 2.0)
        self.presence_penalty.setSingleStep(0.05)
        self.presence_penalty.setDecimals(2)
        self.presence_penalty.setValue(SAMPLING_PENALTY_AUTO)
        self.presence_penalty.setSpecialValueText("auto")
        self.frequency_penalty = QDoubleSpinBox()
        self.frequency_penalty.setRange(SAMPLING_PENALTY_AUTO, 2.0)
        self.frequency_penalty.setSingleStep(0.05)
        self.frequency_penalty.setDecimals(2)
        self.frequency_penalty.setValue(SAMPLING_PENALTY_AUTO)
        self.frequency_penalty.setSpecialValueText("auto")
        self.seed = QSpinBox()
        self.seed.setRange(SAMPLING_SEED_AUTO, 2147483647)
        self.seed.setValue(SAMPLING_SEED_AUTO)
        self.seed.setSpecialValueText("auto")

        sampling_grid = QGridLayout()
        sampling_grid.setHorizontalSpacing(10)
        sampling_grid.setVerticalSpacing(6)
        sampling_fields = [
            ("Temperature:", self.temperature),
            ("Top K:", self.top_k),
            ("Top P:", self.top_p),
            ("Min P:", self.min_p),
            ("Typical P:", self.typical_p),
            ("Seed:", self.seed),
            ("Repeat penalty:", self.repeat_penalty),
            ("Repeat last N:", self.repeat_last_n),
            ("Presence penalty:", self.presence_penalty),
            ("Frequency penalty:", self.frequency_penalty),
        ]
        for index, (label, widget) in enumerate(sampling_fields):
            row, column = divmod(index, 2)
            sampling_grid.addWidget(QLabel(label), row, column * 2)
            sampling_grid.addWidget(widget, row, column * 2 + 1)
        sampling_help = {
            self.temperature: "Randomness (0.0–2.0).\nCLI: --temp\nauto = server default",
            self.top_k: "Keep K most likely tokens; 0 disables.\nCLI: --top-k",
            self.top_p: "Nucleus sampling threshold; 1.0 disables.\nCLI: --top-p",
            self.min_p: "Min probability relative to best token; 0 disables.\nCLI: --min-p",
            self.typical_p: "Locally typical sampling; 1.0 disables.\nCLI: --typical",
            self.seed: "RNG seed. -1 = random; auto omits the flag.\nCLI: --seed",
            self.repeat_penalty: "Penalty for repeated token sequences; 1.0 disables.\nCLI: --repeat-penalty",
            self.repeat_last_n: "Recent tokens penalized; -1 = full context.\nCLI: --repeat-last-n",
            self.presence_penalty: "Penalty based on token presence; 0 disables.\nCLI: --presence-penalty",
            self.frequency_penalty: "Penalty based on token repetition count; 0 disables.\nCLI: --frequency-penalty",
        }
        for widget, help_text in sampling_help.items():
            widget.setToolTip(help_text)
        sampling.addLayout(sampling_grid)

        self.use_mlock = QCheckBox(self.tr("mlock"))
        self.verbose = QCheckBox(self.tr("verbose"))
        self.log_timestamps = QCheckBox(self.tr("log timestamps"))
        memory_flags = QHBoxLayout()
        memory_flags.addWidget(self.use_mlock)
        memory_flags.addStretch(1)
        lperf.addLayout(memory_flags)
        diagnostics_flags = QHBoxLayout()
        diagnostics_flags.addWidget(self.verbose)
        diagnostics_flags.addWidget(self.log_timestamps)
        diagnostics_flags.addStretch(1)
        server_opts.addLayout(diagnostics_flags)

        self.cuda_visible_devices = QLineEdit(placeholderText="CUDA_VISIBLE_DEVICES")
        self.cuda_visible_devices.setMaximumWidth(120)
        self.cuda_module_loading = QLineEdit(placeholderText="CUDA_MODULE_LOADING")
        self.cuda_module_loading.setText("LAZY")
        self.cuda_module_loading.setMaximumWidth(80)
        s_cuda = QHBoxLayout()
        s_cuda.addWidget(QLabel(self.tr("CUDA env:")))
        s_cuda.addWidget(self.cuda_visible_devices)
        s_cuda.addWidget(self.cuda_module_loading)
        s_cuda.addStretch(1)
        lperf.addLayout(s_cuda)

        self.context_shift = QCheckBox(self.tr("context shift"))
        self.no_webui = QCheckBox(self.tr("no webui"))
        self.jinja = QCheckBox(self.tr("jinja"))
        s3 = QHBoxLayout()
        for w in [
            self.context_shift,
            self.no_webui,
            self.jinja,
        ]:
            s3.addWidget(w)
        server_opts.addLayout(s3)

        s_tpl = QHBoxLayout()
        self.use_chat_template = QCheckBox(self.tr("--chat-template-file"))
        self.chat_template_file = QLineEdit(
            placeholderText="Path to .jinja chat template"
        )
        self.chat_template_file.setToolTip(
            "Override the model's built-in chat template with an external .jinja file. "
            "Required for Qwen3.6 tool calls when using the relaxed template."
        )
        self.chat_template_btn = QPushButton(self.tr("..."))
        self.chat_template_btn.setFixedWidth(32)
        self.chat_template_btn.clicked.connect(
            lambda _checked=False: self._browse_chat_template_clicked()
        )
        s_tpl.addWidget(self.use_chat_template)
        s_tpl.addWidget(self.chat_template_file, 1)
        s_tpl.addWidget(self.chat_template_btn)
        server_opts.addLayout(s_tpl)

        sampling.addWidget(
            QLabel(self.tr("Extra params (only uncommon llama-server flags):"))
        )
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText(
            "--dry-multiplier 0.8 --xtc-probability 0.1 ..."
        )
        sampling.addWidget(self.extra_args)

    def _build_integration_section(self):
        # === 5. Интеграция ===
        self.int_panel = CollapsiblePanel(
            self.tr("Integration (OpenCode / PI)"),
            settings_key="panel_integration",
            collapsible=False,
        )

        # Каждый путь завёрнут в контейнер, чтобы показывать только тот,
        # что соответствует выбранному Target (см. _on_integration_target_changed).
        self.opencode_row = QWidget()
        oc_layout = QHBoxLayout(self.opencode_row)
        oc_layout.setContentsMargins(0, 0, 0, 0)
        oc_layout.addWidget(QLabel(self.tr("OpenCode JSON:")))
        self.opencode_config_path = QLineEdit(placeholderText="Path to opencode.json")
        oc_btn = QPushButton(self.tr("..."))
        oc_btn.clicked.connect(self._browse_opencode_clicked)
        oc_layout.addWidget(self.opencode_config_path)
        oc_layout.addWidget(oc_btn)
        self.int_panel.add_widget(self.opencode_row)

        self.pi_row = QWidget()
        pi_layout = QHBoxLayout(self.pi_row)
        pi_layout.setContentsMargins(0, 0, 0, 0)
        pi_layout.addWidget(QLabel(self.tr("PI JSON:")))
        self.pi_config_path = QLineEdit(placeholderText="Path to PI config.json")
        pi_btn = QPushButton(self.tr("..."))
        pi_btn.clicked.connect(self._browse_pi_clicked)
        pi_layout.addWidget(self.pi_config_path)
        pi_layout.addWidget(pi_btn)
        self.int_panel.add_widget(self.pi_row)

        # Максимальный контекст: авто (0) или ручное значение (токены).
        # При 0 значение подтягивается с сервера (GET /slots -> n_ctx) в main.py.
        ctx_layout = QHBoxLayout()
        ctx_layout.addWidget(QLabel(self.tr("Max context (tokens, 0=auto):")))
        self.integration_max_context = QSpinBox()
        self.integration_max_context.setRange(0, 2_000_000)
        self.integration_max_context.setValue(0)
        self.integration_max_context.setSingleStep(1024)
        self.integration_max_context.setToolTip(
            self.tr(
                "Размер окна контекста сервера. 0 — авто (считывается с "
                "запущенного сервера). Агент будет корректно сжимать контекст."
            )
        )
        ctx_layout.addWidget(self.integration_max_context)
        self.int_panel.add_layout(ctx_layout)

        tgt_layout = QHBoxLayout()
        tgt_layout.addWidget(QLabel(self.tr("Target:")))
        self.integration_target = QComboBox()
        self.integration_target.addItem("OpenCode", "opencode")
        self.integration_target.addItem("PI", "pi")
        tgt_layout.addWidget(self.integration_target)
        self.integration_check_btn = QPushButton(self.tr("Check"))
        tgt_layout.addWidget(self.integration_check_btn)
        self.int_panel.add_layout(tgt_layout)

        self.integration_target.currentIndexChanged.connect(
            self._on_integration_target_changed
        )
        self._on_integration_target_changed()

        self.integration_model_label = QLabel(
            "Model to add: not selected", wordWrap=True
        )
        self.int_panel.add_widget(self.integration_model_label)

        self.integration_models_list = QListWidget()
        self.integration_models_list.setMinimumHeight(80)
        self.int_panel.add_widget(self.integration_models_list)

        act_layout = QHBoxLayout()
        self.integration_add_btn = QPushButton(self.tr("Add"))
        self.integration_remove_btn = QPushButton(self.tr("Remove"))
        act_layout.addWidget(self.integration_add_btn)
        act_layout.addWidget(self.integration_remove_btn)
        self.int_panel.add_layout(act_layout)

        self.integration_status = QLabel(
            "Specify config path and click Check", wordWrap=True
        )
        self.int_panel.add_widget(self.integration_status)

    def _on_integration_target_changed(self):
        """Показываем только поле пути, соответствующее выбранному Target."""
        target = self.integration_target.currentData() or "opencode"
        self.opencode_row.setVisible(target == "opencode")
        self.pi_row.setVisible(target == "pi")

    def _build_benchmark_section(self):
        # === 6. Бенчмарк ===
        self.bench_panel = CollapsiblePanel(
            self.tr("Benchmark"),
            settings_key="panel_benchmark",
            collapsible=False,
        )
        bp_layout = QHBoxLayout()
        bp_layout.addWidget(QLabel(self.tr("Prompt (-p):")))
        self.bench_prompt = QSpinBox()
        self.bench_prompt.setRange(16, 4096)
        self.bench_prompt.setValue(128)
        self.bench_prompt.setSingleStep(64)
        bp_layout.addWidget(self.bench_prompt)
        bp_layout.addSpacing(10)
        bp_layout.addWidget(QLabel(self.tr("Gen (-n):")))
        self.bench_gen = QSpinBox()
        self.bench_gen.setRange(16, 4096)
        self.bench_gen.setValue(256)
        self.bench_gen.setSingleStep(64)
        bp_layout.addWidget(self.bench_gen)
        self.bench_panel.add_layout(bp_layout)

        bench_buttons = QHBoxLayout()
        self.test_btn = QPushButton(self.tr("Test Speed"))
        self.test_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 6px;"
        )
        bench_buttons.addWidget(self.test_btn)
        self.bench_panel.add_layout(bench_buttons)

    def _build_cli_section(self):
        # === 7. Preview CLI ===
        self.cli_group = QGroupBox(self.tr("CLI Preview"))
        g_cli = self.cli_group
        cli_layout = QVBoxLayout()
        cli_controls = QHBoxLayout()
        self.cli_manual_mode = QCheckBox(self.tr("Edit CLI"))
        self.cli_manual_mode.setToolTip(
            "Enable direct command editing. Apply CLI parses known flags back into UI and keeps unknown flags in Extra params."
        )
        self.cli_apply_btn = QPushButton(self.tr("Apply CLI"))
        self.cli_apply_btn.setEnabled(False)
        self.cli_apply_btn.setToolTip(
            "Parse the edited command: known flags update UI controls, unknown flags go to Extra params."
        )
        self.cli_copy_btn = QPushButton(self.tr("Copy CLI"))
        self.cli_copy_btn.setToolTip("Copy the current generated command line.")
        self.cli_import_btn = QPushButton(self.tr("Import CLI"))
        self.cli_import_btn.setToolTip(
            "Read a llama-server command line from the clipboard and apply it to settings."
        )
        self.cli_status = QLabel(self.tr("Generated from UI"))
        self.cli_status.setStyleSheet("color: " + STATUS_COLOR_MUTED + ";")
        cli_controls.addWidget(self.cli_manual_mode)
        cli_controls.addWidget(self.cli_apply_btn)
        cli_controls.addWidget(self.cli_copy_btn)
        cli_controls.addWidget(self.cli_import_btn)
        cli_controls.addWidget(self.cli_status, 1)
        self.cli_preview = QTextEdit()
        self.cli_preview.setReadOnly(True)
        self.cli_preview.setPlaceholderText("Command will be displayed here...")
        self.cli_preview.setMinimumHeight(60)
        self.cli_preview.setMaximumHeight(100)
        self.cli_preview.setFont(QFont("Consolas", 9))
        self.cli_preview.setStyleSheet(
            "background-color: #2a2a2a; color: #b5cea8; padding: 4px;"
        )
        g_cli.setLayout(cli_layout)
        cli_layout.addLayout(cli_controls)
        cli_layout.addWidget(self.cli_preview)

    def _scroll_page(self):
        """Возвращает (QScrollArea, content_layout) для прокручиваемой страницы."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        scroll.setWidget(inner)
        return scroll, lay

    def _build_pages(self):
        """QStackedWidget со страницами в порядке NAV_PAGES."""
        stack = QStackedWidget()
        stack.addWidget(self._dashboard_page())
        stack.addWidget(self._paths_page())
        stack.addWidget(self._performance_page())
        stack.addWidget(self._sampling_page())
        stack.addWidget(self._server_page())
        stack.addWidget(self._library_page())
        stack.addWidget(self._integration_page())
        stack.addWidget(self._benchmark_page())
        stack.addWidget(self._autotune_page())
        return stack

    def _dashboard_page(self):
        page, lay = self._scroll_page()
        lay.addWidget(self.overview_content_widget)
        lay.addWidget(self.runtime_stats_group)
        lay.addWidget(self.launch_summary_group)
        lay.addStretch()
        return page

    def _paths_page(self):
        page, lay = self._scroll_page()
        lay.addWidget(self.paths_panel)
        lay.addStretch()
        return page

    def _performance_page(self):
        page, lay = self._scroll_page()
        lay.addWidget(self.model_group)
        lay.addWidget(self.g_launch)
        lay.addWidget(self.cli_group)
        lay.addStretch()
        return page

    def _sampling_page(self):
        page, lay = self._scroll_page()
        lay.addWidget(self.sampling_panel)
        lay.addWidget(self.adv_panel)
        lay.addStretch()
        return page

    def _server_page(self):
        page, lay = self._scroll_page()
        lay.addWidget(self.server_panel)
        lay.addStretch()
        return page

    def _library_page(self):
        page, lay = self._scroll_page()
        lay.addWidget(self.models_panel)
        lay.addStretch()
        return page

    def _integration_page(self):
        page, lay = self._scroll_page()
        lay.addWidget(self.int_panel)
        lay.addStretch()
        return page

    def _benchmark_page(self):
        page, lay = self._scroll_page()
        lay.addWidget(self.bench_panel)
        lay.addStretch()
        return page

    def _autotune_page(self):
        page, lay = self._scroll_page()
        lay.addWidget(self.autotune)
        lay.addStretch()
        return page

    def _on_nav_selected(self, index: int):
        self.pages.setCurrentIndex(index)
        self.ui_settings.setValue("navIndex", index)

    def _build_overview_content(self):
        """Карточки оперативного обзора (бывшая вкладка Overview)."""
        widget = QWidget()
        overview = QVBoxLayout(widget)
        overview.setContentsMargins(8, 8, 8, 8)
        overview.setSpacing(10)

        self.overview_status = QLabel(self.tr("○ Server stopped"))
        self.overview_status.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.overview_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        overview.addWidget(self.overview_status)

        self.overview_model = QLabel(self.tr("No model selected"))
        self.overview_model.setWordWrap(True)
        self.overview_model.setStyleSheet("color: #555;")
        self.overview_model.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        overview.addWidget(self.overview_model)

        def make_metric_card(title: str, value: str = "-", detail: str = ""):
            card = QGroupBox(title)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 16, 10, 10)
            value_label = QLabel(value)
            value_label.setStyleSheet("font-size: 20px; font-weight: bold;")
            value_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            detail_label = QLabel(detail)
            detail_label.setWordWrap(True)
            detail_label.setStyleSheet("color: " + STATUS_COLOR_MUTED_DARK + ";")
            detail_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            card_layout.addWidget(value_label)
            card_layout.addWidget(detail_label)
            return card, value_label, detail_label

        metric_grid = QGridLayout()
        metric_grid.setHorizontalSpacing(10)
        metric_grid.setVerticalSpacing(10)
        card, self.overview_speed_value, self.overview_speed_detail = make_metric_card(
            "Generation", "-", "PP / TG speed"
        )
        metric_grid.addWidget(card, 0, 0)
        card, self.overview_vram_value, self.overview_vram_detail = make_metric_card(
            "Memory", "-", "VRAM"
        )
        metric_grid.addWidget(card, 0, 1)
        card, self.overview_request_value, self.overview_request_detail = (
            make_metric_card("Request", "-", "Current tokens")
        )
        metric_grid.addWidget(card, 0, 2)
        card, self.overview_context_value, self.overview_context_detail = (
            make_metric_card("Context", "-", "Configured window")
        )
        metric_grid.addWidget(card, 1, 0)
        card, self.overview_active_value, self.overview_active_detail = (
            make_metric_card("Active", "0:00", "Model work time")
        )
        metric_grid.addWidget(card, 1, 1)
        card, self.overview_endpoint_value, self.overview_endpoint_detail = (
            make_metric_card("Endpoint", "-", "OpenAI-compatible base URL")
        )
        metric_grid.addWidget(card, 1, 2)
        overview.addLayout(metric_grid)

        self.overview_settings = QLabel(self.tr("Settings: -"))
        self.overview_settings.setWordWrap(True)
        self.overview_settings.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        overview.addWidget(self.overview_settings)
        self.overview_memory_note = QLabel(self.tr(""))
        self.overview_memory_note.setWordWrap(True)
        self.overview_memory_note.setStyleSheet("color: #777;")
        overview.addWidget(self.overview_memory_note)
        overview.addStretch(1)
        return widget

    def _apply_log_maximize(self, on: bool) -> None:
        """Hide (on=True) or show (on=False) the content area so the log dock
        takes full height. Docked sizes are remembered for restore. State is
        persisted via ``save_ui_state``."""
        if on == self._log_maximized:
            return
        if on:
            self._main_vsplit_docked = self.main_vsplit.sizes()
            self.content_splitter.setVisible(False)
        else:
            self.content_splitter.setVisible(True)
            self.main_vsplit.setSizes(self._main_vsplit_docked)
        self._log_maximized = on
        self.log_dock.set_maximized(on)

    def _toggle_log_maximize(self) -> None:
        self._apply_log_maximize(not self._log_maximized)

    def _build_status_bar(self):
        """Компактная полоса статуса поверх вкладок — видна всегда."""
        bar = QFrame()
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        bar.setLineWidth(1)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)

        self.status_indicator = QLabel("○")
        self.status_indicator.setStyleSheet("font-size: 14px;")
        self.status_short = QLabel(self.tr("Server stopped"))
        self.status_short.setStyleSheet("font-weight: bold;")
        self.status_speed = QLabel("—")
        self.status_vram = QLabel("—")

        lay.addWidget(self.status_indicator)
        lay.addWidget(self.status_short)
        lay.addStretch(1)
        lay.addWidget(QLabel(self.tr("Speed:")))
        lay.addWidget(self.status_speed)
        lay.addWidget(QLabel(self.tr("VRAM:")))
        lay.addWidget(self.status_vram)
        return bar

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
            self.speculative_mtp: "Enable MTP speculative decoding when the selected model/package supports it",
            self.spec_draft_model_path: "Separate MTP/draft GGUF path, auto-detected when possible",
            self.reload_btn: "Restarts llama-server with current parameters",
            self.stop_btn: "Stops server",
            self.force_stop_btn: "Force kills llama-server immediately",
            self.autoscroll_logs: "Auto-scroll logs",
            self.tokens_reset_btn: "Save current task token count to Saved and "
            "reset task counter, Current time and Request label",
            self.reset_session_btn: "Zero all live runtime stats: total/task tokens, "
            "Active/Current time, Request label. Saved history is kept.",
            self.reset_saved_btn: "Zero the accumulated Saved history (last and total).",
            self.export_stats_btn: "Export current runtime counters to JSON.",
            self.copy_stats_md_btn: "Copy current runtime counters as Markdown.",
            self.active_time_label: self.tr(
                "Total model processing time since server start or last "
                "Reset session (prompt processing + token generation). "
                "Idle time and queue waiting are not counted."
            ),
            self.current_time_label: self.tr(
                "Time of the last completed request (prompt processing + "
                "token generation). Value comes from llama_print_timings."
            ),
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
        splitter_state = self.ui_settings.value("contentSplitterState")
        if splitter_state and hasattr(self, "content_splitter"):
            self.content_splitter.restoreState(splitter_state)
        vstate = self.ui_settings.value("mainVSplitterSizes")
        if vstate and hasattr(self, "main_vsplit"):
            try:
                self.main_vsplit.setSizes([int(x) for x in vstate])
            except (TypeError, ValueError):
                pass
        if self.ui_settings.value("logDockMaximized", False, type=bool):
            self._apply_log_maximize(True)
        nav_index = self.ui_settings.value("navIndex")
        if nav_index is not None and hasattr(self, "nav_rail"):
            try:
                self.nav_rail.setCurrentRow(int(nav_index))
            except (TypeError, ValueError):
                pass

    def save_ui_state(self):
        self.ui_settings.setValue("geometry", self.saveGeometry())
        self.ui_settings.setValue("windowState", self.saveState())
        if hasattr(self, "content_splitter"):
            self.ui_settings.setValue(
                "contentSplitterState", self.content_splitter.saveState()
            )
        if hasattr(self, "main_vsplit"):
            sizes = (
                self._main_vsplit_docked
                if getattr(self, "_log_maximized", False)
                else self.main_vsplit.sizes()
            )
            self.ui_settings.setValue("mainVSplitterSizes", sizes)
        if hasattr(self, "log_dock"):
            self.ui_settings.setValue(
                "logDockMaximized", bool(getattr(self, "_log_maximized", False))
            )
        if hasattr(self, "nav_rail"):
            self.ui_settings.setValue("navIndex", self.nav_rail.currentRow())

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
        target = self.current_config_target()
        if target == "pi":
            return self.pi_config_path.text().strip()
        return self.opencode_config_path.text().strip()

    def current_base_url(self):
        return f"http://127.0.0.1:{self.port.value()}/v1"

    def current_model_id(self):
        p = self.model_combo.currentData()
        if p:
            return Path(p).stem
        t = self.model_combo.currentText().strip()
        return Path(t).stem if t.lower().endswith(".gguf") and os.path.exists(t) else t
