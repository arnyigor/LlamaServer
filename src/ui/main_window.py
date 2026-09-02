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
    STATUS_COLOR_MUTED,
    STATUS_COLOR_MUTED_DARK,
    STATUS_COLOR_PENDING,
    STATUS_COLOR_READY,
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
from src.ui.header_bar import HeaderBar
from src.ui.log_dock import LogDock
from src.ui.control_strip import ControlStrip
from src.ui.nav_rail import NavRail
from src.ui.toast_overlay import ToastOverlay
from src.ui.panels.generation_page import GenerationBuilder
from src.ui.panels.dashboard_page import DashboardPage
from src.ui.panels.library_page import ModelLibraryPage
from src.ui.panels.integration_page import IntegrationPage
from src.ui.panels.benchmark_page import BenchmarkPage
from src.ui.panels.autotune_page import AutoTunePage


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

        # === Полоса управления сервером (Phase 3: ControlStrip) ===
        self.control_strip = ControlStrip()
        # Реэкспорт кнопок для совместимости с main.py
        self.start_btn = self.control_strip.start_btn
        self.reload_btn = self.control_strip.reload_btn
        self.stop_btn = self.control_strip.stop_btn
        self.force_stop_btn = self.control_strip.force_stop_btn

        # Вертикальный сплиттер: контент + полоса управления + лог-док
        main_vsplit = QSplitter(Qt.Orientation.Vertical)
        main_vsplit.addWidget(self.content_splitter)
        main_vsplit.addWidget(self.control_strip)
        main_vsplit.addWidget(self.log_dock)
        main_vsplit.setStretchFactor(0, 1)
        main_vsplit.setStretchFactor(1, 0)
        main_vsplit.setStretchFactor(2, 0)
        main_vsplit.setSizes([660, 44, 180])
        self.main_vsplit = main_vsplit
        self._main_vsplit_docked = [660, 44, 180]
        self._log_maximized = False
        root.addWidget(self.main_vsplit, 1)

        # Стартовая страница
        self.nav_rail.setCurrentRow(0)

        # === Всплывающее уведомление (Phase 5: toast) ===
        # Родитель — центральный виджет, чтобы оверлей следовал за размером
        # окна (перепозиционирование через eventFilter в ToastOverlay).
        self.toast_overlay = ToastOverlay(self.centralWidget())

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
        важен: GenerationBuilder создаёт панели, которые sampling наполняет
        контентом; DashboardPage использует launch_summary_group из него.
        """
        self._build_launch_controls_section()
        self._build_paths_section()

        # Coupled Launch/Sampling/Server/CLI widgets (shared CollapsiblePanels).
        self.generation = GenerationBuilder(self)

        # Independent nav pages (build their own widgets, re-export to self).
        self.model_library_page = ModelLibraryPage(self)
        self.integration_page = IntegrationPage(self)
        self.benchmark_page = BenchmarkPage(self)
        self.dashboard_page = DashboardPage(self)
        self.autotune_page = AutoTunePage(self)

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

        # Basic/Advanced (Этап 3.2): скрывает продвинутые панели разом.
        # Серверные кнопки (Start/Restart/Stop/Force Stop) перенесены в
        # ControlStrip — нижняя полоса управления (Этап 3).
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

    def _on_integration_target_changed(self):
        """Показываем только поле пути, соответствующее выбранному Target."""
        target = self.integration_target.currentData() or "opencode"
        self.opencode_row.setVisible(target == "opencode")
        self.pi_row.setVisible(target == "pi")

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
        """QStackedWidget со страницами в порядке NAV_PAGES.

        Независимые страницы (Dashboard/Library/Integration/Benchmark/AutoTune)
        — это готовые QWidget-экземпляры, созданные в ``_build_all_widgets``.
        Launch/Sampling/Server собираются из виджетов, созданных
        GenerationBuilder, в отдельных контейнерах.
        """
        stack = QStackedWidget()
        stack.addWidget(self.dashboard_page)
        stack.addWidget(self._paths_page())
        stack.addWidget(self._performance_page())
        stack.addWidget(self._sampling_page())
        stack.addWidget(self._server_page())
        stack.addWidget(self.model_library_page)
        stack.addWidget(self.integration_page)
        stack.addWidget(self.benchmark_page)
        stack.addWidget(self.autotune_page)
        return stack

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

    def _on_nav_selected(self, index: int):
        # Клик по nav-рейлу — явное намерение открыть страницу; если лог был
        # увеличен, возвращаем обычные пропорции, чтобы страница получила
        # нормальное место, а не узкую полоску.
        if self._log_maximized:
            self._apply_log_maximize(False)
        self.pages.setCurrentIndex(index)
        self.ui_settings.setValue("navIndex", index)

    def set_model_list(self, models: list) -> None:
        """Populate the Launch page's model combo and sync the AutoTune picker.

        Single entry point for a fresh scan result: replaces the previous
        pattern of populating both combos in lockstep from ``main.py``, which
        used to leave the AutoTune combo with a single model if a
        ``currentIndexChanged`` handler fired mid-population.
        """
        self.models = models
        self.models_by_path = {m["path"]: m for m in models}
        self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            for m in models:
                self.model_combo.addItem(m["display"], m["path"])
        finally:
            self.model_combo.blockSignals(False)
        if hasattr(self, "autotune"):
            self.autotune._sync_model_items()

    def _apply_log_maximize(self, on: bool) -> None:
        """Give the log dock most of the vertical space (on=True) or restore
        the previous docked proportions (on=False).

        The content area (nav rail + pages) is intentionally never hidden —
        only shrunk to a small minimum — so the page a user was on is still
        visible and nothing appears to vanish. Docked sizes are remembered
        for restore; state is persisted via ``save_ui_state``.
        """
        if on == self._log_maximized:
            return
        if on:
            self._main_vsplit_docked = self.main_vsplit.sizes()
            total = sum(self._main_vsplit_docked) or 1
            control_h = self._main_vsplit_docked[1] if len(self._main_vsplit_docked) > 1 else 44
            content_h = max(120, int(total * 0.12))
            log_h = max(100, total - content_h - control_h)
            self.main_vsplit.setSizes([content_h, control_h, log_h])
        else:
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
        # Лог-док всегда стартует в обычном режиме: сохранённый maximized-флаг
        # намеренно не восстанавливается, иначе окно открывается с развёрнутым
        # логом и скрытым nav-рейлом/страницами без явного объяснения почему.
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
