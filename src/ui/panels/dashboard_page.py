"""Dashboard nav page: live overview metric cards, runtime stats group and the
launch preflight summary. Built as a self-contained ``QScrollArea`` page; widget
references are created on ``mw`` (MainWindowUI) so ``main.py`` keeps working
unchanged.
"""

from PySide6.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QGridLayout,
    QProgressBar,
    QPushButton,
)
from PySide6.QtCore import Qt

from src.core.constants import STATUS_COLOR_MUTED_DARK


class DashboardPage(QScrollArea):
    """Dashboard page: overview cards + runtime stats + launch preflight."""

    def __init__(self, mw):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        self.setWidget(inner)

        self._build_overview(mw, lay)
        self._build_runtime_stats(mw, lay)
        # launch_summary_group is built by GenerationBuilder (runs before this).
        if getattr(mw, "launch_summary_group", None) is not None:
            lay.addWidget(mw.launch_summary_group)
        lay.addStretch(1)

    def _build_overview(self, mw, parent):
        widget = QWidget()
        overview = QVBoxLayout(widget)
        overview.setContentsMargins(8, 8, 8, 8)
        overview.setSpacing(10)

        mw.overview_status = QLabel(mw.tr("○ Server stopped"))
        mw.overview_status.setStyleSheet("font-size: 16px; font-weight: bold;")
        mw.overview_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        overview.addWidget(mw.overview_status)

        mw.overview_model = QLabel(mw.tr("No model selected"))
        mw.overview_model.setWordWrap(True)
        mw.overview_model.setStyleSheet("color: #555;")
        mw.overview_model.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        overview.addWidget(mw.overview_model)

        mw.overview_load_progress = QProgressBar()
        mw.overview_load_progress.setRange(0, 100)
        mw.overview_load_progress.setTextVisible(True)
        mw.overview_load_progress.setFormat("starting...")
        mw.overview_load_progress.setVisible(False)
        overview.addWidget(mw.overview_load_progress)

        def make_metric_card(title, value="-", detail=""):
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
        card, mw.overview_speed_value, mw.overview_speed_detail = make_metric_card(
            "Generation", "-", "PP / TG speed"
        )
        metric_grid.addWidget(card, 0, 0)
        card, mw.overview_vram_value, mw.overview_vram_detail = make_metric_card(
            "Memory", "-", "VRAM"
        )
        metric_grid.addWidget(card, 0, 1)
        card, mw.overview_request_value, mw.overview_request_detail = make_metric_card(
            "Request", "-", "Current tokens"
        )
        metric_grid.addWidget(card, 0, 2)
        card, mw.overview_context_value, mw.overview_context_detail = make_metric_card(
            "Context", "-", "Configured window"
        )
        metric_grid.addWidget(card, 1, 0)
        card, mw.overview_active_value, mw.overview_active_detail = make_metric_card(
            "Active", "0:00", "Model work time"
        )
        metric_grid.addWidget(card, 1, 1)
        card, mw.overview_endpoint_value, mw.overview_endpoint_detail = (
            make_metric_card("Endpoint", "-", "OpenAI-compatible base URL")
        )
        metric_grid.addWidget(card, 1, 2)
        overview.addLayout(metric_grid)

        mw.overview_settings = QLabel(mw.tr("Settings: -"))
        mw.overview_settings.setWordWrap(True)
        mw.overview_settings.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        overview.addWidget(mw.overview_settings)
        mw.overview_memory_note = QLabel(mw.tr(""))
        mw.overview_memory_note.setWordWrap(True)
        mw.overview_memory_note.setStyleSheet("color: #777;")
        overview.addWidget(mw.overview_memory_note)
        overview.addStretch(1)
        mw.overview_content_widget = widget
        parent.addWidget(mw.overview_content_widget)

    def _build_runtime_stats(self, mw, parent):
        mw.runtime_stats_group = QGroupBox(mw.tr("Runtime stats"))
        stats = QGridLayout(mw.runtime_stats_group)
        stats.setContentsMargins(12, 18, 12, 12)
        stats.setHorizontalSpacing(10)
        stats.setVerticalSpacing(6)
        mw.speed_label = QLabel(mw.tr("Speed: -"))
        mw.speed_label.setTextFormat(Qt.RichText)
        mw.speed_label.setProperty("class", "mono")
        mw.tokens_label = QLabel(mw.tr("Tokens: total 0 | task 0"))
        mw.tokens_label.setTextFormat(Qt.RichText)
        mw.tokens_label.setProperty("class", "mono")
        mw.request_tokens_label = QLabel(mw.tr("Request: -"))
        mw.request_tokens_label.setTextFormat(Qt.RichText)
        mw.request_tokens_label.setProperty("class", "mono")
        mw.tokens_saved_label = QLabel(mw.tr("Saved: 0"))
        mw.tokens_saved_label.setTextFormat(Qt.RichText)
        mw.tokens_saved_label.setProperty("class", "mono")
        mw.active_time_label = QLabel(mw.tr("Work time: 0:00 (Prompt 0:00 | Gen 0:00)"))
        mw.active_time_label.setTextFormat(Qt.RichText)
        mw.active_time_label.setProperty("class", "mono")
        mw.current_time_label = QLabel(
            mw.tr("Last request: 0:00 (Prompt 0:00 | Gen 0:00)")
        )
        mw.current_time_label.setTextFormat(Qt.RichText)
        mw.current_time_label.setProperty("class", "mono")
        mw.tokens_reset_btn = QPushButton(mw.tr("Save task & reset"))
        mw.tokens_reset_btn.setToolTip(
            "Save current task token count to Saved and start the next task from zero. "
            "Resets the task counter, Current time and Request label."
        )
        mw.export_stats_btn = QPushButton(mw.tr("Export stats"))
        mw.export_stats_btn.setToolTip(
            "Export current runtime counters to a JSON file."
        )
        mw.copy_stats_md_btn = QPushButton(mw.tr("Copy stats MD"))
        mw.copy_stats_md_btn.setToolTip(
            "Copy current runtime counters to the clipboard as Markdown."
        )
        mw.reset_session_btn = QPushButton(mw.tr("Reset session"))
        mw.reset_session_btn.setToolTip(
            "Zero all live runtime stats: total/task tokens, prompt/generated, "
            "Active and Current time, Request label. Saved history is kept."
        )
        mw.reset_saved_btn = QPushButton(mw.tr("Reset saved"))
        mw.reset_saved_btn.setToolTip(
            "Zero the accumulated Saved history (last and total)."
        )
        for btn in [
            mw.tokens_reset_btn,
            mw.export_stats_btn,
            mw.copy_stats_md_btn,
            mw.reset_session_btn,
            mw.reset_saved_btn,
        ]:
            btn.setMinimumWidth(120)
            btn.setMaximumWidth(150)

        stats.addWidget(mw.speed_label, 0, 0, 1, 3)
        stats.addWidget(mw.tokens_label, 1, 0, 1, 2)
        stats.addWidget(mw.tokens_reset_btn, 1, 2)
        stats.addWidget(mw.request_tokens_label, 2, 0)
        stats.addWidget(mw.tokens_saved_label, 2, 1)
        stats.addWidget(mw.reset_saved_btn, 2, 2)
        stats.addWidget(mw.active_time_label, 3, 0, 1, 2)
        stats.addWidget(mw.reset_session_btn, 3, 2)
        stats.addWidget(mw.current_time_label, 4, 0, 1, 2)
        stats.addWidget(mw.export_stats_btn, 4, 2)
        stats.addWidget(mw.copy_stats_md_btn, 5, 2)
        stats.setColumnStretch(1, 1)
        parent.addWidget(mw.runtime_stats_group)
