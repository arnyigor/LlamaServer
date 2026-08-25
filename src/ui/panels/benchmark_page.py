"""Benchmark nav page.

Built as a self-contained ``QScrollArea`` page; widget references are created on
``mw`` (MainWindowUI) so ``main.py`` keeps working unchanged.
"""

from PySide6.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
)

from src.ui.widgets import CollapsiblePanel


class BenchmarkPage(QScrollArea):
    """Benchmark page: prompt/gen sizes + Test Speed."""

    def __init__(self, mw):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        self.setWidget(inner)
        self._build_benchmark_section(mw)
        lay.addWidget(mw.bench_panel)
        lay.addStretch()

    def _build_benchmark_section(self, mw):
        # === 6. Бенчмарк ===
        mw.bench_panel = CollapsiblePanel(
            mw.tr("Benchmark"),
            settings_key="panel_benchmark",
            collapsible=False,
        )
        bp_layout = QHBoxLayout()
        bp_layout.addWidget(QLabel(mw.tr("Prompt (-p):")))
        mw.bench_prompt = QSpinBox()
        mw.bench_prompt.setRange(16, 4096)
        mw.bench_prompt.setValue(128)
        mw.bench_prompt.setSingleStep(64)
        bp_layout.addWidget(mw.bench_prompt)
        bp_layout.addSpacing(10)
        bp_layout.addWidget(QLabel(mw.tr("Gen (-n):")))
        mw.bench_gen = QSpinBox()
        mw.bench_gen.setRange(16, 4096)
        mw.bench_gen.setValue(256)
        mw.bench_gen.setSingleStep(64)
        bp_layout.addWidget(mw.bench_gen)
        mw.bench_panel.add_layout(bp_layout)

        bench_buttons = QHBoxLayout()
        mw.test_btn = QPushButton(mw.tr("Test Speed"))
        mw.test_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 6px;"
        )
        bench_buttons.addWidget(mw.test_btn)
        mw.bench_panel.add_layout(bench_buttons)
