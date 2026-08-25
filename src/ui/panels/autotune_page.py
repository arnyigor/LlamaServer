"""AutoTune nav page: thin wrapper around the existing ``AutoTuneWidget``.

Built as a self-contained ``QScrollArea`` page; the widget reference is created on
``mw`` (MainWindowUI) so ``main.py`` keeps working unchanged.
"""

from PySide6.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
)

from src.ui.autotune_widget import AutoTuneWidget


class AutoTunePage(QScrollArea):
    """AutoTune page: wrapper over AutoTuneWidget."""

    def __init__(self, mw):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        self.setWidget(inner)
        mw.autotune = AutoTuneWidget()
        lay.addWidget(mw.autotune)
        lay.addStretch()
