"""Bottom server control strip (Phase 3).

Houses the server-lifecycle buttons (Start / Restart / Stop / Force Stop)
previously placed in the top launch-controls row, now reused here and placed
between the content area and the log dock. CPU/RAM/VRAM gauges are omitted:
the app has no system-metrics source (only llama-server ``/slots`` polling),
so live gauges would require adding a metrics provider (e.g. psutil) — out of
scope for this phase.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
)

from src.core.constants import (
    STATUS_COLOR_RUNNING,
    STATUS_COLOR_WARNING,
    STATUS_COLOR_ERROR,
)


class ControlStrip(QWidget):
    """Server control strip: lifecycle buttons, reused from launch controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(44)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)

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
        # Hidden by default; revealed automatically ~5s after Stop if the
        # server has not stopped (see LlamaGUI._reveal_force_stop).
        self.force_stop_btn.setVisible(False)

        lay.addWidget(self.start_btn)
        lay.addWidget(self.reload_btn)
        lay.addWidget(self.stop_btn)
        lay.addWidget(self.force_stop_btn)
        lay.addStretch(1)
