"""Bottom log dock widget (Phase 2).

Extracts the previously-inline log area into a dedicated, resizable widget with
a maximize toggle. Attribute names expected by ``main.py``
(``logs``, ``autoscroll_logs``, ``copy_last_error_btn``, ``open_diagnostics_btn``)
are preserved so the host window can simply re-export them.
"""

from __future__ import annotations

from PySide6.QtCore import Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QPushButton,
    QTextEdit,
)


class LogDock(QWidget):
    """Resizable bottom log dock with a maximize toggle.

    The maximize action is owned by the host window (it must hide/show the
    content area), so this widget only emits :attr:`toggle_maximize` and exposes
    :meth:`set_maximized` to reflect state on the button label.
    """

    toggle_maximize = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(self.tr("Logs:")))
        self.autoscroll_logs = QCheckBox(self.tr("Auto-scroll"), checked=True)
        hdr.addWidget(self.autoscroll_logs)
        hdr.addStretch(1)
        self.copy_logs_btn = QPushButton(self.tr("Copy logs"))
        self.copy_last_error_btn = QPushButton(self.tr("Copy last error"))
        self.copy_last_error_btn.setEnabled(False)
        self.open_diagnostics_btn = QPushButton(self.tr("Open diagnostics"))
        self.maximize_btn = QPushButton(self.tr("Maximize"))
        self.clear_btn = QPushButton(self.tr("Clear"))
        hdr.addWidget(self.copy_logs_btn)
        hdr.addWidget(self.copy_last_error_btn)
        hdr.addWidget(self.open_diagnostics_btn)
        hdr.addWidget(self.maximize_btn)
        hdr.addWidget(self.clear_btn)
        layout.addLayout(hdr)

        self.logs = QTextEdit(readOnly=True, font=QFont("Consolas", 9))
        self.logs.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        layout.addWidget(self.logs, 1)

        self.clear_btn.clicked.connect(self.logs.clear)
        self.maximize_btn.clicked.connect(self.toggle_maximize.emit)
        self.copy_logs_btn.clicked.connect(self._copy_logs)

    def _copy_logs(self) -> None:
        QApplication.clipboard().setText(self.logs.toPlainText())
        self.copy_logs_btn.setText(self.tr("Copied"))
        QTimer.singleShot(1500, lambda: self.copy_logs_btn.setText(self.tr("Copy logs")))

    def set_maximized(self, on: bool) -> None:
        """Update the maximize button label to reflect dock state."""
        self.maximize_btn.setText(self.tr("Restore") if on else self.tr("Maximize"))
