"""Иконочный nav-рейл слева (Этап 1).

QListWidget с иконкой + текстом. Выбор строки транслируется в сигнал
page_selected(int). Порядок/ключи страниц задаются вызывающей стороной
(MainWindowUI._build_pages), чтобы nav и страницы QStackedWidget всегда
совпадали. Иконки — стандартные QStyle.StandardPixmap (кроссплатформенно,
без внешних ресурсов).
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QListWidgetItem,
    QStyle,
)


class NavRail(QListWidget):
    page_selected = Signal(int)

    # key -> стандартная иконка
    ICONS = {
        "dashboard": QStyle.StandardPixmap.SP_ComputerIcon,
        "paths": QStyle.StandardPixmap.SP_DriveHDIcon,
        "performance": QStyle.StandardPixmap.SP_MediaPlay,
        "sampling": QStyle.StandardPixmap.SP_DialogResetButton,
        "server": QStyle.StandardPixmap.SP_BrowserReload,
        "library": QStyle.StandardPixmap.SP_DirIcon,
        "integration": QStyle.StandardPixmap.SP_ArrowRight,
        "benchmark": QStyle.StandardPixmap.SP_MediaSeekForward,
        "autotune": QStyle.StandardPixmap.SP_MediaSkipForward,
    }

    def __init__(self, items=None, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(150)
        self.setMaximumWidth(210)
        self.setIconSize(QSize(20, 20))
        self.setUniformItemSizes(True)
        self.currentRowChanged.connect(self._emit_page_selected)
        if items:
            self.set_items(items)

    def set_items(self, items):
        """items: список (label, key)."""
        self.clear()
        style = QApplication.style()
        for label, key in items:
            pixmap = self.ICONS.get(key, QStyle.StandardPixmap.SP_CustomBase)
            icon = style.standardIcon(pixmap) if style is not None else QIcon()
            self.addItem(QListWidgetItem(icon, label))

    def _emit_page_selected(self, row: int):
        self.page_selected.emit(row)
