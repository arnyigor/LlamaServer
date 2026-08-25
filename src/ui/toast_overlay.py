# src/ui/toast_overlay.py
"""Всплывающее уведомление (toast), переиспользующее сообщения лога.

Минималистичный оверлей: QLabel без рамки, прозрачный для мыши, закреплён
сверху по центру родительского виджета. Новое сообщение заменяет текущее
(таймер перезапускается) — стек сообщений не накапливается. Исчезает плавным
затуханием через QPropertyAnimation.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPropertyAnimation, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QGraphicsOpacityEffect

# Цвета синхронизированы с _LEVEL_COLORS в log_manager.py.
_TOAST_COLORS = {
    "error": QColor("#f48771"),
    "warn": QColor("#dcdcaa"),
    "bench": QColor("#4ec9b0"),
    "info": QColor("#d4d4d4"),
}

_MAX_TEXT = 200


class ToastOverlay(QLabel):
    """Транзиентное уведомление поверх контента окна."""

    def __init__(self, parent: QLabel | None = None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMargin(10)
        self.setWindowFlags(Qt.WindowType.SubWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)
        self._anim: QPropertyAnimation | None = None

        self.hide()
        if parent is not None:
            parent.installEventFilter(self)

    # -- позиционирование ---------------------------------------------------
    def eventFilter(self, obj, event):
        if (
            obj is self.parent()
            and event.type() == QEvent.Type.Resize
            and self.isVisible()
        ):
            self._reposition()
        return super().eventFilter(obj, event)

    def _reposition(self) -> None:
        parent = self.parent()
        if parent is None:
            return
        pw = parent.width()
        w = min(460, pw - 40)
        self.setFixedSize(w, self.sizeHint().height())
        self.move((pw - w) // 2, 14)

    # -- публичный API ------------------------------------------------------
    def show_message(self, text: str, level: str = "info", msec: int = 4000) -> None:
        if not text:
            return
        if len(text) > _MAX_TEXT:
            text = text[: _MAX_TEXT - 1].rstrip() + "…"

        color = _TOAST_COLORS.get(level, _TOAST_COLORS["info"])
        self.setText(text)
        self.setStyleSheet(
            "QLabel{color:%s;background:rgba(40,40,46,0.94);"
            "border:1px solid %s;border-radius:8px;padding:8px 14px;}"
            % (color.name(), color.name())
        )
        # Сбрасываем затухание от предыдущего показа.
        if self._anim is not None:
            self._anim.stop()
        self._effect.setOpacity(1.0)
        self._reposition()
        self.show()
        self.raise_()
        self._timer.start(msec)

    # -- затухание ----------------------------------------------------------
    def _fade_out(self) -> None:
        self._anim = QPropertyAnimation(self._effect, b"opacity")
        self._anim.setDuration(350)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self.hide)
        self._anim.start()
