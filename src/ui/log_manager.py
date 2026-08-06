# src/ui/log_manager.py
"""Менеджер логов с буферизацией, цветами и ограничением памяти."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Deque, Optional

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QTextEdit

from src.core.constants import MAX_LOG_LINES


@dataclass(frozen=True, slots=True)
class LogEntry:
    """Единица лога — неизменяемая."""

    text: str
    level: str


_AUTO_LEVEL_PATTERNS = [
    (re.compile(r"tok/s|tokens/s", re.I), "bench"),
    (re.compile(r"error|failed|exception", re.I), "error"),
    (re.compile(r"warning|warn\b", re.I), "warn"),
    (re.compile(r"server started|running on", re.I), "info"),
]

_LEVEL_COLORS = {
    "error": QColor("#f48771"),
    "warn": QColor("#dcdcaa"),
    "bench": QColor("#4ec9b0"),
    "info": QColor("#d4d4d4"),
}

_CONTENT_COLORS = [
    (re.compile(r"loading model|load time", re.I), QColor("#4ec9b0")),
    (re.compile(r"server started|llama server", re.I), QColor("#b5cea8")),
    (re.compile(r"slot \d+|new request", re.I), QColor("#9cdcfe")),
    (re.compile(r"✅|успешно|success", re.I), QColor("#b5cea8")),
    (re.compile(r"⚠️|warning", re.I), QColor("#dcdcaa")),
]

_PP_SPEED_PATTERN = re.compile(
    r"prompt (?:eval time|processing).*?"
    r"(\d+(?:\.\d+)?)\s*tokens (?:per second|/s)",
    re.I,
)

_TG_SPEED_PATTERN = re.compile(
    r"(?<!prompt )eval time.*?"
    r"(\d+(?:\.\d+)?)\s*tokens (?:per second|/s)"
    r"|\btg\b\s*=\s*(\d+(?:\.\d+)?)\s*t/s"
    r"|(\d+(?:\.\d+)?)\s*tok/s",
    re.I,
)


@lru_cache(maxsize=64)
def _get_format_cached(level: str, content_key: str) -> QTextCharFormat:
    """LRU-кэш форматов — автоматически вытесняет старые."""
    fmt = QTextCharFormat()
    color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["info"])

    if level == "info" and content_key:
        for pattern, content_color in _CONTENT_COLORS:
            if pattern.search(content_key):
                color = content_color
                break

    fmt.setForeground(color)
    return fmt


def _get_format(level: str, text: str = "") -> QTextCharFormat:
    content_key = text[:20].strip().lower() if text else ""
    return _get_format_cached(level, content_key)


class LogManager(QObject):
    """Менеджер логов с буферизацией и ограничением памяти."""

    speed_updated = Signal(str)

    _MAX_BUFFER = 500

    def __init__(self, text_edit: QTextEdit, flush_interval_ms: int = 80):
        super().__init__()
        self._edit = text_edit
        self._buffer: Deque[LogEntry] = deque(maxlen=self._MAX_BUFFER)
        self._line_count = 0

        self._timer = QTimer(self)
        self._timer.setInterval(flush_interval_ms)
        self._timer.timeout.connect(self._flush)
        self._timer.start()

        self._autoscroll = True
        self._pp_speed: Optional[float] = None
        self._tg_speed: Optional[float] = None

        sb = text_edit.verticalScrollBar()
        sb.sliderPressed.connect(self._on_user_scroll)

    @property
    def autoscroll(self) -> bool:
        return self._autoscroll

    @autoscroll.setter
    def autoscroll(self, value: bool) -> None:
        self._autoscroll = value

    def append(self, text: str, level: str = "info") -> None:
        if not text or not text.strip():
            return
        if level == "info":
            for pattern, auto_level in _AUTO_LEVEL_PATTERNS:
                if pattern.search(text):
                    level = auto_level
                    break

        self._buffer.append(LogEntry(text=text, level=level))
        self._extract_speed(text)

    def clear(self) -> None:
        self._buffer.clear()
        self._edit.clear()
        self._line_count = 0
        self._pp_speed = None
        self._tg_speed = None
        self.speed_updated.emit("Speed: -")

    def stop(self) -> None:
        self._timer.stop()
        self._flush()

    def _on_user_scroll(self) -> None:
        sb = self._edit.verticalScrollBar()
        self._autoscroll = sb.value() >= sb.maximum() - 50

    def _flush(self) -> None:
        if not self._buffer:
            return

        entries = list(self._buffer)
        self._buffer.clear()

        cursor = self._edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._edit.setUpdatesEnabled(False)

        try:
            for entry in entries:
                fmt = _get_format(entry.level, entry.text)
                text = entry.text

                # Обрабатываем точки прогресса: если текущий текст — только точки
                stripped = text.strip()
                if stripped and all(c in ". " for c in stripped):
                    # Точки прогресса — вставляем без переноса строки, рядом с предыдущим текстом
                    cursor.insertText(stripped.replace(" ", ""), fmt)
                else:
                    text = text if text.endswith("\n") else text + "\n"
                    cursor.insertText(text, fmt)
                    self._line_count += text.count("\n")

            if self._line_count > MAX_LOG_LINES:
                self._trim_old_lines()
        finally:
            self._edit.setUpdatesEnabled(True)

        if self._autoscroll:
            sb = self._edit.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _trim_old_lines(self) -> None:
        excess = self._line_count - MAX_LOG_LINES
        if excess <= 0:
            return
        trim_cursor = self._edit.textCursor()
        trim_cursor.movePosition(QTextCursor.MoveOperation.Start)
        trim_cursor.movePosition(
            QTextCursor.MoveOperation.Down,
            QTextCursor.MoveMode.KeepAnchor,
            excess,
        )
        trim_cursor.removeSelectedText()
        self._line_count = MAX_LOG_LINES

    def _extract_speed(self, text: str) -> None:
        changed = False

        pp_match = _PP_SPEED_PATTERN.search(text)
        if pp_match:
            self._pp_speed = float(pp_match.group(1))
            changed = True

        tg_match = _TG_SPEED_PATTERN.search(text)
        if tg_match:
            value = next((float(g) for g in tg_match.groups() if g), None)
            if value is not None:
                self._tg_speed = value
                changed = True

        if changed:
            self._emit_speed()

    def _emit_speed(self) -> None:
        parts = []
        if self._pp_speed is not None:
            parts.append(f"PP: {self._pp_speed:.1f} tok/s")
        if self._tg_speed is not None:
            parts.append(f"TG: {self._tg_speed:.1f} tok/s")
        if parts:
            self.speed_updated.emit(" / ".join(parts))
