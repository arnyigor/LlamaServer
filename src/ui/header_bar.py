"""Шапка приложения: только выбор языка.

Компактная горизонтальная панель вверху окна. Механизм сохранения настроек
теперь — единственный «Preset» (per-model performance preset), вынесенный в
панель запуска рядом со Start/Stop. Шапка оставлена минимальной: язык
интерфейса. Сигнал language_changed наружу; MainWindowUI реэкспортирует
language_combo для совместимости с main.py.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)


class HeaderBar(QWidget):
    language_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)

        # Лёгкий заголовок слева (брендинг, не интерактивен).
        title = QLabel(self.tr("Llama Server Studio"))
        title.setStyleSheet("font-weight: bold; color: #bbb;")
        lay.addWidget(title)
        lay.addStretch(1)

        # === Язык ===
        self.language_combo = QComboBox()
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Русский", "ru")
        ui_settings = QSettings("LlamaServerGUI", "UIState")
        saved_lang = str(ui_settings.value("language", "en") or "en").strip().lower()
        self.language_combo.setCurrentIndex(
            max(self.language_combo.findData(saved_lang), 0)
        )
        self.language_combo.setToolTip(
            "Interface language. Applied after the application restarts "
            "(or launch with --lang=ru / --lang=en)."
        )
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        lay.addWidget(QLabel(self.tr("Language:")))
        lay.addWidget(self.language_combo)

    def _on_language_changed(self, index: int):
        lang = str(self.language_combo.itemData(index) or "en")
        QSettings("LlamaServerGUI", "UIState").setValue("language", lang)
        self.language_changed.emit(lang)
